
import collections
import concurrent
import random
import threading
import math

import numpy as np
import attr
import tqdm

import msprime

import _tsinfer

def split_parent_array(P):
    num_sites = P.shape[0]
    start = 0
    for j in range(num_sites - 1):
        if P[j + 1] != P[j]:
            if P[j] != -1:
                yield start, j + 1, P[j]
            start = j + 1
    if P[-1] != -1:
        yield start, num_sites, P[-1]


def build_ancestors(samples, positions, num_threads=1, method="C", show_progress=False):
    num_samples, num_sites = samples.shape
    if method == "C":
        builder = _tsinfer.AncestorBuilder(samples, positions)
    else:
        builder = AncestorBuilder(samples, positions)
    store_builder = _tsinfer.AncestorStoreBuilder(
            builder.num_sites, 8192 * builder.num_sites)

    frequency_classes = builder.get_frequency_classes()
    total_ancestors = 1
    num_focal_sites = 0
    for age, ancestor_focal_sites in frequency_classes:
        assert len(ancestor_focal_sites) > 0
        total_ancestors += len(ancestor_focal_sites)
        for focal_sites in ancestor_focal_sites:
            assert len(focal_sites) > 0
            num_focal_sites += len(focal_sites)
    ancestor_age = np.zeros(total_ancestors, dtype=np.uint32)
    focal_site_ancestor = np.zeros(num_focal_sites, dtype=np.int32)
    focal_site = np.zeros(num_focal_sites, dtype=np.uint32)

    if show_progress:
        progress = tqdm.tqdm(total=total_ancestors, desc="Build ancestors")

    row = 0
    ancestor_id = 1
    for age, ancestor_focal_sites in frequency_classes:
        num_ancestors = len(ancestor_focal_sites)
        A = np.zeros((num_ancestors, builder.num_sites), dtype=np.int8)
        p = np.zeros(num_ancestors, dtype=np.uint32)
        for j, focal_sites in enumerate(ancestor_focal_sites):
            builder.make_ancestor(focal_sites, A[j, :])
        _tsinfer.sort_ancestors(A, p)
        for index in p:
            store_builder.add(A[index, :])
            for site in ancestor_focal_sites[index]:
                focal_site_ancestor[row] = ancestor_id
                focal_site[row] = site
                row += 1
            ancestor_age[ancestor_id] = age
            ancestor_id += 1
            if show_progress:
                progress.update()
    assert row == num_focal_sites
    assert ancestor_id == total_ancestors

    if show_progress:
        progress.close()

    N = store_builder.total_segments
    site = np.zeros(N, dtype=np.uint32)
    start = np.zeros(N, dtype=np.int32)
    end = np.zeros(N, dtype=np.int32)
    state = np.zeros(N, dtype=np.int8)
    store_builder.dump_segments(site, start, end, state)

    assert np.max(end) == total_ancestors
    assert np.min(start) == 0

    if method == "C":
        store = _tsinfer.AncestorStore(
            position=positions, site=site, start=start, end=end, state=state,
            ancestor_age=ancestor_age, focal_site_ancestor=focal_site_ancestor,
            focal_site=focal_site)
    else:
        store = AncestorStore(
            position=positions, site=site, start=start, end=end, state=state,
            ancestor_age=ancestor_age, focal_site_ancestor=focal_site_ancestor,
            focal_site=focal_site)
    # store.print_state()
    return store

def match_ancestors(
        store, recombination_rate, tree_sequence_builder, num_threads=1, method="C",
        show_progress=False):
    all_ancestors = list(range(1, store.num_ancestors))
    ancestor_age = map(store.get_age, all_ancestors)
    same_age_ancestors = collections.defaultdict(list)
    for ancestor_id, age in zip(all_ancestors, ancestor_age):
        same_age_ancestors[age].append(ancestor_id)

    def match_worker(work):
        start, end, focal_sites, num_older_ancestors, ancestor_id, h = work
        if method == "C":
            matcher = _tsinfer.AncestorMatcher(store, recombination_rate)
            traceback = _tsinfer.Traceback(store.num_sites, 2**10)
        else:
            matcher = AncestorMatcher(store, recombination_rate)
            traceback = Traceback(store)
        end_site_parent = matcher.best_path(
                num_ancestors=num_older_ancestors,
                haplotype=h, start_site=start, end_site=end,
                focal_sites=focal_sites, error_rate=0, traceback=traceback)
        return ancestor_id, h, start, end, end_site_parent, traceback

    if show_progress:
        progress = tqdm.tqdm(total=store.num_ancestors - 1, desc="Match ancestors")
    for age in sorted(same_age_ancestors.keys()):
        ancestor_ids = same_age_ancestors[age]
        # print(age, ancestor_ids)

        work = []
        for ancestor_id in ancestor_ids:
            # print("matching ancestor_id", ancestor_id)
            h = np.zeros(store.num_sites, dtype=np.int8)
            segs = tree_sequence_builder.get_used_segments(ancestor_id)
            start_site, end_site, num_older_ancestors, focal_sites = store.get_ancestor(
                    ancestor_id, h)
            for site in focal_sites:
                assert h[site] == 1
            assert len(focal_sites) > 0
            # work.extend([
            #     (start, end, focal_sites, num_older_ancestors, ancestor_id, h)
            #     for start, end in segs])
        # if num_threads == 1:
            # results = map(match_worker, work)
        # else:
            # with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            #     results = executor.map(match_worker, work)
        # for result in results:
            # ancestor_id, h, start, end, end_site_parent, traceback = result

            for start, end in segs:
                res = match_worker((start, end, focal_sites, num_older_ancestors, ancestor_id, h))
                end_site_parent = res[-2]
                traceback = res[-1]
                tree_sequence_builder.update(
                    child=ancestor_id, haplotype=h, start_site=start,
                    end_site=end, end_site_parent=end_site_parent,
                    traceback=traceback)
            if show_progress:
                progress.update()
    if show_progress:
        progress.close()


def match_samples(
        store, samples, recombination_rate, error_rate, tree_sequence_builder,
        num_threads=1, method="C", show_progress=False):
    sample_ids = list(range(samples.shape[0]))
    update_lock = threading.Lock()

    # FIXME hack --- we insist that error_rate is > 0 for low-level code.
    if error_rate == 0:
        error_rate = 1e-200

    if show_progress:
        progress = tqdm.tqdm(total=samples.shape[0], desc="Match samples  ")

    def sample_match_worker(thread_index):
        chunk_size = int(math.ceil(len(sample_ids) / num_threads))
        start = thread_index * chunk_size
        if method == "C":
            matcher = _tsinfer.AncestorMatcher(store, recombination_rate)
            traceback = _tsinfer.Traceback(store.num_sites, 2**10)
        else:
            matcher = AncestorMatcher(store, recombination_rate)
            traceback = Traceback(store)
        for sample_id in sample_ids[start: start + chunk_size]:
            h = samples[sample_id, :]
            end_site_parent = matcher.best_path(
                    num_ancestors=store.num_ancestors,
                    haplotype=h, start_site=0, end_site=store.num_sites,
                    focal_sites=[], error_rate=error_rate, traceback=traceback)
            with update_lock:
                tree_sequence_builder.update(
                    child=store.num_ancestors + sample_id,
                    haplotype=h, start_site=0, end_site=store.num_sites,
                    end_site_parent=end_site_parent, traceback=traceback)
                if show_progress:
                    progress.update()
            traceback.reset()

    if num_threads > 1:
        threads = [
            threading.Thread(target=sample_match_worker, args=(j,))
            for j in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        sample_match_worker(0)

    if show_progress:
        progress.close()

def finalise_tree_sequence(num_samples, store, position, length, tree_sequence_builder):

    site_position = np.zeros(store.num_sites + 1, dtype=np.float64)
    site_position[:store.num_sites] = position
    site_position[store.num_sites] = length
    # Make sure the first coordinate is 0
    site_position[0] = 0

    # Allocate the nodes table.
    num_nodes = store.num_ancestors + num_samples
    num_ancestors = store.num_ancestors
    flags = np.zeros(num_nodes, dtype=np.uint32)
    flags[num_ancestors:] = msprime.NODE_IS_SAMPLE
    time = np.zeros(num_nodes, dtype=np.float64)
    time[:num_ancestors] = num_ancestors - np.arange(num_ancestors)
    nodes = msprime.NodeTable()
    nodes.set_columns(flags=flags, time=time)
    del flags, time

    num_edgesets = tree_sequence_builder.num_edgesets
    num_children = tree_sequence_builder.num_children
    left = np.empty(num_edgesets, dtype=np.float64)
    right = np.empty(num_edgesets, dtype=np.float64)
    parent = np.empty(num_edgesets, dtype=np.int32)
    children = np.empty(num_children, dtype=np.int32)
    children_length = np.empty(num_edgesets, dtype=np.uint32)
    tree_sequence_builder.dump_edgesets(
        left, right, parent, children, children_length)
    # Translate left and right into positions.
    # TODO this should be done at the lower level where its safer.
    left = site_position[left.astype(np.uint32)]
    right = site_position[right.astype(np.uint32)]

    edgesets = msprime.EdgesetTable()
    edgesets.set_columns(
        left=left, right=right, parent=parent, children=children,
        children_length=children_length)
    del left, right, parent, children, children_length

    sites = msprime.SiteTable()
    ancestral_state = np.zeros(store.num_sites, dtype=np.int8) + ord('0')
    ancestral_state_length = np.ones(store.num_sites, dtype=np.uint32)
    sites.set_columns(
        position=position, ancestral_state=ancestral_state,
        ancestral_state_length=ancestral_state_length)
    del ancestral_state, ancestral_state_length

    num_mutations = tree_sequence_builder.num_mutations
    site = np.empty(num_mutations, dtype=np.int32)
    node = np.empty(num_mutations, dtype=np.int32)
    derived_state = np.empty(num_mutations, dtype=np.int8)
    derived_state_length = np.ones(num_mutations, dtype=np.uint32)
    tree_sequence_builder.dump_mutations(
        site=site, node=node, derived_state=derived_state)
    # Convert from 0/1 to '0'/'1' chars
    derived_state += ord('0')
    mutations = msprime.MutationTable()
    mutations.set_columns(
        site=site, node=node, derived_state=derived_state,
        derived_state_length=derived_state_length)
    del site, node, derived_state, derived_state_length

    # print(nodes)
    # print(edgesets)
    # print(sites)
    # print(mutations)

    ts = msprime.load_tables(
        nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
    return ts


def infer(samples, positions, length, recombination_rate, error_rate, method="C",
        num_threads=1, show_progress=False):
    num_samples, num_sites = samples.shape
    store = build_ancestors(samples, positions, num_threads=num_threads, method=method,
            show_progress=show_progress)

    if method == "C":
        tree_sequence_builder = _tsinfer.TreeSequenceBuilder(store, num_samples);
    else:
        tree_sequence_builder = TreeSequenceBuilder(store, num_samples);
    match_samples(
        store, samples, recombination_rate, error_rate, tree_sequence_builder,
        method=method, num_threads=num_threads, show_progress=show_progress)
    match_ancestors(
        store, recombination_rate, tree_sequence_builder, method=method,
        num_threads=num_threads, show_progress=show_progress)
    ts = finalise_tree_sequence(num_samples, store, positions, length, tree_sequence_builder)

    # tree_sequence_builder.print_state()
    # ts = tree_sequence_builder.finalise()

    # for e in ts.edgesets():
    #     print(e)
    # for t in ts.trees():
    #     print(t)

    # print()
    # tss = ts.simplify()
    # for e in tss.edgesets():
    #     print(e)
    # for t in tss.trees():
    #     print(t)
    # return tss
    return ts


@attr.s
class LinkedSegment(object):
    """
    A mapping of a half-open interval to a specific value in a linked list.
    Lists of segments must not contain overlapping intervals.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)
    next = attr.ib(default=None)

@attr.s
class Segment(object):
    """
    A mapping of a half-open interval to a specific value.
    """
    start = attr.ib(default=None)
    end = attr.ib(default=None)
    value = attr.ib(default=None)


def chain_str(head):
    """
    Returns the specified chain of segments as a string.
    """
    ret = ""
    u = head
    while u is not None:
        ret += "({}-{}:{})".format(u.start, u.end, u.value)
        u = u.next
        if u is not None:
            ret += "=>"
    return ret


class TreeSequenceBuilder(object):
    """
    Builds a tree sequence online using the traceback
    """
    def __init__(self, store, num_samples):
        self.store = store
        self.num_sites = store.num_sites
        self.num_samples = num_samples
        self.num_internal_nodes = store.num_ancestors
        self.num_mutations = 0
        self.children = [
            LinkedSegment(0, self.num_sites, []) for _ in range(self.num_internal_nodes)]
        self.mutations = [[] for _ in range(self.num_sites)]

    @property
    def num_edgesets(self):
        num_edgesets = 0
        for head in self.children:
            u = head
            while u is not None:
                if len(u.value) > 0:
                    num_edgesets += 1
                u = u.next
        return num_edgesets

    @property
    def num_children(self):
        num_children = 0
        for head in self.children:
            u = head
            while u is not None:
                num_children += len(u.value)
                u = u.next
        return num_children

    def add_mutation(self, node, site, derived_state):
        self.mutations[site].append((node, derived_state))
        self.num_mutations += 1

    def add_gap(self, start, end, child):
        # print("Add GAP", start, end, child)
        self.add_mapping(start, end, 0, child)

    def add_mapping(self, start, end, parent, child):
        assert start < end
        # Skip leading segments
        u = self.children[parent]
        while u.end <= start:
            u = u.next
        if u.start < start:
            v =LinkedSegment(start, u.end, list(u.value), u.next)
            u.end = start
            u.next = v
            u = v
        while u is not None and u.end <= end:
            u.value.append(child)
            u = u.next
        if u is not None and end > u.start:
            v = LinkedSegment(end, u.end, list(u.value), u.next)
            u.end = end
            u.next = v
            u.value.append(child)

        # print("AFTER")
        # self.print_row(parent)
        # Check integrity
        u = self.children[parent]
        while u is not None:
            if u.start >= start and u.end <= end:
                assert child in u.value
            # else:
            #     assert child not in u.value
            u = u.next

    def get_used_segments(self, parent):
        """
        Returns a list of the contiguous segments on the specified parent.
        """
        u = self.children[parent]
        all_used = []
        while u is not None:
            if len(u.value) > 0:
                all_used.append([u.start, u.end])
            u = u.next
        used = [list(all_used[0])]
        for start, end in all_used[1:]:
            # print("\t", start, end, used)
            if start == used[-1][1]:
                used[-1][1] = end
            else:
                used.append([start, end])
        # print("ALL", all_used)
        # print("FIL", used)
        return used



    def get_num_children(self, parent, site):
        """
        Returns the number of children for the specified parent at the specified
        site.
        """
        u = self.children[parent]
        num_children = 0
        while u is not None:
            if u.start <= site < u.end:
                num_children = len(u.value)
                break
            u = u.next
        return num_children

    def select_parent(self, site, options):
        """
        Selects the parent for the specified site
        """
        min_num_children = self.get_num_children(options[0], site)
        best_parent = options[0]
        for parent in options[1:]:
            num_children = self.get_num_children(parent, site)
            if num_children <= min_num_children:
                best_parent = parent
                min_num_children = num_children

        num_children = [self.get_num_children(parent, site) for parent in options]
        # print("SELECT PARENT")
        # print("OPTIONs = ", options)
        # print("NUMCHIL = ", num_children)
        # print("CHOIE   = ", best_parent)
        return best_parent

    def update(
            self, child=None, haplotype=None, start_site=None, end_site=None,
            end_site_parent=None, traceback=None):
        """
        """
        # print("Running traceback on ", start_site, end_site, end_site_parent)
        # traceback.print_state()
        # self.print_state()
        end = end_site
        V = traceback.site_best_segments
        T = traceback.site_head
        # parent = random.randint(
        #     V[end_site - 1][0].start, V[end_site - 1][0].end - 1)
        options = []
        for v in V[end_site - 1]:
            options.extend(range(v.start, v.end))
        assert len(options) > 0
        parent = self.select_parent(end_site - 1, options)
        for l in range(end_site - 1, start_site, -1):
            state = self.store.get_state(l, parent)
            # print("l = ", l, "state = ", state, "parent = ", parent)
            assert state != -1
            if state != haplotype[l]:
                self.add_mutation(child, l, haplotype[l])
            switch = False
            u = T[l]
            while u is not None:
                if u.start <= parent < u.end:
                    switch = True
                    break
                if u.start > parent:
                    break
                u = u.next
            if switch:
                # Complete a segment at this site
                assert l < end
                self.add_mapping(l, end, parent, child)
                end = l
                options = []
                for v in V[l - 1]:
                    options.extend(range(v.start, v.end))
                assert len(options) > 0
                # print("all options:", options)
                # parent = options[0]
                # parent = random.choice(options)
                # parent = options[-1]
                parent = self.select_parent(end_site - 1, options)
                # print("SWITCH: options = ", parent)
        assert start_site < end

        self.add_mapping(start_site, end, parent, child)
        l = start_site
        state = self.store.get_state(l, parent)
        if state != haplotype[l]:
            self.add_mutation(child, l, haplotype[l])
        # if start_site != 0:
        #     self.add_gap(0, start_site, child)
        # if end_site != self.store.num_sites:
        #     self.add_gap(end_site, self.store.num_sites, child)


    def print_state(self):
        print("Builder state")
        for parent in range(len(self.children)):
            self.print_row(parent)

    def print_row(self, parent):
        print(parent, ":", end="")
        u = self.children[parent]
        assert u.start == 0
        while u is not None:
            print("({},{}->{})".format(u.start, u.end, u.value), end="")
            assert len(u.value) == len(set(u.value))
            assert u.start < u.end
            if u.next is not None:
                assert u.end == u.next.start
            prev = u
            u = u.next
        assert prev.end == self.num_sites
        print()

    def dump_edgesets(self, left, right, parent, children, children_length):
        num_edgesets = 0
        num_children = 0
        for p in range(self.num_internal_nodes - 1, -1, -1):
            u = self.children[p]
            assert u.start == 0
            while u is not None:
                if len(u.value) > 0:
                    left[num_edgesets] = u.start
                    right[num_edgesets] = u.end
                    parent[num_edgesets] = p
                    children_length[num_edgesets] = len(u.value)
                    for c in sorted(u.value):
                        children[num_children] = c
                        num_children += 1
                    num_edgesets += 1
                u = u.next

    def dump_mutations(self, site, node, derived_state):
        num_mutations = 0
        for l in range(self.num_sites):
            for u, state in self.mutations[l]:
                site[num_mutations] = l
                node[num_mutations] = u
                derived_state[num_mutations] = state
                num_mutations += 1
        assert num_mutations == self.num_mutations


class OldTreeSequenceBuilder(object):
    """
    Builds a tree sequence from the copying paths of ancestors and samples.
    This uses a simpler extensional list algorithm.
    """
    # TODO move this into test code.
    def __init__(self, num_samples, num_ancestors, num_sites):
        self.num_sites = num_sites
        self.num_samples = num_samples
        self.num_ancestors = num_ancestors
        # The list of children at every site.
        self.children = [
            [[] for _ in range(num_sites)] for _ in range(num_ancestors + 1)]
        self.mutations = [[] for _ in range(num_sites)]

    def print_state(self):
        print("Tree sequence builder state:")
        for l, children in enumerate(self.children):
            print(l, ":", children)
        print("mutations = :")
        for j, mutations in enumerate(self.mutations):
            print(j, ":", mutations)

    def add_path(self, child, P, A, mutations):
        # print("Add path:", child, P, A,mutations)
        for l in range(self.num_sites):
            if P[l] != -1:
                self.children[P[l]][l].append(child)
            else:
                self.children[0][l].append(child)
        for site in mutations:
            self.mutations[site].append((child, str(A[site])))

    def finalise(self):

        # Allocate the nodes.
        nodes = msprime.NodeTable(self.num_ancestors + self.num_samples + 1)
        nodes.add_row(time=self.num_ancestors + 1)
        for j in range(self.num_ancestors):
            nodes.add_row(time=self.num_ancestors - j)
        for j in range(self.num_samples):
            nodes.add_row(time=0, flags=msprime.NODE_IS_SAMPLE)

        # sort all the children lists
        for children_lists in self.children:
            for children in children_lists:
                children.sort()

        edgesets = msprime.EdgesetTable()
        for j in range(self.num_ancestors, -1, -1):
            row = self.children[j]
            last_c = row[0]
            left = 0
            for l in range(1, self.num_sites):
                if row[l] != last_c:
                    if len(last_c) > 0:
                        edgesets.add_row(
                            left=left, right=l, parent=j, children=tuple(last_c))
                    left = l
                    last_c = row[l]
            if len(last_c) > 0:
                edgesets.add_row(
                    left=left, right=self.num_sites, parent=j, children=tuple(last_c))

        sites = msprime.SiteTable()
        mutations = msprime.MutationTable()
        for j in range(self.num_sites):
            sites.add_row(j, "0")
            for node, derived_state in self.mutations[j]:
                mutations.add_row(j, node, derived_state)

        # self.print_state()
        # print(nodes)
        # print(edgesets)
        # print(sites)
        # print(mutations)
        # right = set(edgesets.right)
        # left = set(edgesets.left)
        # print("Diff:", right - left)

        ts = msprime.load_tables(
            nodes=nodes, edgesets=edgesets, sites=sites, mutations=mutations)
        return ts


@attr.s
class SiteState(object):
    position = attr.ib(default=None)
    segments = attr.ib(default=None)


class AncestorStore(object):
    """
    """
    def __init__(self, position=None,
            site=None, start=None, end=None, state=None,
            ancestor_age=None, focal_site_ancestor=None, focal_site=None):
        self.num_sites = position.shape[0]
        self.num_ancestors = np.max(end)
        assert ancestor_age.shape[0] == self.num_ancestors
        self.sites = [SiteState(pos, []) for pos in position]
        self.ancestor_focal_sites = [[]]
        self.ancestor_age = ancestor_age
        last_ancestor = 0
        assert focal_site_ancestor.shape == focal_site.shape
        for j in range(focal_site_ancestor.shape[0]):
            if focal_site_ancestor[j] != last_ancestor:
                last_ancestor = focal_site_ancestor[j]
                self.ancestor_focal_sites.append([])
            self.ancestor_focal_sites[-1].append(focal_site[j])

        j = 0
        for l in range(self.num_sites):
            while j < site.shape[0] and site[j] == l:
                self.sites[l].segments.append(Segment(start[j], end[j], state[j]))
                j += 1
        self.num_older_ancestors = np.zeros(self.num_sites, dtype=np.uint32)
        num_older_ancestors = 1
        last_frequency = ancestor_age[1] + 1
        for j in range(1, self.num_ancestors):
            if ancestor_age[j] < last_frequency:
                last_frequency = ancestor_age[j]
                num_older_ancestors = j
            self.num_older_ancestors[j] = num_older_ancestors

    def print_state(self):
        print("Store:")
        print("Sites:")
        for l, site in enumerate(self.sites):
            print(l, "\t", site)
        print("Ancestors")
        a = np.zeros(self.num_sites, dtype=int)
        for j in range(self.num_ancestors):
            start, end, num_older_ancestors, focal_sites = self.get_ancestor(j, a)
            age = self.ancestor_age[j]
            h = "".join(str(x) if x != -1 else '*' for x in a)
            print(age, start, end, num_older_ancestors, focal_sites, h, sep="\t")


    def get_state(self, site, ancestor):
        """
        Returns the state of the specified ancestor at the specified site.
        """
        for seg in self.sites[site].segments:
            if seg.start <= ancestor < seg.end:
                break
        if seg.start <= ancestor < seg.end:
            return seg.value
        else:
            return -1

    def get_ancestor(self, ancestor_id, a):
        a[:] = -1
        start_site = None
        end_site = self.num_sites
        for l in range(self.num_sites):
            a[l] = self.get_state(l, ancestor_id)
            if a[l] != -1 and start_site is None:
                start_site = l
            if a[l] == -1 and start_site is not None:
                end_site = l
                break
        focal_sites = self.ancestor_focal_sites[ancestor_id]
        num_older_ancestors = self.num_older_ancestors[ancestor_id]
        if ancestor_id > 0:
            for focal_site in focal_sites:
                assert a[focal_site] == 1
        assert sorted(focal_sites) == focal_sites
        assert len(set(focal_sites)) == len(focal_sites)
        if len(focal_sites) > 0:
            assert start_site <= focal_sites[0] <= focal_sites[-1] < end_site
        return start_site, end_site, num_older_ancestors, focal_sites


@attr.s
class Site(object):
    id = attr.ib(default=None)
    frequency = attr.ib(default=None)


class AncestorBuilder(object):
    """
    Builds inferred ancestors.
    """
    def __init__(self, S, positions):
        self.haplotypes = S
        self.num_samples = S.shape[0]
        self.num_sites = S.shape[1]
        self.sites = [None for j in range(self.num_sites)]
        self.sorted_sites = [None for j in range(self.num_sites)]
        for j in range(self.num_sites):
            self.sites[j] = Site(j, np.sum(S[:, j]))
            self.sorted_sites[j] = Site(j, np.sum(S[:, j]))
        self.sorted_sites.sort(key=lambda x: (-x.frequency, x.id))
        frequency_sites = collections.defaultdict(list)
        for site in self.sorted_sites:
            if site.frequency > 1:
                frequency_sites[site.frequency].append(site)
        # Group together identical sites within a frequency class
        self.frequency_classes = {}
        for frequency, sites in frequency_sites.items():
            patterns = collections.defaultdict(list)
            for site in sites:
                state = tuple(self.haplotypes[:, site.id])
                patterns[state].append(site.id)
            self.frequency_classes[frequency] = list(patterns.values())

    def get_frequency_classes(self):
        ret = []
        for frequency in reversed(sorted(self.frequency_classes.keys())):
            ret.append((frequency, self.frequency_classes[frequency]))
        return ret

    def __build_ancestor_sites(self, focal_site, sites, a):
        S = self.haplotypes
        samples = set()
        for j in range(self.num_samples):
            if S[j, focal_site.id] == 1:
                samples.add(j)
        for l in sites:
            a[l] = 0
            if self.sites[l].frequency > focal_site.frequency:
                # print("\texamining:", self.sites[l])
                # print("\tsamples = ", samples)
                num_ones = 0
                num_zeros = 0
                for j in samples:
                    if S[j, l] == 1:
                        num_ones += 1
                    else:
                        num_zeros += 1
                # TODO choose a branch uniformly if we have equality.
                if num_ones >= num_zeros:
                    a[l] = 1
                    samples = set(j for j in samples if S[j, l] == 1)
                else:
                    samples = set(j for j in samples if S[j, l] == 0)
            if len(samples) == 1:
                # print("BREAK")
                break

    def make_ancestor(self, focal_sites, a):
        a[:] = -1
        focal_site = self.sites[focal_sites[0]]
        sites = range(focal_sites[-1] + 1, self.num_sites)
        self.__build_ancestor_sites(focal_site, sites, a)
        focal_site = self.sites[focal_sites[-1]]
        sites = range(focal_sites[0] - 1, -1, -1)
        self.__build_ancestor_sites(focal_site, sites, a)
        for j in range(focal_sites[0], focal_sites[-1] + 1):
            if j in focal_sites:
                a[j] = 1
            else:
                self.__build_ancestor_sites(focal_site, [j], a)
        return a

class Traceback(object):
    def __init__(self, store):
        self.store = store
        self.reset()

    def add_best_segment(self, site, start, end):
        """
        Adds a maximum likelihood recombinant segment for the specified site.
        """
        self.site_best_segments[site].append(LinkedSegment(start, end))

    def add_recombination(self, site, start, end):
        if self.site_head[site] is None:
            self.site_head[site] = LinkedSegment(start, end)
            self.site_tail[site] = self.site_head[site]
        else:
            if self.site_tail[site].end == start:
                self.site_tail[site].end = end
            else:
                tail = LinkedSegment(start, end)
                self.site_tail[site].next = tail
                self.site_tail[site] = tail

    def reset(self):
        self.site_head = [None for _ in range(self.store.num_sites)]
        self.site_tail = [None for _ in range(self.store.num_sites)]
        self.site_best_segments = [[] for _ in range(self.store.num_sites)]

    def print_state(self):
        print("traceback:")
        for l, head in enumerate(self.site_head):
            print(l, ":", end="")
            s = ""
            for seg in self.site_best_segments[l]:
                s += "({},{})".format(seg.start, seg.end)
            print(s, "=", end="")
            u = head
            while u != None:
                print("({},{})".format(u.start, u.end, u.value), end="")
                u = u.next
            print()



class AncestorMatcher(object):
    """
    """
    def __init__(self, store, recombination_rate):
        self.store = store
        self.recombination_rate = recombination_rate

    def best_path(self, num_ancestors=None, haplotype=None, start_site=None,
            end_site=None, focal_sites=None, error_rate=None, traceback=None):
        """
        Returns the best path through the list of ancestors for the specified
        haplotype.
        """
        h = haplotype
        assert h.shape == (self.store.num_sites,)
        m = self.store.num_sites
        # print("store = ", self.store)
        n = num_ancestors
        L = [Segment(0, n, 1)]
        best_haplotype = 0
        # Ensure that the initial recombination rate is zero
        last_position = self.store.sites[start_site].position
        possible_recombinants = n
        focal_site_index = 0
        while focal_site_index < len(focal_sites) and focal_sites[focal_site_index] < start_site:
            focal_site_index += 1

        # print("BEST PATH", num_ancestors, h, start_site, end_site, focal_sites)


        for site in range(start_site, end_site):
            if focal_site_index < len(focal_sites) and site > focal_sites[focal_site_index]:
                focal_site_index += 1
            # print("site = ", site, focal_site_index, focal_sites)

            L_next = []
            # S = [Segment(*s) for s in self.store.get_site(site)]
            S = self.store.sites[site].segments
            # Compute the recombination rate.
            # TODO also need to get the position here so we can get the length of the
            # region.
            rho = self.recombination_rate * (self.store.sites[site].position - last_position)
            r = 1 - np.exp(-rho / possible_recombinants)
            pr = r / possible_recombinants
            qr = 1 - r + r / possible_recombinants

            # print()
            # print("site = ", site)
            # print("rho = ", rho, "r = ", r)
            # print("pos = ", self.store.sites[site].position)
            # print("n = ", n)
            # print("re= ", possible_recombinants)
            # print("pr= ", pr)
            # print("qr= ", qr)
            # print("L = ", L)
            # print("S = ", S)
            # print("h = ", h[site])
            # print("b = ", best_haplotype)

            l = 0
            s = 0
            start = 0
            while start != n:
                end = n
                if l < len(L):
                    if L[l].start > start:
                        end = min(end, L[l].start)
                    else:
                        end = min(end, L[l].end)
                if s < len(S):
                    if S[s].start > start:
                        end = min(end, S[s].start)
                    else:
                        end = min(end, S[s].end)
                # print("\tLOOP HEAD: start = ", start, "end = ", end)
                # print("\ts = ", s)
                # print("\tl = ", l)
                assert start < end
                # The likelihood of this interval is always 0 if it does not intersect
                # with S
                if s < len(S) and not (S[s].start >= end or S[s].end <= start):
                    state = S[s].value
                    # If this interval does not intersect with L, the likelihood is 0
                    likelihood = 0
                    if l < len(L) and not (L[l].start >= end or L[l].end <= start):
                        likelihood = L[l].value

                    x = likelihood * qr
                    y = pr  # v for maximum is 1 by normalisation
                    # print("\t", start, end, x, y)
                    if x >= y:
                        z = x
                    else:
                        z = y
                        traceback.add_recombination(site, start, end)

                    # Determine the likelihood for this segment.
                    if error_rate == 0:
                        # Ancestor matching.
                        likelihood_next = z * int(state == h[site])
                        # if site in focal_sites:
                        #     assert h[site] == 1
                        #     assert state == 0
                        #     likelihood_next = z
                        #     focal_site_index += 1

                        if focal_site_index < len(focal_sites):
                            if site == focal_sites[focal_site_index]:
                                assert h[site] == 1
                                assert state == 0
                                likelihood_next = z
                            else:
                                assert site < focal_sites[focal_site_index]
                    else:
                        # Sample matching.
                        if state == h[site]:
                            likelihood_next = z * (1 - error_rate)
                        else:
                            likelihood_next = z * error_rate

                    # Update the L_next array
                    if len(L_next) == 0:
                        L_next = [Segment(start, end, likelihood_next)]
                    else:
                        if L_next[-1].end == start and L_next[-1].value == likelihood_next:
                            L_next[-1].end = end
                        else:
                            L_next.append(Segment(start, end, likelihood_next))
                start = end
                if l < len(L) and L[l].end <= start:
                    l += 1
                if s < len(S) and S[s].end <= start:
                    s += 1

            L = L_next
            max_value = -1
            for seg in L:
                assert seg.start < seg.end
                if seg.value >= max_value:
                    max_value = seg.value
            for seg in L:
                if seg.value == max_value:
                    traceback.add_best_segment(site, seg.start, seg.end)
            if max_value > 0:
                # Renormalise L
                for seg in L:
                    seg.value /= max_value
            last_position = self.store.sites[site].position
            # Compute the possible recombination destinations for the next iteration.
            s = 0
            possible_recombinants = 0
            while s < len(S) and S[s].start < n:
                possible_recombinants += min(n, S[s].end) - S[s].start
                s += 1

        # print("focal_site_index = ", focal_site_index, len(focal_sites))
        # assert focal_site_index == len(focal_sites)



    def print_state(self):
        print("Matcher state")
        print("num_ancestors = ", self.num_ancestors)
        print("num_sites = ", self.num_sites)


