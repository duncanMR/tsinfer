#
# Copyright (C) 2023 University of Oxford
#
# This file is part of tsinfer.
#
# tsinfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tsinfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Ancestor handling routines.
"""
import logging
import numpy as np
import tskit
import uuid
from numba import njit, int8, int32, int64, float64, types, boolean
from numba.experimental import jitclass
import attr
import collections
import time as time_
import tsinfer.constants as constants
logger = logging.getLogger(__name__)


@attr.s
class Edge:
    """
    A singley linked list of edges.
    """

    left = attr.ib(default=None)
    right = attr.ib(default=None)
    parent = attr.ib(default=None)
    child = attr.ib(default=None)
    next = attr.ib(default=None)


@attr.s
class Site:
    """
    A single site for the ancestor builder.
    """

    id = attr.ib()
    time = attr.ib()
    

@attr.s(order=False, eq=False)
class Ancestor:
    """
    An ancestor object.
    """
    start = attr.ib()
    end = attr.ib()
    focal_sites = attr.ib()
    full_haplotype = attr.ib()
    sample_set_by_site = attr.ib()
    min_sample_count = attr.ib()

    @property
    def haplotype(self):
        return self.full_haplotype[self.start : self.end]
    

    def __eq__(self, other):
        return (
            self.start == other.start
            and self.end == other.end
            and np.array_equal(self.focal_sites, other.focal_sites)
            and np.array_equal(self.full_haplotype, other.full_haplotype)
        )

spec = [
    ("num_samples", int32),
    ("num_sites", int32),
    ("sites_time", float64[:]),
    ("genotype_store", int8[:]),
    ("sample_func", types.FunctionType(int64(int64))),
    ("full_haplotype", int8[:]),
    ("sample_set_by_site", int32[:, :]),
    ("min_sample_count", int32),
    ("start", int32),
    ("end", int32),
    ("freq_threshold", float64),
]

@jitclass(spec)
class NumbaAncestorBuilder:
    def __init__(self, sites_time, num_samples, num_sites, genotype_store, sample_func, freq_threshold):
        self.sites_time = sites_time
        self.num_samples = num_samples
        self.num_sites = num_sites
        self.genotype_store = genotype_store
        self.sample_func = sample_func
        self.freq_threshold = freq_threshold
        self.full_haplotype = np.full(self.num_sites, -1, dtype=np.int8)
        self.sample_set_by_site = np.zeros((self.num_sites, self.num_samples), dtype=np.int32)
        self.min_sample_count = -1
        self.start = -1
        self.end = -1

    def get_site_genotypes(self, site_id):
        start = site_id * self.num_samples
        stop = start + self.num_samples
        genotypes = self.genotype_store[start:stop]
        return genotypes

    def get_site_genotype(self, site_id, sample):
        return self.genotype_store[site_id * self.num_samples + sample]

    def get_consistent_samples(self, site):
        genotypes = self.get_site_genotypes(site)
        sample_set = np.zeros(self.num_samples, dtype=np.int32)
        j = 0
        k = 0
        while j < self.num_samples:
            if genotypes[j] == 1:
                sample_set[k] = j
                k += 1
            j += 1
        sample_set_size = k

        return sample_set, sample_set_size
    
    def write_sample_set(self, sample_set, sample_set_size, site):
        j = 0
        while j < sample_set_size:
            #print(self.sample_set_by_site)
            #print(sample_set[j])
            #print(site)
            self.sample_set_by_site[site, sample_set[j]] = 1
            j += 1

    def compute_ancestral_states(self, focal_site, direction):
        """
        For a given focal site, and set of sites to fill in (usually all the ones
        leftwards or rightwards), augment the haplotype array a with the inferred sites
        Together with `make_ancestor`, which calls this function, these describe the main
        algorithm as implemented in Fig S2 of the preprint, with the buffer.

        At the moment we assume that the derived state is 1. We should alter this so
        that we allow the derived state to be a different non-zero integer.
        """
        
        focal_time = self.sites_time[focal_site]
        print(focal_time)
        sample_set, sample_set_size = self.get_consistent_samples(focal_site)
        self.write_sample_set(sample_set, sample_set_size, focal_site)
        assert sample_set_size > 0

        last_site = focal_site
        if focal_time >= self.freq_threshold:
            return last_site
        # Break when we've lost half of the samples
        min_sample_set_size = self.sample_func(sample_set_size)
        self.min_sample_count = min_sample_set_size
        disagree = np.full(self.num_samples, False)
        site_index = focal_site + direction
        while site_index >= 0 and site_index < self.num_sites:

            self.full_haplotype[site_index] = 0
            self.write_sample_set(sample_set, sample_set_size, site_index)
            last_site = site_index
            if self.sites_time[site_index] > focal_time:
                ones = 0
                zeros = 0
                j = 0
                while j < sample_set_size:
                    genotype = self.get_site_genotype(site_index, sample_set[j])
                    if genotype == 1:
                        ones += 1
                    elif genotype == 0:
                        zeros += 1
                    j += 1
                if ones + zeros == 0:
                    self.full_haplotype[site_index] = -1
                else:
                    consensus = 1 if ones >= zeros else 0
                    j = 0
                    while j < sample_set_size:
                        u = sample_set[j]
                        genotype = self.get_site_genotype(site_index, u)
                        if disagree[u] and (genotype != consensus) and (genotype != -1):
                            sample_set[j] = -1
                        j += 1
                    self.full_haplotype[site_index] = consensus

                    if len(sample_set) <= min_sample_set_size:
                        break

                    j = 0
                    while j < sample_set_size:
                        u = sample_set[j]
                        genotype = self.get_site_genotype(site_index, u)
                        if u != -1:
                            disagree[u] = (genotype != consensus) and (genotype != -1)
                        j += 1

                    # Repack the sample set array
                    j = 0
                    tmp_size = 0
                    while j < sample_set_size:
                        if sample_set[j] != -1:
                            sample_set[tmp_size] = sample_set[j]
                            tmp_size += 1
                        j += 1
                    sample_set_size = tmp_size

                    if sample_set_size <= min_sample_set_size:
                        break
            site_index += direction

        assert self.full_haplotype[last_site] != -1
        return last_site

    def compute_between_focal_sites(self, focal_sites):
        focal_site = focal_sites[0]
        focal_time = self.sites_time[focal_site]
        sample_set, sample_set_size = self.get_consistent_samples(focal_site)
        for site in focal_sites:
            self.write_sample_set(sample_set, sample_set_size, site)
        assert sample_set_size > 0

        # Interpolate ancestral haplotype within focal region (i.e. region
        #  spanning from leftmost to rightmost focal site)
        k = 0
        while k < (len(focal_sites) - 1):
            # Interpolate region between focal site j and focal site j+1
            site_index = focal_sites[k] + 1
            
            while site_index < focal_sites[k + 1]:
                self.full_haplotype[site_index] = 0
                self.write_sample_set(sample_set, sample_set_size, site_index)
                if self.sites_time[site_index] > focal_time:
                    ones = 0
                    zeros = 0
                    j = 0
                    while j < sample_set_size:
                        genotype = self.get_site_genotype(site_index, sample_set[j])
                        if genotype == 1:
                            ones += 1
                        elif genotype == 0:
                            zeros += 1
                        j += 1
                    if ones + zeros == 0:
                        self.full_haplotype[site_index] = -1
                    elif ones >= zeros:
                        self.full_haplotype[site_index] = 1
                site_index += 1
            k += 1

    def make_ancestor(self, focal_sites):
        """
        Fills out the array a with the haplotype
        return the start and end of an ancestor
        """
        focal_site = focal_sites[0]
        for site in focal_sites:
            self.full_haplotype[site] = 1
        self.sample_set_by_site = np.zeros((self.num_sites, self.num_samples), dtype=np.int32)
        self.compute_between_focal_sites(focal_sites)

        # Extend rightwards from rightmost focal site
        focal_site = focal_sites[-1]
        last_site = self.compute_ancestral_states(focal_site, +1)
        self.end = last_site + 1
        # Extend leftwards from leftmost focal site")
        focal_site = focal_sites[0]
        last_site = self.compute_ancestral_states(focal_site, -1)
        self.start = last_site
        


class AncestorBuilder:
    """
    Builds inferred ancestors.
    This implementation partially allows for multiple focal sites per ancestor
    """

    def __init__(
        self,
        num_samples,
        max_sites,
        sample_func,
        freq_threshold,
        one_site_per_anc,
        method=1,
        genotype_encoding=None,
    ):
        self.num_samples = num_samples
        self.sites = []
        self.builder = None
        self.freq_threshold = freq_threshold
        self.method = method
        self.sites_time = np.zeros(max_sites, dtype=np.float64)
        self.one_site_per_anc = one_site_per_anc
        # Create a mapping from time to sites. Different sites can exist at the same
        # timepoint. If we expect them to be part of the same ancestor node we can give
        # them the same ancestor_uid: the time_map contains values keyed by time, with
        # values consisting of a dictionary, d, of uid=>[array_of_site_ids]
        # It is handy to be able to add to d without checking, so we make this a
        # defaultdict of defaultdicts
        self.time_map = collections.defaultdict(lambda: collections.defaultdict(list))
        if genotype_encoding is None:
            genotype_encoding = constants.GenotypeEncoding.EIGHT_BIT
        assert genotype_encoding == constants.GenotypeEncoding.EIGHT_BIT
        self.genotype_encoding = genotype_encoding
        self.encoded_genotypes_size = num_samples
        if genotype_encoding == constants.GenotypeEncoding.ONE_BIT:
            self.encoded_genotypes_size = num_samples // 8 + int((num_samples % 8) != 0)
        self.genotype_store = np.zeros(
            max_sites * self.encoded_genotypes_size, dtype=np.uint8
        )
        self.sample_func = sample_func

    @property
    def num_sites(self):
        return len(self.sites)

    @property
    def mem_size(self):
        # Just here for compatibility with the C implementation.
        return 0

    def get_site_genotypes_subset(self, site_id, samples):
        start = site_id * self.encoded_genotypes_size
        g = np.zeros(len(samples), dtype=np.int8)
        if self.genotype_encoding == constants.GenotypeEncoding.ONE_BIT:
            for j, u in enumerate(samples):
                byte_index = u // 8
                bit_index = u % 8
                byte = self.genotype_store[start + byte_index]
                mask = 1 << bit_index
                g[j] = int((byte & mask) != 0)
        else:
            for j, u in enumerate(samples):
                g[j] = self.genotype_store[start + u]
        gp = self.get_site_genotypes(site_id)
        np.testing.assert_array_equal(gp[samples], g)
        return g

    def get_site_genotypes(self, site_id):
        start = site_id * self.encoded_genotypes_size
        stop = start + self.encoded_genotypes_size
        g = self.genotype_store[start:stop]
        if self.genotype_encoding == constants.GenotypeEncoding.ONE_BIT:
            g = np.unpackbits(g, bitorder="little")[: self.num_samples]
        g = g.astype(np.int8)
        return g

    def store_site_genotypes(self, site_id, genotypes):
        if self.genotype_encoding == constants.GenotypeEncoding.ONE_BIT:
            assert np.all(genotypes >= 0) and np.all(genotypes <= 1)
            genotypes = np.packbits(genotypes, bitorder="little")
        start = site_id * self.encoded_genotypes_size
        stop = start + self.encoded_genotypes_size
        self.genotype_store[start:stop] = genotypes

    def get_genotypes_mat(self):
        g_mat = self.genotype_store[: (self.num_sites * self.num_samples)].reshape(
            self.num_sites, self.num_samples
        )
        return g_mat

    def add_site(self, time, genotypes):
        """
        Adds a new site at the specified ID to the builder.
        """
        site_id = len(self.sites)
        self.store_site_genotypes(site_id, genotypes)
        self.sites.append(Site(site_id, time))
        self.sites_time[site_id] = time
        sites_at_fixed_timepoint = self.time_map[time]
        # Sites with an identical variant distribution (i.e. with the same
        # genotypes.tobytes() value) and at the same time, are put into the same ancestor
        # to which we allocate a unique ID (just use the genotypes value)
        if self.one_site_per_anc is True:
            ancestor_uid = str(uuid.uuid4())
        else:
            ancestor_uid = genotypes.tobytes()
    
        # Add each site to the list for this ancestor_uid at this timepoint
        sites_at_fixed_timepoint[ancestor_uid].append(site_id)

    def print_state(self, return_str=False):
        string = "Current state of ancestor builder:\nSites:\n"
        for j in range(self.num_sites):
            site = self.sites[j]
            genotypes = self.get_site_genotypes(j)
            string += f" {j}\t{genotypes}\t{site.time:.3f}\n"
        string += "Time map:\n"
        for t in sorted(self.time_map.keys()):
            sites_at_fixed_timepoint = self.time_map[t]
            if len(sites_at_fixed_timepoint) > 0:
                string += (
                    f" t = {t:.3f} with {len(sites_at_fixed_timepoint)} ancestors\n"
                )
                for ancestor, sites in sites_at_fixed_timepoint.items():
                    string += f"\t{ancestor} : {sites}\n"
        string += "Ancestor descriptors:\n"
        for t, focal_sites in self.ancestor_descriptors():
            string += f"\tt = {t:.3f}, sites = {focal_sites}\n"
        if return_str is True:
            return string
        else:
            print(string)

    def break_ancestor(self, a, b, samples):
        """
        Returns True if we should split the ancestor with focal sites at
        a and b into two separate ancestors (if there is an older site between them
        which is not compatible with the focal site distribution)
        """
        # return True
        for j in range(a + 1, b):
            if self.sites[j].time > self.sites[a].time:
                gj = self.get_site_genotypes_subset(j, samples)
                gj = gj[gj != tskit.MISSING_DATA]
                if not (np.all(gj == 1) or np.all(gj == 0)):
                    return True
        return False

    def ancestor_descriptors(self):
        """
        Returns a list of (time, focal_sites) tuples describing the ancestors in
        in arbitrary order.
        """
        ret = []
        for t in self.time_map.keys():
            for focal_sites in self.time_map[t].values():
                genotypes = self.get_site_genotypes(focal_sites[0])
                samples = np.where(genotypes == 1)[0]
                start = 0
                for j in range(len(focal_sites) - 1):
                    if self.break_ancestor(focal_sites[j], focal_sites[j + 1], samples):
                        ret.append((t, focal_sites[start : j + 1]))
                        start = j + 1
                ret.append((t, focal_sites[start:]))
        return ret

    def make_ancestor(self, focal_sites):
        """
        Fills out the array a with the haplotype
        return the start and end of an ancestor
        """
        if self.method == "primary":
            if self.builder is None:
                self.builder = NumbaAncestorBuilder(
                    sites_time=self.sites_time,
                    num_samples=self.num_samples,
                    num_sites=self.num_sites,
                    genotype_store=self.genotype_store,
                    sample_func=self.sample_func,
                    freq_threshold=self.freq_threshold,
                )
            self.builder.make_ancestor(focal_sites)
            
            #print(self.builder.full_sample_counts[self.builder.start:self.builder.end])
            return Ancestor(
                start=self.builder.start,
                end=self.builder.end,
                focal_sites=focal_sites,
                full_haplotype=self.builder.full_haplotype,
                sample_set_by_site=self.builder.sample_set_by_site,
                min_sample_count=self.builder.min_sample_count,
            )
        elif self.method == "alternative":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown method {self.method}")



def merge_overlapping_ancestors(start, end, time):
    # Merge overlapping, same-time ancestors. We do this by scanning along a single
    # time epoch from left to right, detecting breaks.
    sort_indices = np.lexsort((start, time))
    start = start[sort_indices]
    end = end[sort_indices]
    time = time[sort_indices]
    old_indexes = {}
    # For efficiency, pre-allocate the output arrays to the maximum possible size.
    new_start = np.full_like(start, -1)
    new_end = np.full_like(end, -1)
    new_time = np.full_like(time, -1)

    i = 0
    new_index_pos = 0
    while i < len(start):
        j = i + 1
        group_overlap = [i]
        max_right = end[i]
        # While we're in the same time epoch, and the next ancestor
        # overlaps with the group, add this ancestor to the group.
        while j < len(start) and time[j] == time[i] and start[j] < max_right:
            max_right = max(max_right, end[j])
            group_overlap.append(j)
            j += 1

        # Emit the found group
        old_indexes[new_index_pos] = group_overlap
        new_start[new_index_pos] = start[i]
        new_end[new_index_pos] = max_right
        new_time[new_index_pos] = time[i]
        new_index_pos += 1
        i = j
    # Trim the output arrays to the actual size.
    new_start = new_start[:new_index_pos]
    new_end = new_end[:new_index_pos]
    new_time = new_time[:new_index_pos]
    return new_start, new_end, new_time, old_indexes, sort_indices


@njit
def run_linesweep(event_times, event_index, event_type, new_time):
    # Run the linesweep over the ancestor start-stop events,
    # building up the dependency graph as a count of dependencies for each ancestor,
    # and a list of dependant children for each ancestor.
    n = len(new_time)

    # numba really likes to know the type of the list elements, so we tell it by adding
    # a dummy element to the list and then popping it off.
    # `active` is the list of ancestors that overlap with the current linesweep position.
    active = [-1]
    active.pop()
    children = [[-1] for _ in range(n)]
    for c in range(n):
        children[c].pop()
    incoming_edge_count = np.zeros(n, dtype=np.int32)
    for i in range(len(event_times)):
        index = event_index[i]
        e_time = event_times[i]
        if event_type[i] == 1:
            for j in active:
                if new_time[j] > e_time:
                    incoming_edge_count[index] += 1
                    children[j].append(index)
                elif new_time[j] < e_time:
                    incoming_edge_count[j] += 1
                    children[index].append(j)
            active.append(index)
        else:
            active.remove(index)

    # Convert children to ragged array format so we can pass arrays to the
    # next numba function, `find_groups`.
    children_data = []
    children_indices = [0]
    for child_list in children:
        children_data.extend(child_list)
        children_indices.append(len(children_data))
    children_data = np.array(children_data, dtype=np.int32)
    children_indices = np.array(children_indices, dtype=np.int32)
    return children_data, children_indices, incoming_edge_count


@njit
def find_groups(children_data, children_indices, incoming_edge_count):
    # We find groups of ancestors that can be matched in parallel by topologically
    # sorting the dependency graph. We do this by deconstructing the graph, removing
    # nodes with no incoming edges, and adding them to a group.
    n = len(children_indices) - 1
    group_id = np.full(n, -1, dtype=np.int32)
    current_group = 0
    while True:
        # Find the nodes with no incoming edges
        no_incoming = np.where(incoming_edge_count == 0)[0]
        if len(no_incoming) == 0:
            break
        # Remove them from the graph
        for i in no_incoming:
            incoming_edge_count[i] = -1
            incoming_edge_count[
                children_data[children_indices[i] : children_indices[i + 1]]
            ] -= 1
        # Add them to the group
        group_id[no_incoming] = current_group
        current_group += 1
    return group_id


def group_ancestors_by_linesweep(start, end, time):
    # For a given set of ancestors, we want to group them for matching in parallel.
    # For each ancestor, any overlapping, older ancestors must be in an earlier group,
    # and any overlapping, younger ancestors in a later group. Any overlapping same-age
    # ancestors must be in the same group so they don't match to each other.
    # We do this by first merging the overlapping same-age ancestors. Then build a
    # dependency graph of the ancestors by linesweep. Then form groups by topological
    # sort. Finally, we un-merge the same-age ancestors.

    assert len(start) == len(end)
    assert len(start) == len(time)
    t = time_.time()
    (
        new_start,
        new_end,
        new_time,
        old_indexes,
        sort_indices,
    ) = merge_overlapping_ancestors(start, end, time)
    logger.info(f"Merged to {len(new_start)} ancestors in {time_.time() - t:.2f}s")

    # Build a list of events for the linesweep
    t = time_.time()
    n = len(new_time)
    # Create events arrays by copying and concatenating inputs
    event_times = np.concatenate([new_time, new_time])
    event_pos = np.concatenate([new_start, new_end])
    event_index = np.concatenate([np.arange(n), np.arange(n)])
    event_type = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])
    # Sort events by position, then ends before starts
    event_sort_indices = np.lexsort((event_type, event_pos))
    event_times = event_times[event_sort_indices]
    event_index = event_index[event_sort_indices]
    event_type = event_type[event_sort_indices]
    logger.info(f"Built {len(event_times)} events in {time_.time() - t:.2f}s")

    t = time_.time()
    children_data, children_indices, incoming_edge_count = run_linesweep(
        event_times, event_index, event_type, new_time
    )
    logger.info(
        f"Linesweep generated {np.sum(incoming_edge_count)} dependencies in"
        f" {time_.time() - t:.2f}s"
    )

    t = time_.time()
    group_id = find_groups(children_data, children_indices, incoming_edge_count)
    logger.info(f"Found groups in {time_.time() - t:.2f}s")

    t = time_.time()
    # Convert the group id array to lists of ids for each group
    ancestor_grouping = {}
    for group in np.unique(group_id):
        ancestor_grouping[group] = np.where(group_id == group)[0]

    # Now un-merge the same-age ancestors, simultaneously mapping back to the original,
    # unsorted indexes
    for group in ancestor_grouping:
        ancestor_grouping[group] = sorted(
            [
                sort_indices[item]
                for i in ancestor_grouping[group]
                for item in old_indexes[i]
            ]
        )
    logger.info(f"Un-merged in {time_.time() - t:.2f}s")
    logger.info(
        f"{len(ancestor_grouping)} groups with median size "
        f"{np.median([len(ancestor_grouping[group]) for group in ancestor_grouping])}"
    )
    return ancestor_grouping
