import msprime
import numpy as np
import tsinfer
import collections
import tskit
import bisect
import random
import colorama
import pandas as pd
import numba


def assert_smc(ts):
    """
    Check if the specified tree sequence fulfils SMC requirements. This
    means that we cannot have any discontinuous parent segments.
    """
    parent_intervals = collections.defaultdict(list)
    for es in ts.edgesets():
        parent_intervals[es.parent].append((es.left, es.right))
    for intervals in parent_intervals.values():
        if len(intervals) > 0:
            intervals.sort()
            for j in range(1, len(intervals)):
                if intervals[j - 1][1] != intervals[j][0]:
                    raise ValueError("Only SMC simulations are supported")


def get_ancestral_haplotypes(ts):
    """
    Returns a numpy array of the haplotypes of the ancestors in the
    specified tree sequence.
    """
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags[:]
    flags[:] = 1
    nodes.set_columns(time=nodes.time, flags=flags)

    sites = tables.sites.position
    tsp = tables.tree_sequence()
    B = tsp.genotype_matrix().T

    A = np.full((ts.num_nodes, ts.num_sites), tskit.MISSING_DATA, dtype=np.int8)
    for edge in ts.edges():
        start = bisect.bisect_left(sites, edge.left)
        end = bisect.bisect_right(sites, edge.right)
        if sites[end - 1] == edge.right:
            end -= 1
        A[edge.parent, start:end] = B[edge.parent, start:end]
    A[: ts.num_samples] = B[: ts.num_samples]
    return A


def get_ancestor_descriptors(A):
    """
    Given an array of ancestral haplotypes A in forwards time-order (i.e.,
    so that A[0] == 0), return the descriptors for each ancestor within
    this set and remove any ancestors that do not have any novel mutations.
    Returns the list of ancestors, the start and end site indexes for
    each ancestor, and the list of focal sites for each one.

    This assumes that the input is SMC safe, and will return incorrect
    results on ancestors that contain trapped genetic material.
    """
    L = A.shape[1]
    ancestors = [np.zeros(L, dtype=np.int8)]
    focal_sites = [[]]
    start = [0]
    end = [L]
    # ancestors = []
    # focal_sites = []
    # start = []
    # end = []
    mask = np.ones(L)
    for a in A:
        masked = np.logical_and(a == 1, mask).astype(int)
        new_sites = np.where(masked)[0]
        mask[new_sites] = 0
        segment = np.where(a != tskit.MISSING_DATA)[0]
        # Skip any ancestors that are entirely unknown
        if segment.shape[0] > 0:
            s = segment[0]
            e = segment[-1] + 1
            assert np.all(a[s:e] != tskit.MISSING_DATA)
            assert np.all(a[:s] == tskit.MISSING_DATA)
            assert np.all(a[e:] == tskit.MISSING_DATA)
            ancestors.append(a)
            focal_sites.append(new_sites)
            start.append(s)
            end.append(e)
    return np.array(ancestors, dtype=np.int8), start, end, focal_sites


def build_simulated_ancestors(ancestor_data, ts, time_chunking=False):
    # Any non-smc tree sequences are rejected.
    assert_smc(ts)
    assert ancestor_data.num_sites > 0
    A = get_ancestral_haplotypes(ts)
    # This is all nodes, but we only want the non samples. We also reverse
    # the order to make it forwards time.
    A = A[ts.num_samples :][::-1]

    # get_ancestor_descriptors ensures that the ultimate ancestor is included.
    ancestors, start, end, focal_sites = get_ancestor_descriptors(A)
    N = len(ancestors)
    if time_chunking:
        time = np.zeros(N)
        intersect_mask = np.zeros(A.shape[1], dtype=int)
        t = 0
        for j in range(N):
            if np.any(intersect_mask[start[j] : end[j]] == 1):
                t += 1
                intersect_mask[:] = 0
            intersect_mask[start[j] : end[j]] = 1
            time[j] = t
    else:
        time = np.arange(N)
    time = -1 * (time - time[-1]) + 1
    for a, s, e, focal, t in zip(ancestors, start, end, focal_sites, time):
        assert np.all(a[:s] == tskit.MISSING_DATA)
        assert np.all(a[s:e] != tskit.MISSING_DATA)
        assert np.all(a[e:] == tskit.MISSING_DATA)
        assert all(s <= site < e for site in focal)
        ancestor_data.add_ancestor(
            start=s,
            end=e,
            time=t,
            focal_sites=np.array(focal, dtype=np.int32),
            haplotype=a[s:e],
        )

def generate_true_and_inferred_ancestors(ts, engine="C"):
    """
    Run a simulation under args and return the samples, plus the true and the inferred
    ancestors
    """
    sample_data = tsinfer.SampleData.from_tree_sequence(ts)
    inferred_anc = tsinfer.generate_ancestors(sample_data, engine=engine)
    true_anc = tsinfer.AncestorData(
        sample_data.sites_position, sample_data.sequence_length
    )
    build_simulated_ancestors(true_anc, ts)
    true_anc.finalise()
    return sample_data, true_anc, inferred_anc


def ancestor_data_by_pos(anc1, anc2):
    """
    Return indexes into ancestor data, keyed by focal site position, returning only
    those indexes where positions are the same for both ancestors. This is useful
    e.g. for plotting length v length scatterplots.
    """
    anc_by_focal_pos = []
    for anc in (anc1, anc2):
        position_to_index = {
            anc.sites_position[:][site_index]: i
            for i, sites in enumerate(anc.ancestors_focal_sites[:])
            for site_index in sites
        }
        anc_by_focal_pos.append(position_to_index)

    # NB with error we may not have exactly the same focal sites in exact & estimated
    shared_indices = set.intersection(*[set(a.keys()) for a in anc_by_focal_pos])

    return {
        pos: np.array([anc_by_focal_pos[0][pos], anc_by_focal_pos[1][pos]], np.int64)
        for pos in shared_indices
    }


def compare_true_vs_inferred_anc(
    sample_data, true_anc, inferred_anc,
):
    """
    Calculate quality measures per focal site, as these are comparable from inferred
    to true ancestors. This is a bit complicated because we don't always have the same
    inference sites in inferred & true ancestors, so we need to only check the sites
    that are shared. We also need to limit the bounds over which we calculate quality
    so that we only look at the regions of overlap between true and inferred ancestors
    """
    anc_indices = ancestor_data_by_pos(true_anc, inferred_anc)
    shared_positions = np.array(list(sorted(anc_indices.keys())))
    # append sequence_length to pos so that ancestors_end[:] indices are always valid
    true_positions = np.append(true_anc.sites_position[:], sample_data.sequence_length)
    inferred_positions = np.append(
        inferred_anc.sites_position[:], sample_data.sequence_length
    )
    # only include sites which are focal in both exact and estim in the genome-wise masks
    true_sites_mask = np.isin(true_anc.sites_position[:], shared_positions)
    inferred_sites_mask = np.isin(inferred_anc.sites_position[:], shared_positions)
    assert np.sum(true_sites_mask) == np.sum(inferred_sites_mask) == len(anc_indices)

    # store the data to plot for each focal_site, keyed by position
    freq = {var.site.position: np.sum(var.genotypes) for var in sample_data.variants()}
    inferred_freq = np.array(
        [freq[p] for p in inferred_anc.sites_position], dtype=np.int64
    )
    olap_n_sites = {}
    olap_n_should_be_1_higher_freq = {}
    olap_n_should_be_0_higher_freq = {}
    olap_n_should_be_1_low_eq_freq = {}
    olap_n_should_be_0_low_eq_freq = {}
    olap_left = {}
    olap_right = {}
    true_left = {}
    true_right = {}
    inferred_node = {}
    inferred_time = {}
    inferred_left = {}
    inferred_right = {}
    true_len = {}
    est_len = {}
    true_time = {}
    true_node = {}
    # find the left and right edges of the overlap - iterate by true time in reverse
    for i, focal_pos in enumerate(
        sorted(
            shared_positions,
            key=lambda pos: -true_anc.ancestors_time[:][anc_indices[pos][0]],
        )
    ):
        true_index, inferred_index = anc_indices[focal_pos]
        # left (start) is biggest of exact and estim
        true_start = true_positions[true_anc.ancestors_start[:][true_index]]
        inferred_start = inferred_positions[
            inferred_anc.ancestors_start[:][inferred_index]
        ]
        if true_start > inferred_start:
            olap_start_exact = true_anc.ancestors_start[:][true_index]
            olap_start = true_positions[olap_start_exact]
            olap_start_estim = np.searchsorted(
                inferred_anc.sites_position[:], olap_start
            )
        else:
            olap_start_estim = inferred_anc.ancestors_start[:][inferred_index]
            olap_start = inferred_positions[olap_start_estim]
            olap_start_exact = np.searchsorted(true_anc.sites_position[:], olap_start)

        # right (end) is smallest of exact and estim
        true_end = true_positions[true_anc.ancestors_end[:][true_index]]
        inferred_end = inferred_positions[inferred_anc.ancestors_end[:][inferred_index]]
        if true_end < inferred_end:
            olap_end_exact = true_anc.ancestors_end[:][true_index]
            olap_end = true_positions[olap_end_exact]
            olap_end_estim = np.searchsorted(inferred_anc.sites_position[:], olap_end)
        else:
            olap_end_estim = inferred_anc.ancestors_end[:][inferred_index]
            olap_end = inferred_positions[olap_end_estim]
            olap_end_exact = np.searchsorted(true_anc.sites_position[:], olap_end)

        offset1 = true_anc.ancestors_start[:][true_index]
        offset2 = inferred_anc.ancestors_start[:][inferred_index]

        true_full_hap = true_anc.ancestors_full_haplotype[:, true_index, 0]
        # slice the full haplotype to include only the overlapping region
        true_olap = true_full_hap[olap_start_exact:olap_end_exact]
        # make a 1/0 array with only the comparable sites
        true_comp = true_olap[true_sites_mask[olap_start_exact:olap_end_exact]]

        inferred_full_hap = inferred_anc.ancestors_full_haplotype[:, inferred_index, 0]
        inferred_olap = inferred_full_hap[olap_start_estim:olap_end_estim]
        small_inferred_mask = inferred_sites_mask[olap_start_estim:olap_end_estim]
        inferred_comp = inferred_olap[small_inferred_mask]

        assert len(true_comp) == len(inferred_comp)

        # save the statistics into variables indexed by position
        bad_sites = true_comp != inferred_comp
        should_be_1 = true_comp & ~inferred_comp
        should_be_0 = inferred_comp & ~true_comp

        assert np.sum(should_be_1 | should_be_0) == np.sum(bad_sites)

        olap_n_sites[focal_pos] = len(true_comp)
        olap_left[focal_pos] = olap_start
        olap_right[focal_pos] = olap_end
        true_left[focal_pos] = true_start
        true_right[focal_pos] = true_end
        inferred_left[focal_pos] = inferred_start
        inferred_right[focal_pos] = inferred_end
        inferred_node[focal_pos] = inferred_index
        true_node[focal_pos] = true_index
        true_len[focal_pos] = true_anc.ancestors_length[:][true_index]
        est_len[focal_pos] = inferred_anc.ancestors_length[:][inferred_index]
        true_time[focal_pos] = true_anc.ancestors_time[:][true_index]
        inferred_time[focal_pos] = inferred_anc.ancestors_time[:][inferred_index]
        sites_freq = inferred_freq[olap_start_estim:olap_end_estim]
        higher_freq = sites_freq[small_inferred_mask] > freq[focal_pos]
        olap_n_should_be_1_higher_freq[focal_pos] = np.sum(should_be_1 & higher_freq)
        olap_n_should_be_0_higher_freq[focal_pos] = np.sum(should_be_0 & higher_freq)
        olap_n_should_be_1_low_eq_freq[focal_pos] = np.sum(should_be_1 & ~higher_freq)
        olap_n_should_be_0_low_eq_freq[focal_pos] = np.sum(should_be_0 & ~higher_freq)
        assert olap_right[focal_pos] - olap_left[focal_pos] <= true_len[focal_pos]
        assert olap_n_should_be_1_higher_freq[
            focal_pos
        ] + olap_n_should_be_0_higher_freq[focal_pos] + olap_n_should_be_1_low_eq_freq[
            focal_pos
        ] + olap_n_should_be_0_low_eq_freq[
            focal_pos
        ] == np.sum(
            bad_sites
        )

    # create the data for use, ordered by real time (and make a new time index)
    df = pd.DataFrame.from_records(
        [
            (
                p,
                freq[p],
                olap_n_sites[p],
                true_len[p],
                est_len[p],
                olap_right[p] - olap_left[p],
                olap_left[p],
                olap_right[p],
                inferred_node[p],
                inferred_left[p],
                inferred_right[p],
                true_node[p],
                true_left[p],
                true_right[p],
                olap_n_should_be_1_higher_freq[p],
                olap_n_should_be_1_low_eq_freq[p],
                olap_n_should_be_0_higher_freq[p],
                olap_n_should_be_0_low_eq_freq[p],
                t,
                true_time[p],
                inferred_time[p],
            )
            for t, p in enumerate(sorted(shared_positions, key=lambda x: true_time[x]))
        ],
        columns=(
            "focal_site_pos",
            "focal_site_frequency",
            "num_overlapping_sites",
            "true_length",
            "inferred_length",
            "overlap_length",
            "overlap_left",
            "overlap_right",
            "inferred_node",
            "inferred_left",
            "inferred_right",
            "true_node",
            "true_left",
            "true_right",
            "err_hifreq_should_be_1",
            "err_lowfreq_should_be_1",
            "err_hifreq_should_be_0",
            "err_lowfreq_should_be_0",
            "true_time_order",
            "true_time",
            "inferred_time",
        ),
    )

    # we want to know for each site whether it the frequency puts it within the same
    # bounds as the known time order, and if not, whether we have inferred it as
    # too old or too young. So we make an ordered list of "expected" freqs
    freq_bins = np.bincount(df.focal_site_frequency)
    freq_repeated = np.repeat(np.arange(len(freq_bins)), freq_bins)
    # add another column on to the expected freq, as calculated from the actual time
    df["expected_frequency"] = freq_repeated[df["true_time_order"].values]
    df["num_mismatches"] = (
        df["err_hifreq_should_be_1"]
        + df["err_lowfreq_should_be_1"]
        + df["err_hifreq_should_be_0"]
        + df["err_lowfreq_should_be_0"]
    )
    df["inaccuracy"] = df.num_mismatches / df.num_overlapping_sites
    df["inferred_time_inaccuracy"] = df.expected_frequency - df.focal_site_frequency
    df["inference_error_bias"] = (
        df["err_hifreq_should_be_1"] + df["err_lowfreq_should_be_1"]
    ) / df.num_mismatches
    df["err_hiF"] = df["err_hifreq_should_be_1"] + df["err_hifreq_should_be_0"]
    df["err_loF"] = df["err_lowfreq_should_be_1"] + df["err_lowfreq_should_be_0"]

    print(
        "{} ancestors, {} with at least one error".format(
            len(df), np.sum(df.num_mismatches != 0)
        )
    )
    print(
        df[
            [
                "err_hifreq_should_be_1",
                "err_lowfreq_should_be_1",
                "err_hifreq_should_be_0",
                "err_lowfreq_should_be_0",
            ]
        ].sum()
    )
    inferred_nodes = df.inferred_node.values
    ancestor_ts = tsinfer.match_ancestors(sample_data, inferred_anc, num_threads=8)
    inferred_ts = tsinfer.match_samples(
        sample_data, ancestor_ts, num_threads=8, post_process=False
    )
    simplified_ts = inferred_ts.simplify(filter_nodes=False, keep_unary=True)
    copied_left, copied_right = extract_copying_data(
        num_nodes=simplified_ts.num_nodes,
        edges_left=simplified_ts.edges_left,
        edges_right=simplified_ts.edges_right,
        edges_parent=simplified_ts.edges_parent,
        node_subset=inferred_nodes,
    )
    # Inferred left starts at first site but edges start at 0
    copied_left = np.maximum(copied_left, df.inferred_left)
    copied_length = copied_right - copied_left
    assert np.all(copied_length >= 0)
    df["copied_left"] = copied_left
    df["copied_right"] = copied_right
    df["copied_length"] = copied_length
    df['copied_length'] = df['copied_length']
    df["copied_length_ratio"] = df.copied_length / df.inferred_length
    #df.drop_duplicates(subset=['true_node', 'inferred_node'], inplace=True)

    return df, simplified_ts


@numba.njit
def extract_copying_data(num_nodes, edges_left, edges_right, edges_parent, node_subset):
    """Copied from tsqc repo"""
    num_edges = edges_left.shape[0]
    copied_left = np.zeros(num_nodes, dtype=np.float64) + np.inf
    copied_right = np.zeros(num_nodes, dtype=np.float64)

    for e in range(num_edges):
        p = edges_parent[e]
        right = edges_right[e]
        left = edges_left[e]
        if copied_left[p] > left:
            copied_left[p] = left
        if copied_right[p] < right:
            copied_right[p] = right
    
    return copied_left[node_subset], copied_right[node_subset]


