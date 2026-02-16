# Visualisation of the copying process and ancestor generation using PIL
import math
import os
import struct
import sys

import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import svgwrite

import tsinfer


def load_hmm_log(path):
    path_begin = {}
    path_end = {}
    site_rows = []

    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"TSILHMML":
            raise ValueError(f"Bad magic: {magic!r}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 2:
            raise ValueError(f"Unsupported version: {version}")

        while True:
            t = f.read(1)
            if not t:
                break
            rec_type = t[0]

            if rec_type == 1:  # PATH_BEGIN
                ancestor_id, start, end = struct.unpack("<Qii", f.read(16))
                path_begin[ancestor_id] = {"start": start, "end": end}

            elif rec_type == 2:  # SITE_VALUES
                ancestor_id, site, k = struct.unpack("<QiI", f.read(16))
                vals = np.frombuffer(f.read(8 * k), dtype="<f8").copy()
                node_ids = np.frombuffer(f.read(4 * k), dtype="<i4").copy()
                site_rows.append(
                    {
                        "ancestor_id": ancestor_id,
                        "site": site,
                        "k": k,
                        "likelihoods": vals,
                        "likelihood_nodes": node_ids,
                    }
                )

            elif rec_type == 3:  # PATH_END
                ancestor_id, status, total_memory = struct.unpack("<QiQ", f.read(20))
                path_end[ancestor_id] = {"status": status, "total_memory": total_memory}

            else:
                raise ValueError(f"Unknown record type: {rec_type}")

    sites_df = pd.DataFrame(site_rows)

    ancestor_ids = sorted(set(path_begin) | set(path_end))
    paths_df = pd.DataFrame(
        [
            {
                "ancestor_id": pid,
                "start": path_begin.get(pid, {}).get("start"),
                "end": path_begin.get(pid, {}).get("end"),
                "status": path_end.get(pid, {}).get("status"),
                "total_memory": path_end.get(pid, {}).get("total_memory"),
            }
            for pid in ancestor_ids
        ]
    )

    return sites_df, paths_df


def make_long_df(df, anc_data):
    """
    Flatten per-site likelihood arrays into one row per likelihood value.

    Input columns:
    - ancestor_id
    - site
    - k
    - likelihoods
    - likelihood_nodes

    Output columns:
    - ancestor_id
    - site
    - k
    - full_likelihood
    - likelihood
    - likelihood_node_id
    """
    required = {"ancestor_id", "site", "k", "likelihoods", "likelihood_nodes"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    df["ancestor_time"] = anc_data.ancestors_time[df["ancestor_id"]]

    k = df["k"].to_numpy(dtype=np.int64, copy=False)
    if np.any(k < 0):
        raise ValueError("Column 'k' must be non-negative")

    likelihoods = df["likelihoods"].to_numpy(dtype=object, copy=False)
    likelihood_nodes = df["likelihood_nodes"].to_numpy(dtype=object, copy=False)

    # Validate payload lengths once to fail early on malformed rows.
    like_lens = np.fromiter(
        (len(x) for x in likelihoods), dtype=np.int64, count=len(df)
    )
    node_lens = np.fromiter(
        (len(x) for x in likelihood_nodes), dtype=np.int64, count=len(df)
    )
    if not np.array_equal(like_lens, k):
        raise ValueError("Lengths in 'likelihoods' do not match column 'k'")
    if not np.array_equal(node_lens, k):
        raise ValueError("Lengths in 'likelihood_nodes' do not match column 'k'")

    if k.sum() == 0:
        return pd.DataFrame(
            {
                "ancestor_id": pd.Series(dtype=df["ancestor_id"].dtype),
                "ancestor_time": pd.Series(dtype=df["ancestor_time"].dtype),
                "site": pd.Series(dtype=df["site"].dtype),
                "k": pd.Series(dtype=np.int64),
                "full_likelihood": pd.Series(dtype=np.float64),
                "likelihood_node_id": pd.Series(dtype=np.int32),
                "likelihood": pd.Series(dtype=np.int8),
            }
        )

    likelihood_flat = np.concatenate(likelihoods)
    long_df = pd.DataFrame(
        {
            "ancestor_id": np.repeat(df["ancestor_id"].to_numpy(copy=False), k),
            "ancestor_time": np.repeat(df["ancestor_time"].to_numpy(copy=False), k),
            "site": np.repeat(df["site"].to_numpy(copy=False), k),
            "k": np.repeat(k, k),
            "full_likelihood": likelihood_flat,
            "likelihood_node_id": np.concatenate(likelihood_nodes),
            "likelihood": np.equal(likelihood_flat, 1.0).astype(np.int8),
        }
    )
    return long_df


def summarise_likelihoods(df, ancestor_id, likelihood_node_id, genome_length):
    likelihood_col = "likelihood"
    required = {"ancestor_id", "site", "likelihood_node_id", likelihood_col}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    if genome_length <= 0:
        raise ValueError("genome_length must be positive")

    ancestor_arr = np.atleast_1d(ancestor_id)
    node_arr = np.atleast_1d(likelihood_node_id)
    if ancestor_arr.size > 1 and node_arr.size > 1:
        raise ValueError("One of ancestor_id or likelihood_node_id must be scalar")
    if ancestor_arr.size == 1:
        ancestor_arr = np.repeat(ancestor_arr, node_arr.size)
    else:
        node_arr = np.repeat(node_arr, ancestor_arr.size)

    targets = pd.DataFrame(
        {
            "ancestor_id": ancestor_arr.astype(np.int64, copy=False),
            "likelihood_node_id": node_arr.astype(np.int64, copy=False),
        }
    )
    targets = targets.drop_duplicates(ignore_index=True)

    work = df.loc[:, ["ancestor_id", "site", "likelihood_node_id", likelihood_col]]
    work = work.merge(targets, how="inner", on=["ancestor_id", "likelihood_node_id"])
    if len(work) == 0:
        out = targets.copy()
        out["site_start"] = np.nan
        out["site_end"] = np.nan
        out["site_span_incl_gaps"] = 0.0
        out["likelihood_start"] = np.nan
        out["num_switches"] = 0
        out["num_gaps"] = 0
        out["site_span_excl_gaps"] = 0.0
        out["site_span_is_0"] = 0.0
        out["site_span_is_1"] = 0.0
        out["num_0_to_1_transitions"] = 0
        out["num_1_to_0_transitions"] = 0
        out["prop_is_0"] = np.nan
        out["prop_is_1"] = np.nan
        out["coverage_prop_incl_gaps"] = 0.0
        out["coverage_prop_excl_gaps"] = 0.0
        return out

    work = work.sort_values(
        ["ancestor_id", "likelihood_node_id", "site"], kind="mergesort"
    )
    if work.duplicated(["ancestor_id", "likelihood_node_id", "site"]).any():
        raise ValueError("Found duplicate (ancestor_id, likelihood_node_id, site) rows")

    g = work.groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)
    prev_site = g["site"].shift()
    site_step = work["site"] - prev_site
    run_break = site_step.ne(1) | site_step.isna()
    prev_like = g[likelihood_col].shift()
    like_change = work[likelihood_col].ne(prev_like) & prev_like.notna()
    work = work.assign(
        run_break=run_break,
        like_change=like_change,
        is_0_to_1=(prev_like == 0) & (work[likelihood_col] == 1),
        is_1_to_0=(prev_like == 1) & (work[likelihood_col] == 0),
    )
    g2 = work.groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)
    work["run_id"] = g2["run_break"].cumsum()
    work["state_id"] = g2["run_break"].cumsum() + g2["like_change"].cumsum()

    summary = g2.agg(
        site_start=("site", "first"),
        site_end=("site", "last"),
        likelihood_start=(likelihood_col, "first"),
        num_switches=("like_change", "sum"),
        num_0_to_1_transitions=("is_0_to_1", "sum"),
        num_1_to_0_transitions=("is_1_to_0", "sum"),
    ).reset_index()
    summary["site_span_incl_gaps"] = summary["site_end"] - summary["site_start"]

    runs = (
        work.groupby(
            ["ancestor_id", "likelihood_node_id", "run_id"], sort=False, observed=True
        )["site"]
        .agg(run_start="first", run_end="last")
        .reset_index()
    )
    runs["run_span"] = runs["run_end"] - runs["run_start"]
    run_stats = runs.groupby(
        ["ancestor_id", "likelihood_node_id"], sort=False, observed=True
    ).agg(
        num_runs=("run_id", "size"),
        site_span_excl_gaps=("run_span", "sum"),
    )
    run_stats["num_gaps"] = run_stats["num_runs"] - 1
    run_stats = run_stats.drop(columns=["num_runs"]).reset_index()

    states = (
        work.groupby(
            ["ancestor_id", "likelihood_node_id", "state_id", likelihood_col],
            sort=False,
            observed=True,
        )["site"]
        .agg(seg_start="first", seg_end="last")
        .reset_index()
    )
    states["seg_span"] = states["seg_end"] - states["seg_start"]
    span_0 = (
        states.loc[states[likelihood_col] == 0]
        .groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)[
            "seg_span"
        ]
        .sum()
        .rename("site_span_is_0")
        .reset_index()
    )
    span_1 = (
        states.loc[states[likelihood_col] == 1]
        .groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)[
            "seg_span"
        ]
        .sum()
        .rename("site_span_is_1")
        .reset_index()
    )

    out = targets.merge(summary, how="left", on=["ancestor_id", "likelihood_node_id"])
    out = out.merge(run_stats, how="left", on=["ancestor_id", "likelihood_node_id"])
    out = out.merge(span_0, how="left", on=["ancestor_id", "likelihood_node_id"])
    out = out.merge(span_1, how="left", on=["ancestor_id", "likelihood_node_id"])

    for col in [
        "num_switches",
        "num_0_to_1_transitions",
        "num_1_to_0_transitions",
        "num_gaps",
    ]:
        out[col] = out[col].fillna(0).astype(np.int64)
    for col in ["site_span_excl_gaps", "site_span_is_0", "site_span_is_1"]:
        out[col] = out[col].fillna(0.0)
    out["site_span_incl_gaps"] = out["site_span_incl_gaps"].fillna(0.0)
    out["coverage_prop_incl_gaps"] = out["site_span_incl_gaps"] / float(genome_length)
    out["coverage_prop_excl_gaps"] = out["site_span_excl_gaps"] / float(genome_length)

    denom = out["site_span_excl_gaps"].to_numpy(dtype=np.float64, copy=False)
    out["prop_is_0"] = np.divide(
        out["site_span_is_0"].to_numpy(dtype=np.float64, copy=False),
        denom,
        out=np.zeros(out.shape[0], dtype=np.float64),
        where=denom > 0,
    )
    out["prop_is_1"] = np.divide(
        out["site_span_is_1"].to_numpy(dtype=np.float64, copy=False),
        denom,
        out=np.zeros(out.shape[0], dtype=np.float64),
        where=denom > 0,
    )
    return out[
        [
            "ancestor_id",
            "likelihood_node_id",
            "site_start",
            "site_end",
            "site_span_incl_gaps",
            "likelihood_start",
            "num_switches",
            "num_gaps",
            "site_span_excl_gaps",
            "site_span_is_0",
            "site_span_is_1",
            "num_0_to_1_transitions",
            "num_1_to_0_transitions",
            "prop_is_0",
            "prop_is_1",
            "coverage_prop_incl_gaps",
            "coverage_prop_excl_gaps",
        ]
    ]


def summarise_all_likelihoods(df):
    required = {"ancestor_id", "site", "likelihood_node_id", "likelihood"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    has_ancestor_time = "ancestor_time" in df.columns

    columns = [
        "ancestor_id",
        "likelihood_node_id",
        "site_start",
        "site_end",
        "site_span_incl_gaps",
        "likelihood_start",
        "num_switches",
        "num_gaps",
        "site_span_excl_gaps",
        "site_span_is_0",
        "site_span_is_1",
        "num_0_to_1_transitions",
        "num_1_to_0_transitions",
        "prop_is_0",
        "prop_is_1",
        "coverage_prop_incl_gaps",
        "coverage_prop_excl_gaps",
    ]
    if has_ancestor_time:
        columns = ["ancestor_id", "ancestor_time"] + columns[1:]
    if len(df) == 0:
        return pd.DataFrame(columns=columns)

    genome_length = int(df["site"].max()) + 1
    work_cols = ["ancestor_id", "site", "likelihood_node_id", "likelihood"]
    if has_ancestor_time:
        work_cols.insert(1, "ancestor_time")
    work = df.loc[:, work_cols]
    work = work.sort_values(
        ["ancestor_id", "likelihood_node_id", "site"], kind="mergesort"
    )
    if work.duplicated(["ancestor_id", "likelihood_node_id", "site"]).any():
        raise ValueError("Found duplicate (ancestor_id, likelihood_node_id, site) rows")

    g = work.groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)
    prev_site = g["site"].shift()
    site_step = work["site"] - prev_site
    run_break = site_step.ne(1) | site_step.isna()
    prev_like = g["likelihood"].shift()
    like_change = work["likelihood"].ne(prev_like) & prev_like.notna()
    work = work.assign(
        run_break=run_break,
        like_change=like_change,
        is_0_to_1=(prev_like == 0) & (work["likelihood"] == 1),
        is_1_to_0=(prev_like == 1) & (work["likelihood"] == 0),
    )
    g2 = work.groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)
    work["run_id"] = g2["run_break"].cumsum()
    work["state_id"] = g2["run_break"].cumsum() + g2["like_change"].cumsum()

    summary_aggs = {
        "site_start": ("site", "first"),
        "site_end": ("site", "last"),
        "likelihood_start": ("likelihood", "first"),
        "num_switches": ("like_change", "sum"),
        "num_0_to_1_transitions": ("is_0_to_1", "sum"),
        "num_1_to_0_transitions": ("is_1_to_0", "sum"),
    }
    if has_ancestor_time:
        summary_aggs["ancestor_time"] = ("ancestor_time", "first")
    out = g2.agg(**summary_aggs).reset_index()
    out["site_span_incl_gaps"] = out["site_end"] - out["site_start"]

    runs = (
        work.groupby(
            ["ancestor_id", "likelihood_node_id", "run_id"], sort=False, observed=True
        )["site"]
        .agg(run_start="first", run_end="last")
        .reset_index()
    )
    runs["run_span"] = runs["run_end"] - runs["run_start"]
    run_stats = runs.groupby(
        ["ancestor_id", "likelihood_node_id"], sort=False, observed=True
    ).agg(
        num_runs=("run_id", "size"),
        site_span_excl_gaps=("run_span", "sum"),
    )
    run_stats["num_gaps"] = run_stats["num_runs"] - 1
    run_stats = run_stats.drop(columns=["num_runs"]).reset_index()

    states = (
        work.groupby(
            ["ancestor_id", "likelihood_node_id", "state_id", "likelihood"],
            sort=False,
            observed=True,
        )["site"]
        .agg(seg_start="first", seg_end="last")
        .reset_index()
    )
    states["seg_span"] = states["seg_end"] - states["seg_start"]
    span_0 = (
        states.loc[states["likelihood"] == 0]
        .groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)[
            "seg_span"
        ]
        .sum()
        .rename("site_span_is_0")
        .reset_index()
    )
    span_1 = (
        states.loc[states["likelihood"] == 1]
        .groupby(["ancestor_id", "likelihood_node_id"], sort=False, observed=True)[
            "seg_span"
        ]
        .sum()
        .rename("site_span_is_1")
        .reset_index()
    )

    out = out.merge(run_stats, how="left", on=["ancestor_id", "likelihood_node_id"])
    out = out.merge(span_0, how="left", on=["ancestor_id", "likelihood_node_id"])
    out = out.merge(span_1, how="left", on=["ancestor_id", "likelihood_node_id"])

    for col in [
        "num_switches",
        "num_0_to_1_transitions",
        "num_1_to_0_transitions",
        "num_gaps",
    ]:
        out[col] = out[col].fillna(0).astype(np.int64)
    for col in ["site_span_excl_gaps", "site_span_is_0", "site_span_is_1"]:
        out[col] = out[col].fillna(0.0)
    out["site_span_incl_gaps"] = out["site_span_incl_gaps"].fillna(0.0)

    out["coverage_prop_incl_gaps"] = out["site_span_incl_gaps"] / float(genome_length)
    out["coverage_prop_excl_gaps"] = out["site_span_excl_gaps"] / float(genome_length)

    denom = out["site_span_excl_gaps"].to_numpy(dtype=np.float64, copy=False)
    out["prop_is_0"] = np.divide(
        out["site_span_is_0"].to_numpy(dtype=np.float64, copy=False),
        denom,
        out=np.zeros(out.shape[0], dtype=np.float64),
        where=denom > 0,
    )
    out["prop_is_1"] = np.divide(
        out["site_span_is_1"].to_numpy(dtype=np.float64, copy=False),
        denom,
        out=np.zeros(out.shape[0], dtype=np.float64),
        where=denom > 0,
    )
    return out[columns]


def sample_likelihoods_by_id(df, by, id, num_samples, seed=1):
    required = {"ancestor_id", "site", "likelihood_node_id", "likelihood"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    if by not in {"ancestor", "node"}:
        raise ValueError("by must be 'ancestor' or 'node'")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if len(df) == 0:
        return summarise_likelihoods(
            df, ancestor_id=[], likelihood_node_id=[], genome_length=1
        )

    genome_length = int(df["site"].max()) + 1
    rng = np.random.default_rng(seed)

    if by == "ancestor":
        subset_df = df.loc[df["ancestor_id"] == id]
        candidates = subset_df["likelihood_node_id"].drop_duplicates().to_numpy()
        if num_samples >= candidates.size:
            sampled_ids = np.sort(candidates)
        else:
            sampled_ids = np.sort(
                rng.choice(candidates, size=num_samples, replace=False)
            )
        return summarise_likelihoods(
            subset_df,
            ancestor_id=id,
            likelihood_node_id=sampled_ids,
            genome_length=genome_length,
        )

    subset_df = df.loc[df["likelihood_node_id"] == id]
    candidates = subset_df["ancestor_id"].drop_duplicates().to_numpy()
    if num_samples >= candidates.size:
        sampled_ids = np.sort(candidates)
    else:
        sampled_ids = np.sort(rng.choice(candidates, size=num_samples, replace=False))
    return summarise_likelihoods(
        subset_df,
        ancestor_id=sampled_ids,
        likelihood_node_id=id,
        genome_length=genome_length,
    )


def plot_likelihood_nodes(df):
    """
    Plot aggregated tracked likelihood nodes (k) by site.

    Expects columns:
    - site
    - k
    - ancestor_id (optional)
    """
    required = {"site", "k"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")

    stats = (
        df.groupby("site", sort=True)["k"]
        .agg(k_mean="mean", k_max="max")
        .reset_index()
        .sort_values("site")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    line_color = "C0"
    ax.fill_between(
        stats["site"], 0, stats["k_max"], color=line_color, alpha=0.3, linewidth=0
    )
    ax.plot(
        stats["site"],
        stats["k_mean"],
        color=line_color,
        linewidth=1,
        label="Mean node count per site",
    )

    ax.set_xlabel("site")
    ax.set_ylabel("k")
    ax.set_title("Tracked Likelihood Nodes per Site")
    ax.legend()
    return fig, ax


def plot_likelihood_values(df, n_chunks):
    """
    Plot proportions of unique likelihood values aggregated into chunks.

    The DataFrame must contain columns ``site`` and ``likelihood``. The sites
    are sorted, split into ``n_chunks`` roughly equal bins, and proportions
    of each likelihood value are stacked in a bar per chunk.
    """
    if "likelihood" in df.columns:
        likelihood_col = "likelihood"
    elif "likelihoods" in df.columns:
        likelihood_col = "likelihoods"
    else:
        raise ValueError("Missing columns for likelihood plot: likelihoods/likelihood")

    required = {"site", likelihood_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for likelihood plot: {missing}")
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    unique_sites = np.sort(df["site"].unique())
    if unique_sites.size == 0:
        raise ValueError("DataFrame contains no sites")

    chunk_size = max(1, math.ceil(unique_sites.size / n_chunks))
    site_to_order = {site: idx for idx, site in enumerate(unique_sites)}
    if likelihood_col == "likelihoods":
        # sites_df stores arrays; flatten with vectorized repeat/concatenate.
        df = make_long_df(df)
        likelihood_col = "likelihood"
    df["chunk"] = (
        df["site"].map(site_to_order).floordiv(chunk_size).clip(upper=n_chunks - 1)
    )

    counts = (
        df.groupby(["chunk", likelihood_col], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(range(n_chunks), fill_value=0)
    )

    totals = counts.sum(axis=1).replace(0, 1)
    proportions = counts.div(totals, axis=0)

    chunk_labels = []
    for chunk in range(n_chunks):
        first_idx = chunk * chunk_size
        last_idx = min(first_idx + chunk_size - 1, unique_sites.size - 1)
        if first_idx >= unique_sites.size:
            chunk_labels.append("empty")
            continue
        start_site = unique_sites[first_idx]
        end_site = unique_sites[last_idx]
        chunk_labels.append(f"{start_site}-{end_site}")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(n_chunks)
    bottom = np.zeros(n_chunks)
    colors = plt.get_cmap("tab10")
    for idx, value in enumerate(sorted(counts.columns)):
        values = proportions[value].fillna(0).values
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=colors(idx % 10),
            label=str(value),
            width=0.8,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(chunk_labels, rotation=45, ha="right")
    ax.set_xlabel("Genomic region (discrete sites)")
    ax.set_ylabel("Proportion of nodes")
    ax.set_title("Likelihood distribution by region across all ancestors")
    ax.legend(title="Likelihood")
    ax.margins(x=0.01)
    return fig, ax


def boxplot_by_epoch(df, num_epochs, variable, log_y=False):
    required = {"ancestor_time", variable}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    work = df.loc[:, ["ancestor_time", variable]].dropna()
    if len(work) == 0:
        raise ValueError("No rows available after dropping NaNs")
    if (work["ancestor_time"] <= 0).any():
        raise ValueError(
            "ancestor_time must be strictly positive for log-spaced epochs"
        )

    log_time = np.log10(work["ancestor_time"].to_numpy(dtype=np.float64, copy=False))
    edges = np.linspace(log_time.min(), log_time.max(), num_epochs + 1)
    epoch = np.digitize(log_time, edges[1:-1], right=False).astype(np.int64)
    work = work.assign(epoch=epoch)

    grouped = work.groupby("epoch", sort=True)[variable]
    epochs = np.arange(num_epochs)
    data = [
        grouped.get_group(i).to_numpy(copy=False) for i in epochs if i in grouped.groups
    ]
    positions = [int(i) + 1 for i in epochs if i in grouped.groups]
    labels = [
        f"{10 ** edges[i]:.2g}-{10 ** edges[i + 1]:.2g}"
        for i in epochs
        if i in grouped.groups
    ]
    if len(data) == 0:
        raise ValueError("No non-empty epochs available to plot")

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(data)), 4))
    ax.boxplot(data, positions=positions, widths=0.6, showfliers=True)
    ax.set_xlabel("epoch")
    ax.set_ylabel(variable)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"{variable} by log-spaced ancestor-time epoch")
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    return fig, ax


def plot_prop_by_epoch(df, num_epochs, num_bins, variable, log_y=False):
    required = {"ancestor_time", variable}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    work = df.loc[:, ["ancestor_time", variable]].dropna()
    if len(work) == 0:
        raise ValueError("No rows available after dropping NaNs")
    if (work["ancestor_time"] <= 0).any():
        raise ValueError(
            "ancestor_time must be strictly positive for log-spaced epochs"
        )

    log_time = np.log10(work["ancestor_time"].to_numpy(dtype=np.float64, copy=False))
    edges = np.linspace(log_time.min(), log_time.max(), num_epochs + 1)
    epoch = np.digitize(log_time, edges[1:-1], right=False).astype(np.int64)
    work = work.assign(epoch=epoch)

    grouped = work.groupby("epoch", sort=True)[variable]
    epoch_ids = [i for i in range(num_epochs) if i in grouped.groups]
    if len(epoch_ids) == 0:
        raise ValueError("No non-empty epochs available to plot")

    fig, axes = plt.subplots(
        1,
        len(epoch_ids),
        figsize=(max(8, 3.2 * len(epoch_ids)), 3.6),
        sharey=True,
        constrained_layout=True,
    )
    if len(epoch_ids) == 1:
        axes = [axes]

    for ax, i in zip(axes, epoch_ids):
        values = grouped.get_group(i).to_numpy(copy=False)
        ax.hist(values, bins=num_bins, density=True, color="C0", alpha=0.8)
        ax.set_title(f"{10 ** edges[i]:.2g}-{10 ** edges[i + 1]:.2g}")
        ax.set_xlabel(variable)
        ax.set_ylabel("density")
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(f"{variable} distribution by ancestor time epoch")
    return fig, axes


def plot_count_by_epoch(df, num_epochs, num_bins, variable, log_y=False):
    required = {"ancestor_time", variable}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame missing required columns: {missing_str}")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    work = df.loc[:, ["ancestor_time", variable]].dropna()
    if len(work) == 0:
        raise ValueError("No rows available after dropping NaNs")
    if (work["ancestor_time"] <= 0).any():
        raise ValueError(
            "ancestor_time must be strictly positive for log-spaced epochs"
        )

    values = work[variable].to_numpy(copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{variable} contains non-finite values")
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{variable} must be integer-valued for count plots")

    log_time = np.log10(work["ancestor_time"].to_numpy(dtype=np.float64, copy=False))
    edges = np.linspace(log_time.min(), log_time.max(), num_epochs + 1)
    epoch = np.digitize(log_time, edges[1:-1], right=False).astype(np.int64)
    work = work.assign(epoch=epoch, value_int=values.astype(np.int64, copy=False))

    grouped = work.groupby("epoch", sort=True)["value_int"]
    epoch_ids = [i for i in range(num_epochs) if i in grouped.groups]
    if len(epoch_ids) == 0:
        raise ValueError("No non-empty epochs available to plot")

    fig, axes = plt.subplots(
        1,
        len(epoch_ids),
        figsize=(max(8, 3.2 * len(epoch_ids)), 3.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(epoch_ids) == 1:
        axes = [axes]

    x = np.arange(num_bins + 1)
    x_labels = [str(i) for i in range(num_bins)] + [f">={num_bins}"]
    for ax, i in zip(axes, epoch_ids):
        vals = grouped.get_group(i).to_numpy(dtype=np.int64, copy=False)
        vals = np.minimum(vals, num_bins)
        counts = np.bincount(vals, minlength=num_bins + 1)
        ax.bar(x, counts, color="C0", alpha=0.85)
        ax.set_title(f"{10 ** edges[i]:.2g}-{10 ** edges[i + 1]:.2g}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(variable)
        ax.set_ylabel("count")
        if log_y:
            ax.set_yscale("log")

    fig.suptitle(f"{variable} counts by ancestor time epoch")
    return fig, axes


class AncestorBuilderViz:
    """
    Visualisation for the process of building ancestors.
    """

    def __init__(self, sample_data, ancestor_data, width=800, height=400):
        self.ancestor_data = ancestor_data
        self.sample_data = sample_data
        self.width = width
        self.height = height
        self.x_pad = 20
        self.y_pad = 20
        self.x_unit = (width - 2 * self.x_pad) / sample_data.num_sites
        self.y_unit = (height - 2 * self.y_pad) / (sample_data.num_samples + 2)

    def x_trans(self, v):
        return self.x_pad + v * self.x_unit

    def y_trans(self, v):
        return self.height - (self.y_pad + v * self.y_unit)

    def draw_matrix(self, dwg, focal_sites, ancestor, current_site=None):
        A = self.sample_data.sites_genotypes[:].T
        n, m = A.shape

        for site in focal_sites:
            dwg.add(
                dwg.rect(
                    (self.x_trans(site), self.y_trans(n)),
                    (self.x_unit, n * self.y_unit),
                    fill="grey",
                )
            )

        labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
        lines = dwg.add(dwg.g(id="lines", stroke="black", stroke_width=3))
        for x in range(m + 1):
            a = self.x_trans(x), self.y_trans(0)
            b = self.x_trans(x), self.y_trans(n)
            lines.add(dwg.line(a, b))

        for y in range(n + 1):
            a = self.x_trans(0), self.y_trans(y)
            b = self.x_trans(m), self.y_trans(y)
            lines.add(dwg.line(a, b))

        for x in range(m):
            for y in range(n):
                labels.add(
                    dwg.text(
                        str(A[y, x]), (self.x_trans(x + 0.5), self.y_trans(y + 0.5))
                    )
                )
        y = n + 1
        for x in range(m):
            labels.add(
                dwg.text(
                    str(ancestor[x]), (self.x_trans(x + 0.5), self.y_trans(y + 0.5))
                )
            )

    def draw(self, ancestor_id, filename_pattern):
        start = self.ancestor_data.ancestors_start[ancestor_id]
        end = self.ancestor_data.ancestors_end[ancestor_id]
        focal_sites = self.ancestor_data.ancestors_focal_sites[ancestor_id]
        a = np.zeros(self.sample_data.num_sites, dtype=int)
        a[:] = -1
        a[start:end] = self.ancestor_data.ancestors_haplotype[ancestor_id]
        print(start, end, focal_sites, a)

        dwg = svgwrite.Drawing(size=(self.width, self.height), debug=True)
        self.draw_matrix(dwg, focal_sites, a)
        with open(filename_pattern.format(0), "w") as f:
            f.write(dwg.tostring())


def draw_edges(ts, width=800, height=600):
    """
    Returns an SVG depiction of the edges in the specified tree sequence.
    """
    dwg = svgwrite.Drawing(size=(width, height), debug=True)
    x_pad = 20
    y_pad = 20
    x_unit = (width - 2 * x_pad) / ts.sequence_length
    y_unit = (height - 2 * y_pad) / (ts.num_nodes + 1)

    def x_trans(v):
        return x_pad + v * x_unit

    def y_trans(v):
        return height - (y_pad + v * y_unit)

    lines = dwg.add(dwg.g(id="lines", stroke="black", stroke_width=3))
    left_labels = dwg.add(dwg.g(font_size=14, text_anchor="start"))
    mid_labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
    for u in range(ts.num_nodes):
        left_labels.add(dwg.text(str(u), (0, y_trans(u))))
    for x in ts.breakpoints():
        dwg.add(
            dwg.line(
                (x_trans(x), 2 * y_pad),
                (x_trans(x), height),
                stroke="grey",
                stroke_width=1,
            )
        )
        dwg.add(dwg.text(str(x), (x_trans(x), y_pad), writing_mode="tb"))

    for edge in ts.edges():
        a = x_trans(edge.left), y_trans(edge.child)
        b = x_trans(edge.right), y_trans(edge.child)
        c = x_trans(edge.left + (edge.right - edge.left) / 2), y_trans(edge.child) - 5
        mid_labels.add(dwg.text(str(edge.parent), c))
        dwg.add(dwg.circle(center=a, r=3, fill="black"))
        dwg.add(dwg.circle(center=b, r=3, fill="black"))
        lines.add(dwg.line(a, b))

    for site in ts.sites():
        assert len(site.mutations) >= 1
        mutation = site.mutations[0]
        a = x_trans(site.position), y_trans(mutation.node)
        dwg.add(dwg.circle(center=a, r=1, fill="red"))
        for mutation in site.mutations[1:]:
            a = x_trans(site.position), y_trans(mutation.node)
            dwg.add(dwg.circle(center=a, r=1, fill="blue"))

    return dwg.tostring()


def draw_ancestors(ts, width=800, height=600):
    """
    Returns an SVG depiction of the ancestors in the specified tree sequence.
    """
    dwg = svgwrite.Drawing(size=(width, height), debug=True)
    x_pad = 20
    y_pad = 20
    x_unit = (width - 2 * x_pad) / ts.sequence_length
    y_unit = (height - 2 * y_pad) / (ts.num_nodes + 1)

    def x_trans(v):
        return x_pad + v * x_unit

    def y_trans(v):
        return height - (y_pad + v * y_unit)

    lines = dwg.add(dwg.g(id="lines", stroke="black", stroke_width=3))
    left_labels = dwg.add(dwg.g(font_size=14, text_anchor="start"))
    mid_labels = dwg.add(dwg.g(font_size=14, text_anchor="middle"))
    for u in range(ts.num_nodes):
        left_labels.add(dwg.text(str(u), (0, y_trans(u))))
    for x in ts.breakpoints():
        dwg.add(
            dwg.line(
                (x_trans(x), 2 * y_pad),
                (x_trans(x), height),
                stroke="grey",
                stroke_width=1,
            )
        )
        dwg.add(dwg.text(f"{x}", (x_trans(x), y_pad), writing_mode="tb"))

    for e in ts.edgesets():
        a = x_trans(e.left), y_trans(e.parent)
        b = x_trans(e.right), y_trans(e.parent)
        c = x_trans(e.left + (e.right - e.left) / 2), y_trans(e.parent) - 5
        mid_labels.add(dwg.text(str(e.children), c))
        dwg.add(dwg.circle(center=a, r=3, fill="black"))
        dwg.add(dwg.circle(center=b, r=3, fill="black"))
        lines.add(dwg.line(a, b))

    for site in ts.sites():
        mutation = site.mutations[0]
        a = x_trans(site.position), y_trans(mutation.node)
        dwg.add(dwg.circle(center=a, r=1, fill="red"))
        for mutation in site.mutations[1:]:
            a = x_trans(site.position), y_trans(mutation.node)
            dwg.add(dwg.circle(center=a, r=1, fill="blue"))
    return dwg.tostring()


class Visualiser:
    def __init__(
        self, original_ts, sample_data, ancestor_data, inferred_ts, box_size=8
    ):
        # Make sure the singletons have been removed.
        for v in original_ts.variants():
            if np.sum(v.genotypes) < 2:
                raise ValueError("Only non singletons will be considered")
        self.box_size = box_size
        self.sample_data = sample_data
        self.original_ts = original_ts
        self.inferred_ts = inferred_ts
        self.ancestor_data = ancestor_data
        self.samples = original_ts.genotype_matrix().T
        self.num_samples = self.original_ts.num_samples
        self.num_sites = self.ancestor_data.num_sites
        node_time = inferred_ts.tables.nodes.time
        self.num_ancestors = np.where(node_time > 0)[0].shape[0]
        self.ancestors = np.zeros(
            (self.num_ancestors, original_ts.num_sites), dtype=np.uint8
        )
        for j, a in enumerate(ancestor_data.ancestors()):
            self.ancestors[j, a.start : a.end] = a.haplotype
            self.ancestors[j, : a.start] = tsinfer.UNKNOWN_ALLELE
            self.ancestors[j, a.end :] = tsinfer.UNKNOWN_ALLELE

        # TODO This only partially works for extra ancestors created by path
        # compression. We'll get -1 lines for extra ancestors created from
        # ancestors. However, extra ancestors created from matching samples
        # will break this code. We really need to just match node IDs to
        # y coordinates. Breaking up into samples and ancestors is awkward.

        # Find the site indexes for the true breakpoints
        breakpoints = list(original_ts.breakpoints())
        self.true_breakpoints = breakpoints[1:-1]

        self.top_padding = box_size
        self.left_padding = box_size
        self.bottom_padding = box_size
        self.mid_padding = 2 * box_size
        self.right_padding = box_size
        self.background_colour = ImageColor.getrgb("white")
        self.copying_outline_colour = ImageColor.getrgb("white")
        self.colours = {
            255: ImageColor.getrgb("pink"),
            0: ImageColor.getrgb("blue"),
            1: ImageColor.getrgb("red"),
        }
        self.copy_colours = {
            255: ImageColor.getrgb("white"),
            0: ImageColor.getrgb("black"),
            1: ImageColor.getrgb("green"),
        }
        self.error_colours = {
            0: ImageColor.getrgb("purple"),
            1: ImageColor.getrgb("orange"),
        }

        # Make the haplotype box
        num_haplotype_rows = 1
        self.row_map = {0: 0}

        # print(inferred_ts.tables.nodes)
        print("Ancestors = ", self.ancestors.shape, self.num_ancestors)

        num_haplotype_rows += 1
        for j in range(self.num_ancestors):
            self.row_map[j] = num_haplotype_rows
            num_haplotype_rows += 1
        num_haplotype_rows += 1
        for j in range(self.num_samples):
            self.row_map[self.num_ancestors + j] = num_haplotype_rows
            num_haplotype_rows += 1

        self.width = box_size * self.num_sites + self.left_padding + self.right_padding
        self.height = (
            self.top_padding
            + self.bottom_padding
            + self.mid_padding
            + num_haplotype_rows * box_size
        )
        self.ts_origin = (self.left_padding, self.top_padding)
        self.haplotype_origin = (self.left_padding, self.top_padding + self.mid_padding)
        self.base_image = Image.new(
            "RGB", (self.width, self.height), color=self.background_colour
        )

        b = self.box_size
        origin = self.haplotype_origin
        self.x_coordinate_map = {
            site.position: origin[0] + site.id * b for site in original_ts.sites()
        }
        self.draw_base()

    def draw_base(self):
        draw = ImageDraw.Draw(self.base_image)
        self.draw_base_haplotypes(draw)
        self.draw_true_breakpoints(draw)
        self.draw_errors(draw)

    def draw_errors(self, draw):
        b = self.box_size
        origin = self.haplotype_origin
        for site in self.original_ts.sites():
            for mut in site.mutations[1:]:
                row = self.row_map[self.num_ancestors + mut.node]
                y = row * b + origin[1]
                x = site.id * b + origin[0]
                fill = self.error_colours[int(mut.derived_state)]
                print("error at", site.id, mut.node, mut.derived_state)
                draw.rectangle([(x, y), (x + b, y + b)], fill=fill)

    def draw_true_breakpoints(self, draw):
        b = self.box_size
        origin = self.haplotype_origin
        coordinates = sorted(self.x_coordinate_map.keys())
        for bp in self.true_breakpoints:
            # Find the smallest coordinate > position
            for position in coordinates:
                if position >= bp:
                    break
            x = self.x_coordinate_map[position]
            y1 = origin[0] + self.row_map[0] * b
            y2 = origin[1] + (self.row_map[len(self.row_map) - 1] + 1) * b
            draw.line([(x, y1), (x, y2)], fill="purple", width=3)

    def draw_base_haplotypes(self, draw):
        b = self.box_size
        origin = self.haplotype_origin
        for node in self.row_map.keys():
            y = self.row_map[node] * b + origin[1] + b / 2
            x = origin[0]
            draw.text((x - b, y), str(node), fill="black")
            x = self.width - self.right_padding
            mapped = (node - len(self.row_map) + 1) * -1
            if mapped < self.num_samples:
                mapped = (mapped - self.num_samples + 1) * -1
            draw.text((x + b / 4, y), str(mapped), fill="black")

        # Draw the ancestors
        for j in range(self.ancestors.shape[0]):
            a = self.ancestors[j]
            row = self.row_map[j]
            y = row * b + origin[1]
            for k in range(self.num_sites):
                x = k * b + origin[0]
                if a[k] != -1:
                    draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])
        # Draw the samples
        for j in range(self.samples.shape[0]):
            a = self.samples[j]
            row = self.row_map[self.num_ancestors + j]
            y = row * b + origin[1]
            for k in range(self.num_sites):
                x = k * b + origin[0]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.colours[a[k]])

    def draw_haplotypes(self, filename):
        self.base_image.save(filename)

    def draw_copying_path(self, filename, child_row, parents, breakpoints):
        origin = self.haplotype_origin
        b = self.box_size
        m = self.num_sites
        image = self.base_image.copy()
        draw = ImageDraw.Draw(image)
        y = self.row_map[child_row] * b + origin[1]
        x = origin[0]
        draw.rectangle(
            [(x, y), (x + m * b, y + b)], outline=self.copying_outline_colour
        )
        for k in range(m):
            if parents[k] != -1:
                row = self.row_map[parents[k]]
                y = row * b + origin[1]
                x = k * b + origin[0]
                a = self.ancestors[parents[k], k]
                draw.rectangle([(x, y), (x + b, y + b)], fill=self.copy_colours[a])

        for position in breakpoints:
            x = self.x_coordinate_map[position]
            y1 = origin[0] + self.row_map[0] * b
            y2 = origin[1] + (self.row_map[len(self.row_map) - 1] + 1) * b
            draw.line([(x, y1), (x, y2)], fill="black")

        # Draw the positions of the sites.
        font = ImageFont.load_default()
        for site in self.original_ts.sites():
            label = f"{site.id} {site.position:.6f}"
            img_txt = Image.new("L", font.getsize(label), color="white")
            draw_txt = ImageDraw.Draw(img_txt)
            draw_txt.text((0, 0), label, font=font)
            t = img_txt.rotate(90, expand=1)
            x = origin[0] + site.id * b
            y = origin[1] - b
            image.paste(t, (x, y))
        # print("Saving", filename)
        image.save(filename)

    def draw_copying_paths(self, pattern):
        N = self.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.num_sites), dtype=int) - 1
        ts = self.inferred_ts
        site_index = {}
        sites = list(ts.sites())
        for site in ts.sites():
            site_index[site.position] = site.id
        site_index[ts.sequence_length] = ts.num_sites
        site_index[0] = 0
        for e in ts.edges():
            left = site_index[e.left]
            right = site_index[e.right]
            assert left < right
            P[e.child, left:right] = e.parent
        n = self.samples.shape[0]
        breakpoints = []
        for j in range(1, self.num_ancestors + n):
            for k in np.where(P[j][1:] != P[j][:-1])[0]:
                breakpoints.append(sites[k + 1].position)
            self.draw_copying_path(pattern.format(j - 1), j, P[j], breakpoints)


def visualise(
    ts,
    recombination_rate,
    error_rate,
    engine="C",
    box_size=8,
    perfect_ancestors=False,
    path_compression=False,
    time_chunking=False,
):

    sample_data = tsinfer.SampleData.from_tree_sequence(ts)

    if perfect_ancestors:
        ancestor_data = tsinfer.AncestorData(
            sample_data.sites_position, sample_data.sequence_length
        )
        tsinfer.build_simulated_ancestors(
            sample_data, ancestor_data, ts, time_chunking=time_chunking
        )
        ancestor_data.finalise()
    else:
        ancestor_data = tsinfer.generate_ancestors(sample_data, engine=engine)

    ancestors_ts = tsinfer.match_ancestors(
        sample_data,
        ancestor_data,
        engine=engine,
        path_compression=path_compression,
        extended_checks=True,
    )
    inferred_ts = tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        engine=engine,
        simplify=False,
        path_compression=path_compression,
        extended_checks=True,
    )

    prefix = "tmp__NOBACKUP__/"
    visualiser = Visualiser(
        ts, sample_data, ancestor_data, inferred_ts, box_size=box_size
    )
    visualiser.draw_copying_paths(os.path.join(prefix, "copying_{}.png"))

    # tsinfer.print_tree_pairs(ts, inferred_ts, compute_distances=False)
    inferred_ts = tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        engine=engine,
        simplify=True,
        path_compression=False,
        stabilise_node_ordering=True,
    )

    tsinfer.print_tree_pairs(ts, inferred_ts, compute_distances=True)
    sys.stdout.flush()
    print(
        "num_sites = ",
        inferred_ts.num_sites,
        "num_mutations= ",
        inferred_ts.num_mutations,
    )

    for site in inferred_ts.sites():
        if len(site.mutations) > 1:
            print(
                "Multiple mutations at ",
                site.id,
                "over",
                [mut.node for mut in site.mutations],
            )


def run_viz(
    n,
    L,
    rate,
    seed,
    mutation_rate=0,
    engine="C",
    perfect_ancestors=True,
    perfect_mutations=True,
    path_compression=False,
    time_chunking=True,
    error_rate=0,
):
    recomb_map = msprime.RecombinationMap.uniform_map(length=L, rate=rate, num_loci=L)
    ts = msprime.simulate(
        n,
        recombination_map=recomb_map,
        random_seed=seed,
        model="smc_prime",
        mutation_rate=mutation_rate,
    )
    if perfect_mutations:
        ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 512)
    else:
        ts = tsinfer.strip_singletons(tsinfer.insert_errors(ts, error_rate, seed))
    print("num_sites = ", ts.num_sites)

    with open("tmp__NOBACKUP__/edges.svg", "w") as f:
        f.write(draw_edges(ts))
    with open("tmp__NOBACKUP__/ancestors.svg", "w") as f:
        f.write(draw_ancestors(ts))
    visualise(
        ts,
        rate,
        0,
        engine=engine,
        box_size=26,
        perfect_ancestors=perfect_ancestors,
        path_compression=path_compression,
        time_chunking=time_chunking,
    )


def visualise_ancestors():
    ts = msprime.simulate(10, mutation_rate=2, recombination_rate=2, random_seed=3)
    ts = tsinfer.strip_singletons(ts)
    sample_data = tsinfer.SampleData.from_tree_sequence(ts)
    ancestor_data = tsinfer.generate_ancestors(sample_data)
    viz = AncestorBuilderViz(sample_data, ancestor_data)

    viz.draw(6, "ancestors_{}.svg")


def main():

    # visualise_ancestors()

    # run_viz(
    #     15, 1000, 0.0020, 11, mutation_rate=0.02, perfect_ancestors=True,
    #     perfect_mutations=True, time_chunking=True, engine="C", path_compression=False,
    #     error_rate=0.00)

    run_viz(15, 1000, 0.002, 2, engine=tsinfer.PY_ENGINE, perfect_ancestors=False)


if __name__ == "__main__":
    main()
