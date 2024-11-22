"""
Visualisation of the copying process and ancestor generation using PIL
"""

import os
import sys
import tempfile

import msprime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import SVG, display

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import svgwrite
import tskit
import tsinfer

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

pd.options.mode.chained_assignment = None
sns.set_theme(style="whitegrid")

color_dict = {
    "inferred": "#d62728",  # red
    "copied": "#ff7f0e",  # orange
    "true": "#1f77b4",  # blue
}


def plot_ancestor_segments(
    df, ax, rect_height=0.4, gap=0, lw=1, title="Ancestor comparison", type="site"
):
    if type not in ["site", "pos"]:
        raise ValueError("type must be 'site' or 'pos'")
    df = df.copy()
    df.sort_values("inferred_index", inplace=True, ascending=False)

    for y, (_, row) in enumerate(df.iterrows()):
        copied_left = row[f"copied_{type}_left"]
        copied_right = row[f"copied_{type}_right"]
        inferred_left = row[f"inferred_{type}_left"]
        inferred_right = row[f"inferred_{type}_right"]
        true_left = row[f"true_{type}_left"]
        true_right = row[f"true_{type}_right"]
        # Inferred/copied above
        ax.add_patch(
            Rectangle(
                (inferred_left, y + gap / 2),
                copied_left - inferred_left,
                rect_height,
                color=color_dict["inferred"],
                label="Inferred" if y == 0 else "",
            )
        )
        ax.add_patch(
            Rectangle(
                (copied_left, y + gap / 2),
                copied_right - copied_left,
                rect_height,
                color=color_dict["copied"],
                label="Copied" if y == 0 else "",
            )
        )
        ax.add_patch(
            Rectangle(
                (copied_right, y + gap / 2),
                inferred_right - copied_right,
                rect_height,
                color=color_dict["inferred"],
            )
        )
        # True segment below
        ax.add_patch(
            Rectangle(
                (true_left, y - rect_height - gap / 2),
                true_right - true_left,
                rect_height,
                color=color_dict["true"],
                label="True" if y == 0 else "",
            )
        )

        # Plot focal sites as vertical lines
        focal_list = np.array(row[f"focal_{type}_list"])
        for focal in focal_list:
            focal_pos = float(focal)
            ax.vlines(
                focal, y - rect_height, y + rect_height + gap, color="black", lw=lw
            )

        # Add a faint gray line to separate pairs of rectangles
        ax.hlines(
            y + rect_height + 0.1,
            xmin=df[f"inferred_{type}_left"].min() - 1e6,
            xmax=df[f"inferred_{type}_right"].max() + 1e6,
            color="gray",
            lw=1,
            linestyle="--",
            alpha=0.5,
        )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["inferred_index"].values)
    xlabel = "Site index" if type == "site" else "Genomic position (bp)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Ancestor ID")
    ax.grid(False)
    ax.set_title(title)
    x_min = df[f"inferred_{type}_left"].min()
    x_max = df[f"inferred_{type}_right"].max()
    margin = (x_max - x_min) * 0.01
    ax.set_xlim(x_min - margin, x_max + margin)

    # Create custom legend handles with vertical line for 'Focal site'
    legend_elements = [
        Rectangle((0, 0), 1, 1, color=color_dict["inferred"], label="Inferred"),
        Rectangle((0, 0), 1, 1, color=color_dict["copied"], label="Copied"),
        Rectangle((0, 0), 1, 1, color=color_dict["true"], label="True"),
        Line2D(
            [0],
            [0],
            color="black",
            marker="|",
            linestyle="None",
            markersize=15,
            label="Focal site",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")  # Legend inside the plot


def plot_sample_sets(df, ax):
    df = df.copy()
    df.sort_values("inferred_index", inplace=True, ascending=True)
    y_positions = range(len(df))

    # Create a stacked horizontal bar chart
    # ax.barh(y=y_positions, width=df['num_allowed_conflicts'], label='Number of allowed conflicts', color='#29963a')
    # ax.barh(y=y_positions, width=df['min_sample_set_size'], left=df['num_allowed_conflicts'], label='Remaining sample set', color='#157023')

    ax.set_xlabel("Number of allowed conflicts")
    ax.set_ylabel("Ancestor ID")  # No y-axis label since it's shared
    ax.set_yticks(range(len(df)))  # Hide y-axis ticks
    ax.set_yticklabels(df["inferred_index"].values)
    ax.invert_yaxis()  # Match the order of the segments plot
    # ax.legend()
    ax.grid(False)


def create_ancestor_segments_plotter(df, type="site"):
    # Get the unique sample_frac values
    sample_frac_list = sorted(df["sample_frac"].unique())
    sample_frac_index = 0  # Start at the first sample_frac value
    sample_frac = sample_frac_list[sample_frac_index]

    # Create widgets
    prev_button = widgets.Button(description="Previous")
    next_button = widgets.Button(description="Next")
    sample_frac_slider = widgets.SelectionSlider(
        options=sample_frac_list,
        value=sample_frac,
        description="Sample Frac:",
        continuous_update=False,
    )
    num_ancestors_input = widgets.IntText(value=10, description="Num Ancestors:")

    # Output widget for the plot
    output = widgets.Output()

    # Vertical slider (scrollbar) for scrolling through inferred_node
    inferred_nodes = df["inferred_index"].unique()
    inferred_nodes.sort()
    inferred_node_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,  # Will be updated in update_plot
        step=1,
        orientation="vertical",
        readout=False,  # Hide the default readout
        continuous_update=False,
        layout=widgets.Layout(height="400px", transform="rotate(180deg)"),
    )
    # Label to display the current inferred_node value
    slider_value_label = widgets.Label(value="")

    up_button = widgets.Button(
        description="", icon="arrow-up", layout=widgets.Layout(width="30px")
    )
    down_button = widgets.Button(
        description="", icon="arrow-down", layout=widgets.Layout(width="30px")
    )

    # Define the function to update the plot
    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            # Get current parameter values
            sample_frac = sample_frac_slider.value
            num_ancestors = num_ancestors_input.value
            slider_value = (
                inferred_node_slider.value
            )  # This will be from 0 to max_slider_value
            # Reverse the start_idx
            start_idx = inferred_node_slider.max - slider_value

            # Filter df based on sample_frac
            df_filtered = df[df["sample_frac"] == sample_frac].copy()
            # Sort ascending by inferred_node
            df_filtered.sort_values("inferred_index", inplace=True)
            df_filtered.reset_index(drop=True, inplace=True)

            if df_filtered.empty:
                print("No data available for the selected parameters.")
                inferred_node_slider.max = 0
                inferred_node_slider.value = 0
                slider_value_label.value = ""
                return

            # Update inferred_nodes based on the filtered data
            inferred_nodes_filtered = df_filtered["inferred_index"].unique()
            inferred_nodes_filtered.sort()

            # Update the inferred_node_slider max value
            max_slider_value = max(len(inferred_nodes_filtered) - num_ancestors, 0)
            inferred_node_slider.max = max_slider_value
            inferred_node_slider.min = 0
            # Adjust start_idx if necessary
            if start_idx > max_slider_value:
                start_idx = max_slider_value
                inferred_node_slider.value = inferred_node_slider.max - start_idx

            # Get the subset of inferred_nodes to display
            end_idx = start_idx + num_ancestors
            inferred_nodes_to_display = inferred_nodes_filtered[start_idx:end_idx]

            # Update the slider_value_label to display the current inferred_node
            if start_idx < len(inferred_nodes_filtered):
                current_inferred_node = inferred_nodes_filtered[start_idx]
                slider_value_label.value = f"{current_inferred_node}"
            else:
                slider_value_label.value = ""

            # Filter df_filtered to include only the inferred_nodes_to_display
            df_plot = df_filtered[
                df_filtered["inferred_index"].isin(inferred_nodes_to_display)
            ]
            df_plot.sort_values("inferred_index", ascending=False, inplace=True)
            df_plot.reset_index(drop=True, inplace=True)

            if df_plot.empty:
                print("No data available for the selected parameters.")
            else:
                # Plot the data
                min_freq = df_plot["frequency"].min()
                max_freq = df_plot["frequency"].max()
                title = f"Ancestor segments (Freq: {min_freq:.4f}-{max_freq:.4f}, Sample Frac: {sample_frac})"

                fig, ax_segments = plt.subplots(
                    figsize=(20, 7)
                )  # , gridspec_kw={'width_ratios': [4, 1]})

                # Plot segments
                plot_ancestor_segments(df_plot, ax_segments, title=title, type=type)

                plt.tight_layout()
                plt.show()

    # Event handlers for widgets
    def on_prev_clicked(b):
        nonlocal sample_frac_index
        if sample_frac_index > 0:
            sample_frac_index -= 1
            sample_frac_slider.value = sample_frac_list[sample_frac_index]

    def on_next_clicked(b):
        nonlocal sample_frac_index
        if sample_frac_index < len(sample_frac_list) - 1:
            sample_frac_index += 1
            sample_frac_slider.value = sample_frac_list[sample_frac_index]

    def on_sample_frac_change(change):
        nonlocal sample_frac_index
        sample_frac = change["new"]
        sample_frac_index = sample_frac_list.index(sample_frac)
        update_plot()

    def on_num_ancestors_change(change):
        update_plot()

    def on_inferred_node_slider_change(change):
        update_plot()

    def on_down_clicked(b):
        if inferred_node_slider.value > inferred_node_slider.min:
            inferred_node_slider.value -= 1

    def on_up_clicked(b):
        if inferred_node_slider.value < inferred_node_slider.max:
            inferred_node_slider.value += 1

    # Set up observers
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)
    sample_frac_slider.observe(on_sample_frac_change, names="value")
    num_ancestors_input.observe(on_num_ancestors_change, names="value")
    inferred_node_slider.observe(on_inferred_node_slider_change, names="value")
    up_button.on_click(on_up_clicked)
    down_button.on_click(on_down_clicked)

    # Adjust the width of up and down buttons
    up_button.layout.width = "60px"
    down_button.layout.width = "60px"

    # Layout widgets
    controls_top = widgets.HBox(
        [prev_button, sample_frac_slider, next_button, num_ancestors_input]
    )
    # Arrange the scrollbar with Up and Down buttons and label
    scrollbar = widgets.VBox(
        [up_button, inferred_node_slider, down_button, slider_value_label]
    )
    # Place scrollbar on the left
    display(widgets.VBox([controls_top, widgets.HBox([scrollbar, output])]))

    update_plot()


def plot_tree(
    ts,
    sites,
    tree=None,
    title=None,
    position=None,
    index=None,
    time_scale=None,
    size=(1000, 400),
):
    if position is not None:
        tree = ts.at(position)
    elif index is not None:
        tree = ts.at_index(index)
    mut_labels = {}
    for mut in ts.mutations():
        ancestral = ts.site(mut.site).ancestral_state
        derived = mut.derived_state
        if mut.site in sites:
            mut_labels[mut.id] = f"{ancestral}{mut.site}{derived}"
        else:
            mut_labels[mut.id] = ""
    return tree.draw_svg(
        mutation_labels=mut_labels,
        size=size,
        y_axis=True,
        title=title,
        time_scale=time_scale,
    )


def make_mut_labels(ts, sites_pos):
    sites = np.where(np.isin(ts.sites_position, sites_pos))[0]
    mut_labels = {}
    for mut in ts.mutations():
        ancestral = ts.site(mut.site).ancestral_state
        derived = mut.derived_state
        if mut.site in sites:
            mut_labels[mut.id] = f"{ancestral}{mut.site}{derived}"
        else:
            mut_labels[mut.id] = ""
    return mut_labels


def tree_comparison(
    ts_1,
    ts_2,
    sites_pos,
    title_1="Inferred TS",
    title_2="True TS",
    time_scale=None,
    max_num_trees=3,
    size=(500, 600),
):
    # Ensure sites_pos is a sorted numpy array
    sites_pos = np.sort(np.array(sites_pos))

    # Initialize trees at the starting position
    starting_pos = sites_pos[0]
    tree_1 = ts_1.at(starting_pos)

    # Determine mutation labels for ts_1 and ts_2
    mut_labels_1 = make_mut_labels(ts_1, sites_pos)
    mut_labels_2 = make_mut_labels(ts_2, sites_pos)

    # Initialize output widgets
    output = widgets.Output()
    breakpoints = ts_2.breakpoints(as_array=True)

    # Create navigation buttons
    prev_site_button = widgets.Button(description="Previous site")
    next_site_button = widgets.Button(description="Next site")
    prev_tree_button = widgets.Button(description="Previous tree")
    next_tree_button = widgets.Button(description="Next tree")
    position_label = widgets.Label(
        value=f"Position: {tree_1.interval.left}-{tree_1.interval.right}"
    )
    controls = widgets.HBox(
        [
            prev_site_button,
            next_site_button,
            prev_tree_button,
            next_tree_button,
            position_label,
        ]
    )

    # Display controls and output
    display(controls, output)

    def update_display(x_lim):
        with output:
            output.clear_output(wait=True)
            # Generate SVG for tree_1
            svg1 = tree_1.draw_svg(
                mutation_labels=mut_labels_1,
                size=size,
                y_axis=True,
                title=title_1,
                time_scale=time_scale,
            )
            num_trees = np.sum((breakpoints > x_lim[0]) & (breakpoints < x_lim[1])) + 1
            multi_width = size[0] * min(max_num_trees, num_trees)
            # Generate SVG for ts_2 within x_lim
            svg2 = ts_2.draw_svg(
                mutation_labels=mut_labels_2,
                size=(int(multi_width), size[1]),
                y_axis=True,
                title=title_2,
                time_scale=time_scale,
                x_lim=x_lim,
                max_num_trees=max_num_trees,
            )
            # Wrap SVGs in HTML widgets
            html1 = widgets.HTML(value=svg1)
            html2 = widgets.HTML(value=svg2)
            # Display SVGs side by side
            display(widgets.HBox([html1, html2]))
            # Update position label
            position_label.value = (
                f"Position: {tree_1.interval.left}-{tree_1.interval.right}"
            )

    def on_prev_tree_clicked(b):
        nonlocal tree_1
        if tree_1.index > 0:
            tree_1.prev()
            x_lim = tree_1.interval
            update_display(x_lim)

    def on_next_tree_clicked(b):
        nonlocal tree_1
        if tree_1.index < ts_1.num_trees - 1:
            tree_1.next()
            x_lim = tree_1.interval
            update_display(x_lim)

    def on_prev_site_clicked(b):
        nonlocal tree_1
        current_pos = tree_1.interval.left
        prev_sites = sites_pos[sites_pos < current_pos]
        if len(prev_sites) > 0:
            prev_site_pos = prev_sites[-1]  # Last site less than current position
            tree_1.seek(prev_site_pos)
            x_lim = tree_1.interval
            update_display(x_lim)
        else:
            print("No previous site.")

    def on_next_site_clicked(b):
        nonlocal tree_1
        current_pos = tree_1.interval.right
        next_sites = sites_pos[sites_pos > current_pos]
        if len(next_sites) > 0:
            next_site_pos = next_sites[0]  # First site greater than current position
            tree_1.seek(next_site_pos)
            x_lim = tree_1.interval
            update_display(x_lim)
        else:
            print("No next site.")

    # Attach event handlers
    prev_tree_button.on_click(on_prev_tree_clicked)
    next_tree_button.on_click(on_next_tree_clicked)
    prev_site_button.on_click(on_prev_site_clicked)
    next_site_button.on_click(on_next_site_clicked)

    # Initial display
    x_lim = tree_1.interval
    update_display(x_lim)


def plot_ancestor_boxplot(
    df, cutoffs, var_dict, var_labels, title, y_units="bp", y_log=False, save_path=None
):
    df = df.copy()  # To avoid SettingWithCopyWarning
    df["frequency_bin"] = pd.cut(df["frequency"], bins=cutoffs, include_lowest=True)
    df["frequency_bin"] = df["frequency_bin"].apply(
        lambda x: f"({x.left:.2f}, {x.right:.2f}]"
    )

    # Extract variables and corresponding colors from var_dict
    vars = list(var_dict.keys())
    colors = list(var_dict.values())

    # Create a mapping from variable names to their formatted labels
    var_label_mapping = dict(zip(vars, var_labels))

    # Melt the dataframe and map the 'type' column to the formatted labels
    lengths_df = pd.melt(
        df,
        id_vars=["frequency_bin"],
        value_vars=vars,
        var_name="type",
        value_name="value",
    )
    lengths_df["type"] = lengths_df["type"].map(var_label_mapping)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 7), gridspec_kw={"height_ratios": [4, 1]}
    )

    # Use colors from var_dict in the boxplot palette, mapped by formatted labels
    palette = {label: color for label, color in zip(var_labels, colors)}
    sns.boxplot(
        x="frequency_bin",
        y="value",
        hue="type",
        data=lengths_df,
        palette=palette,
        saturation=1,
        hue_order=var_labels,
        ax=ax1,
    )

    # Adjust legend position to the right of the plot
    ax1.legend(title="Ancestor type", loc="upper left", bbox_to_anchor=(1.02, 1))

    ax1.set_xlabel("Ancestor age interval (frequency)")
    ax1.set_ylabel(f"Ancestor length ({y_units})")
    ax1.set_title(title)
    if y_log:
        ax1.set_yscale("log")

    # Plot quantile counts
    quantile_counts = df["frequency_bin"].value_counts(sort=False)
    sns.barplot(
        x=quantile_counts.index,
        y=quantile_counts.values,
        ax=ax2,
        color="#ced4da",
        linewidth=1,
        edgecolor="black",
    )
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel("")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Add margin for legend on the right

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()


def compare_ancestors(ancestor_dict):
    assert len(ancestor_dict) == 2
    offsets = [0, 2e-3]
    colors = ["red", "blue"]
    fig, ax = plt.subplots(figsize=(15, 15))
    for index, (type, ancestor_data) in enumerate(ancestor_dict.items()):
        anc_time = ancestor_data.ancestors_time[:] + offsets[index]
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]

        ax.hlines(y=anc_time, xmin=start, xmax=end, label=type, colors=colors[index])

    ax.set_xlabel("Position")
    ax.set_ylabel("Time (frequency)")
    ax.legend(title="Method")
    plt.show()


def plot_perf_by_index(df, alpha=0.8, title_core=None):
    g = sns.FacetGrid(
        df,
        row="num_samples",
        col="iteration",
        hue="engine",
        margin_titles=True,
        sharey="row",
        sharex=False,
        height=3,
        aspect=1.3,
        palette="tab10",
    )

    # Map the plot function to each facet
    g.map(
        sns.lineplot,
        "anc_index",
        "duration",
        alpha=alpha,
    )

    g.set_axis_labels("Ancestor index", "Duration (s)")
    g.set(yscale="log")
    g.add_legend()

    if title_core is not None:
        title = f"{title_core}: time taken per ancestor"
        plt.subplots_adjust(top=0.9)
        g.figure.suptitle(title, y=0.99, fontweight="bold")

    plt.show()


def plot_speedup(ax, df):
    g_df = df.groupby(["num_samples", "engine"]).sum("duration").reset_index()
    c_duration = g_df[g_df["engine"] == "C"][["num_samples", "duration"]].set_index(
        "num_samples"
    )
    speedup_df = g_df[g_df["engine"] != "C"]
    speedup_df["speedup"] = speedup_df.apply(
        lambda row: c_duration.loc[row["num_samples"], "duration"] / row["duration"],
        axis=1,
    )
    num_alts = len(speedup_df["engine"].unique())
    palette = sns.color_palette("tab10")[1 : num_alts + 1]

    sns.barplot(
        data=speedup_df,
        x="num_samples",
        y="speedup",
        hue="engine",
        palette=palette,
        saturation=1,
        ax=ax,
    ).set(
        xlabel="Number of sample nodes",
        ylabel="Speedup over C method",
        title="Speedup vs C (total time of all ancestors)",
    )
    sns.move_legend(ax, "lower right")


def plot_perf_summary(
    output_dir,
    output_prefix,
    plot_path=None,
    title_core=None,
    alpha=0.5,
    engine_label_map=None,
):
    df = pd.read_csv(f"{output_dir}/{output_prefix}_log.tsv", sep="\t")

    if engine_label_map is None:
        engine_label_map = {"C": "C", "N": "Numba", "N2": "Numba alt"}
    df["engine"] = df["engine"].map(engine_label_map)

    plot_perf_by_index(df, alpha=alpha, title_core=title_core)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    fig.subplots_adjust(top=0.7)
    if title_core is not None:
        title = f"{title_core}: performance summary"
        fig.suptitle(
            title,
            y=0.99,
            x=0.55,
            fontweight="bold",
        )

    second_df = df[df.iteration == 1]
    assert len(second_df) > 0
    sns.barplot(
        data=second_df,
        x="num_samples",
        y="duration",
        hue="engine",
        estimator=np.median,
        errorbar=("pi", 90),
        capsize=0.1,
        palette="tab10",
        saturation=1,
        ax=axs[0],
    ).set(
        xlabel="Number of sample nodes",
        ylabel="Median duration (s)",
        title="Median time taken per ancestor (2nd iteration)",
        yscale="log",
    )

    plot_speedup(axs[1], second_df)

    # sns.lineplot(
    #     data=df[df.engine == "C"],
    #     x="anc_index",
    #     y="span",
    #     hue="num_samples",
    #     palette="Dark2",
    #     ax=axs[2, 1],
    #     alpha=alpha,
    # ).set(
    #     xlabel="Ancestor index",
    #     ylabel="Ancestor span",
    #     title=f"Ancestor span by index",
    # )

    # Adjust layout
    plt.tight_layout()
    plt.show()

    if plot_path is not None:
        fig.savefig(plot_path)

    return df


def compare_ancestors(ancestor_dict):
    assert len(ancestor_dict) == 2
    offsets = [0, 2e-3]
    colors = ["red", "blue"]
    fig, ax = plt.subplots(figsize=(15, 15))
    for index, (type, ancestor_data) in enumerate(ancestor_dict.items()):
        anc_time = ancestor_data.ancestors_time[:] + offsets[index]
        start = ancestor_data.ancestors_start[:]
        end = ancestor_data.ancestors_end[:]

        ax.hlines(y=anc_time, xmin=start, xmax=end, label=type, colors=colors[index])

    ax.set_xlabel("Position")
    ax.set_ylabel("Time (frequency)")
    ax.legend(title="Method")
    plt.show()


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
        A_t = self.sample_data.sites_genotypes[:]
        # Need to remove fixed sites
        fixed_sites = np.all(A_t == A_t[:, [0]], axis=1)
        A = (A_t[~fixed_sites]).T
        n, m = A.shape
        print(f"Drawing matrix with {n} samples and {m} sites")
        print(f"Ancestor length = {len(ancestor)}")
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

    def draw(self, ancestor_id):
        anc = self.ancestor_data.ancestor(ancestor_id)
        focal_sites = anc.focal_sites
        start = anc.start
        end = anc.end
        a = anc.full_haplotype[start:end]

        dwg = svgwrite.Drawing(size=(self.width, self.height), debug=True)
        self.draw_matrix(dwg, focal_sites, a)
        # with open(filename_pattern.format(0), "w") as f:
        #    f.write(dwg.tostring())
        with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f:
            f.write(dwg.tostring().encode("utf-8"))
            temp_filename = f.name
        display(SVG(filename=temp_filename))


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
        dwg.add(dwg.circle(center=a, r=5, fill="red"))
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as f:
        f.write(dwg.tostring().encode("utf-8"))
        temp_filename = f.name
    display(SVG(filename=temp_filename))


#    return dwg.tostring()


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
            self.ancestors[j, : a.start] = tskit.MISSING_DATA
            self.ancestors[j, a.end :] = tskit.MISSING_DATA

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
            0: ImageColor.getrgb("#4dabf7"),
            1: ImageColor.getrgb("#ff8787"),
        }
        self.copy_colours = {
            255: ImageColor.getrgb("white"),
            0: ImageColor.getrgb("#1971c2"),
            1: ImageColor.getrgb("#800000"),
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
        print(f"origin: {origin}")
        for node in self.row_map.keys():
            print(f"Drawing node {node} at row {self.row_map[node]}")
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
            label = f"{site.id}: {site.position}"
            # Use draw.textbbox to get the size of the text
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            img_txt = Image.new("L", (text_width, text_height), color="white")
            draw_txt = ImageDraw.Draw(img_txt)
            draw_txt.text((0, 0), label, font=font)
            t = img_txt.rotate(90, expand=1)
            x = origin[0] + site.id * b
            y = origin[1] - b
            image.paste(t, (x, y))
        image.save(filename)

    def draw_copying_paths(self, pattern):
        N = self.num_ancestors + self.samples.shape[0]
        P = np.zeros((N, self.num_sites), dtype=int) - 1
        ts = self.inferred_ts
        site_index = {}
        sites = list(self.original_ts.sites())
        for site in self.original_ts.sites():
            site_index[site.position] = site.id
        site_index[ts.sequence_length] = self.original_ts.num_sites
        site_index[0] = 0
        for e in ts.edges():
            left = site_index[e.left]
            right = site_index[e.right]
            assert left < right
            # print(f'{e.child}s parent is {e.parent} from {left} to {right}')
            P[e.child, left:right] = e.parent
            # print(P)
        print(P)
        n = self.samples.shape[0]
        breakpoints = []
        for j in range(1, self.num_ancestors + n):
            print(f"Drawing copying path for {j} with parent P[j] = {P[j]}")
            for k in np.where(P[j][1:] != P[j][:-1])[0]:
                breakpoints.append(sites[k + 1].position)
            self.draw_copying_path(pattern.format(j), j, P[j], breakpoints)


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
    )
    inferred_ts = tsinfer.match_samples(
        sample_data,
        ancestors_ts,
        engine=engine,
        post_process=False,
        path_compression=path_compression,
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
        post_process=True,
        path_compression=False,
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
    recombination_rate,
    seed,
    mutation_rate=0,
    engine="C",
    perfect_ancestors=True,
    perfect_mutations=False,
    path_compression=False,
    time_chunking=True,
    error_rate=0,
):
    # ts0 = msprime.sim_ancestry(n, sequence_length=L, random_seed=seed, recombination_rate=recombination_rate)
    # ts = msprime.sim_mutations(ts0, rate=mutation_rate, random_seed=seed)
    base_ts = msprime.sim_ancestry(
        3, sequence_length=1000, random_seed=5, recombination_rate=recombination_rate
    )
    ts = msprime.sim_mutations(
        base_ts, rate=mutation_rate, random_seed=4, model="binary"
    )
    # ts = msprime.simulate(
    #     n,
    #     recombination_map=recomb_map,
    #     random_seed=seed,
    #     model="smc_prime",
    #     mutation_rate=mutation_rate,
    # )

    if perfect_mutations:
        ts = tsinfer.insert_perfect_mutations(ts, delta=1 / 512)
    else:
        ts = tsinfer.strip_singletons(ts)
    print("num_sites = ", ts.num_sites)

    with open("tmp__NOBACKUP__/edges.svg", "w") as f:
        f.write(draw_edges(ts))
    with open("tmp__NOBACKUP__/ancestors.svg", "w") as f:
        f.write(draw_ancestors(ts))
    visualise(
        ts,
        recombination_rate,
        error_rate=0,
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

    run_viz(
        12,
        500,
        recombination_rate=0.004,
        mutation_rate=0.001,
        seed=2,
        engine=tsinfer.C_ENGINE,
        perfect_ancestors=False,
    )


if __name__ == "__main__":
    main()
