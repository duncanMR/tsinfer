"""
Visualisation of the copying process and ancestor generation using PIL
"""

import os
import sys
import tempfile
import ast
import msprime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import SVG, display
import tsinfer
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML


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
    df, ax, rect_height=0.4, gap=0, lw=1, title="Ancestor comparison", type="site", sim=True
):
    if type not in ["site", "pos"]:
        raise ValueError("type must be 'site' or 'pos'")
    df = df.copy()
    df.sort_values("inferred_index", inplace=True, ascending=False)
    assert len(df) > 0
    for y, (_, row) in enumerate(df.iterrows()):
        copied_left = row[f"copied_{type}_left"]
        copied_right = row[f"copied_{type}_right"]
        copied_span = row[f"copied_{type}_span"]
        inferred_left = row[f"inferred_{type}_left"]
        inferred_right = row[f"inferred_{type}_right"]

        # Adjust rectangle height for non-sim case
        current_rect_height = rect_height if sim else rect_height * 2

        if copied_span == 0:
            # No copied segment
            ax.add_patch(
                Rectangle(
                    (inferred_left, y + gap / 2),
                    inferred_right - inferred_left,
                    current_rect_height,
                    color=color_dict["inferred"],
                    label="Inferred" if y == 0 else "",
                )
            )
        else:
            ax.add_patch(
                Rectangle(
                    (inferred_left, y + gap / 2),
                    copied_left - inferred_left,
                    current_rect_height,
                    color=color_dict["inferred"],
                    label="Inferred" if y == 0 else "",
                )
            )
            ax.add_patch(
                Rectangle(
                    (copied_left, y + gap / 2),
                    copied_right - copied_left,
                    current_rect_height,
                    color=color_dict["copied"],
                    label="Copied" if y == 0 else "",
                )
            )
            ax.add_patch(
                Rectangle(
                    (copied_right, y + gap / 2),
                    inferred_right - copied_right,
                    current_rect_height,
                    color=color_dict["inferred"],
                )
            )

        if sim is True:
            # True segment below
            true_left = row[f"true_{type}_left"]
            true_right = row[f"true_{type}_right"]
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
        focal_list = row[f"focal_{type}_list"]
        assert len(focal_list) > 0
        for focal in focal_list:
            if sim is True:
                ax.vlines(
                    focal, y - rect_height, y + rect_height + gap, color="black", lw=lw
                )
            else:
                ax.vlines(
                    focal, y - gap, y + current_rect_height, color="black", lw=lw
                )

        # Add a faint gray line to separate pairs of rectangles
        ax.hlines(
            y + current_rect_height + 0.1,
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
    ]

    if sim:
        legend_elements.append(
            Rectangle((0, 0), 1, 1, color=color_dict["true"], label="True")
        )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            marker="|",
            linestyle="None",
            markersize=15,
            label="Focal site",
        )
    )
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


def plot_ancestor_segments_interactive(df, type="site", sim=True):
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
                assert len(df_plot) > 0
                # Plot segments
                plot_ancestor_segments(df_plot, ax_segments, title=title, type=type, sim=sim)

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


def compare_trees(
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

    def update_display(x_lim, position=None):
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
            if position is not None:
                label = f"Interval: [{tree_1.interval.left}, {tree_1.interval.right}); Site pos: {position}"
            else:
                label = f"Interval: [{tree_1.interval.left}, {tree_1.interval.right})"
            position_label.value = (
                label
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
            update_display(x_lim, prev_site_pos)
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
            update_display(x_lim, next_site_pos)
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
    df, cutoffs=None, var='span', type='site', title='Ancestor lengths', y_log=False, save_path=None
):
    df = df.copy()  # To avoid SettingWithCopyWarning
    if cutoffs is None:
        cutoffs = np.unique(np.percentile(df['frequency'], np.linspace(0, 100, 9)))

    if type == 'site':
        y_units = 'sites'
    elif type == 'pos':
        y_units = 'bp'
    else:
        raise ValueError("type must be 'site' or 'pos'")
    
    if var == 'span':
        if 'true_site_span' in df.columns:
            vars=[f'true_{type}_span', f'inferred_{type}_span', f'copied_{type}_span']
            var_labels = ['True (as simulated)', 'Inferred', 'Copied from']
            colors=[color_dict['true'], color_dict['inferred'], color_dict['copied']]
        else:
            vars=[f'inferred_{type}_span', f'copied_{type}_span']
            var_labels = ['Inferred', 'Copied from']
            colors=[color_dict['inferred'], color_dict['copied']]
    elif var == 'overlap_ratio':
        vars=['inferred_overlap_ratio', 'true_overlap_ratio']
        var_labels = ['Inferred overlap ratio', 'True overlap ratio']
        colors=[color_dict['inferred'], color_dict['true']]
    

    df["frequency_bin"] = pd.cut(df["frequency"], bins=cutoffs, include_lowest=True)
    df["frequency_bin"] = df["frequency_bin"].apply(
        lambda x: f"({x.left:.2f}, {x.right:.2f}]"
    )
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

