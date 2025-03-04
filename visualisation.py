"""
Visualisation of the copying process and ancestor generation using PIL
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import SVG, display, clear_output
import tsinfer
import ipywidgets as widgets
from IPython.display import display
from IPython.core.display import HTML
import os
import sys

import msprime
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import svgwrite
import tempfile
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
    which 
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
    elif var == 'overshoot':
        vars= [f'{type}_left_overshoot', f'{type}_right_overshoot',
              f'old_{type}_left_overshoot', f'old_{type}_right_overshoot']
        var_labels = ['Left overshoot (new)', 'Right overshoot (new)', 
                      'Left overshoot (old)', 'Right overshoot (old)']
        colors = ['yellowgreen', 'darkolivegreen', 'darksalmon', 'indianred']
    else:
        raise ValueError("var must be 'span' or 'overlap_ratio'")
    

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
        a = np.full(self.sample_data.num_sites, -1, dtype=int)
        a[start:end] = anc.haplotype
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


from IPython.display import HTML, display
import numpy as np
import tskit

import ipywidgets as widgets

class AncesterBuilderViz:
    def __init__(self, sample_data, true_anc, inferred_anc, df_in, inferred_anc_old=None,
                 label_col_width=50, cell_width=30, cell_height=30, filler_width=15):
        self.sample_data = sample_data
        self.true_anc = true_anc
        self.inferred_anc = inferred_anc
        self.inference_sites = np.searchsorted(sample_data.sites_position, inferred_anc.sites_position)
        self.genotypes = sample_data.sites_genotypes[self.inference_sites, :]
        self.num_samples = sample_data.num_samples
        self.num_sites = self.genotypes.shape[0]
        self.inferred_haplotypes = inferred_anc.ancestors_full_haplotype[:, :, 0]
        if inferred_anc_old is not None:
            self.inferred_haplotypes_old = inferred_anc_old.ancestors_full_haplotype
        else:  
            self.inferred_haplotypes_old = None
        assert self.genotypes.shape[0] == self.inferred_haplotypes.shape[0] #same no. sites
        df = df_in.copy()
        df['focal_AC'] = (df.frequency * sample_data.num_samples).astype('int')
        df.sort_values('inferred_index', inplace=True)
        df.set_index('inferred_index', inplace=True, drop=True)
        self.df = df
        self.site_AC = np.sum(self.genotypes, axis=1)
        #assert np.array_equal(self.site_AC[df.focal_site_left], df.focal_AC)
        
        # Dimension properties
        self.label_col_width = label_col_width
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.filler_width = filler_width

    def draw_cell(self, parts, x, y, genotype=None, fill='white', stroke_width=0.5):
        #if fills is a string, make fill_dict with 0 and 1 mapping to the string. Otherwise, use as a dict
        if isinstance(fill, str):
            fill = {0: fill, 1: fill}
        if genotype == 1:
            parts.append(f'<rect x="{x}" y="{y}" width="{self.cell_width}" height="{self.cell_height}" fill="{fill[1]}" stroke="black" stroke-width="{stroke_width}" />')
            parts.append(f'<text x="{x + self.cell_width/2}" y="{y + self.cell_height/2}" text-anchor="middle" font-weight="bold" alignment-baseline="middle">1</text>')
        elif genotype == 0:
            parts.append(f'<rect x="{x}" y="{y}" width="{self.cell_width}" height="{self.cell_height}" fill="{fill[0]}" stroke="black" stroke-width="{stroke_width}" />')
            parts.append(f'<text x="{x + self.cell_width/2}" y="{y + self.cell_height/2}" text-anchor="middle" alignment-baseline="middle">0</text>')
        else:
            parts.append(f'<rect x="{x}" y="{y}" width="{self.cell_width}" height="{self.cell_height}" fill="white" stroke="none"/>')

    def draw_symbol(self, parts, x, y, genotype, color):
        r = self.cell_width * 0.4
        x_center = x + self.cell_width/2
        y_center = y + self.cell_height/2
        if genotype == 1:
            parts.append(f'<circle cx="{x_center}" cy="{y_center}" r="{r}" fill="{color}" stroke="black" stroke-width="1" />')
            parts.append(f'<text x="{x_center}" y="{y_center}" text-anchor="middle" alignment-baseline="middle" font-weight="bold" fill="black">1</text>')
        else:
            pts = f'{x_center},{y_center - r} {x_center + r},{y_center} {x_center},{y_center + r} {x_center - r},{y_center}'
            parts.append(f'<polygon points="{pts}" fill="{color}" stroke="black" stroke-width="1" />')
            parts.append(f'<text x="{x_center}" y="{y_center}" text-anchor="middle" alignment-baseline="middle" font-weight="bold" fill="black">0</text>')

    def draw_header_col(self, parts, label, values, x, site_type, border_color='black', text_color='black'):
        parts.append(f'<rect x="{x}" y="0" width="{self.label_col_width}" height="{self.cell_height}" fill="white" stroke="{border_color}" />')
        parts.append(f'<text x="{x + self.label_col_width/2}" y="{self.cell_height/2}" text-anchor="middle" alignment-baseline="middle" font-weight="bold" fill="{text_color}">{label}</text>')
        for i, val in enumerate(values):
            y = self.cell_height + i * self.cell_height
            if (site_type[i] in [2,3]) and border_color != 'none':
                fill_color = '#e6e8f0'
                style = 'font-weight:bold'
            else:
                fill_color = 'white'
                style = ''
            parts.append(f'<rect x="{x}" y="{y}" width="{self.label_col_width}" height="{self.cell_height}" fill="{fill_color}" stroke="{border_color}" />')
            parts.append(f'<text x="{x + self.label_col_width/2}" y="{y + self.cell_height/2}" text-anchor="middle" alignment-baseline="middle" style="{style}" fill="{text_color}">{val}</text>')

    def draw_line(self, parts, x, y, col_index=0):
        if col_index == 0:
            left = x
            right = x + self.cell_width + self.filler_width
        elif col_index == -1:
            left = x - self.filler_width
            right = x + 0.5*self.cell_width
        else:
            left = x - self.cell_width/2
            right = x + self.cell_width + self.filler_width
        y_center = y + self.cell_height/2
        parts.append(f'<line x1="{left}" y1="{y_center}" x2="{right}" y2="{y_center}" stroke="black" stroke-width="5" />')

    def visualise_ancestor(self, anc_id, min_sample_count=None):
        inferred_left, inferred_right = self.df.loc[anc_id, ['inferred_site_left', 'inferred_site_right']]
        #focal_AC = self.df.loc[anc_id, 'focal_AC']
        
        sample_set_selector = self.df.sample_set_by_site[anc_id].copy()
        sample_count = np.sum(sample_set_selector, axis=1)
        max_sample_count = np.max(sample_count)
        default_min_sample_count = self.df.loc[anc_id, 'min_sample_count']
        print(f'Default min_sample_count = {default_min_sample_count}')
        if min_sample_count is None:
            min_sample_count = default_min_sample_count
            inferred_left, inferred_right = self.df.loc[anc_id, ['inferred_site_left', 'inferred_site_right']]
            print(f"Inferred left = {inferred_left}, inferred right = {inferred_right}")
        else:
            if (min_sample_count < default_min_sample_count - 1) or (min_sample_count > max_sample_count):
                raise ValueError(f"min_sample_count must be between {default_min_sample_count} and {max_sample_count}")
            sample_count_selector = sample_count >= min_sample_count
            indices = np.where(sample_count_selector)[0]
            assert indices.size > 0
            inferred_left = indices[0]
            inferred_right = indices[-1]
            print(f"Setting inferred_left = {inferred_left}, inferred_right = {inferred_right}")
            sample_set_selector[:inferred_left, :] = False
            sample_set_selector[inferred_right+1:, :] = False
        
        #focal_site_list = self.df.loc[anc_id, 'focal_site_list']
        #focal_sites_selector = np.zeros(self.num_sites, dtype=bool)
        #focal_sites_selector[focal_site_list] = True
        num_samples = self.num_samples
        num_sites = self.num_sites
        ancestor_haplotype = self.inferred_haplotypes[:, anc_id]
        
        #sites = np.arange(self.num_sites)
        #ancestor_extent_selector = (inferred_left <= sites) & (sites <= inferred_right)
        #inference_sites_selector = (self.site_AC > focal_AC) & ancestor_extent_selector
        site_type = self.df.loc[anc_id, 'site_type']
        consensus = self.df.loc[anc_id, 'consensus']
        assert len(site_type) == num_sites
        true_full_haplotype = self.df.loc[anc_id, 'true_haplotype']
        true_haplotype = true_full_haplotype[self.inference_sites]
        assert len(true_haplotype) == num_sites
        errors = (true_haplotype != ancestor_haplotype) & (true_haplotype != tskit.MISSING_DATA)
        if self.inferred_haplotypes_old is None:
            total_width = 2*self.label_col_width + num_samples * (self.cell_width + self.filler_width) + 2*(self.label_col_width + self.filler_width)
        else:
            total_width = 2*self.label_col_width + num_samples * (self.cell_width + self.filler_width) + 3*(self.label_col_width + self.filler_width)
        total_height = self.cell_width + num_sites*self.cell_height
        parts = []
        parts.append(f'<svg width="{total_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg" style="font-family: sans-serif;">')
        # Headers
        self.draw_header_col(parts, label="Site", values=list(range(num_sites)), x=0, site_type=site_type, border_color='none')
        self.draw_header_col(parts, label="AC", values=list(self.site_AC), x=self.label_col_width, site_type=site_type)
        
        # Genotypes
        for j in range(num_samples):
            x = 2*self.label_col_width + j * (self.cell_width + self.filler_width) + self.filler_width
            #parts.append(f'<rect x="{x}" y="0" width="{self.cell_width}" height="{self.cell_height}" fill="white" stroke="black" />')
            parts.append(f'<text x="{x + self.cell_width/2}" y="{self.cell_width/2}" text-anchor="middle" alignment-baseline="middle">{j}</text>')\

            for i in range(num_sites):
                genotype = self.genotypes[i, j]
                y = self.cell_height + i * self.cell_height
                if sample_set_selector[i, j]:
                    self.draw_cell(parts, x, y, genotype, fill='lightskyblue')
                    if site_type[i] == 1: #other informative site
                        #self.draw_line(parts, x, y, col_index=j)
                        if genotype == consensus[i]:
                            self.draw_symbol(parts, x, y, genotype, color='mediumseagreen')
                        else:
                            self.draw_symbol(parts, x, y, genotype, color='red')
                    elif site_type[i] == 2: #inference site
                        self.draw_line(parts, x, y, col_index=j)
                        #assert ancestor_haplotype[i] == consensus[i]
                        if genotype == consensus[i]:
                            self.draw_symbol(parts, x, y, genotype, color='orange')
                        else:
                            self.draw_symbol(parts, x, y, genotype, color='red')
                    elif site_type[i] == 3: #focal site
                        self.draw_line(parts, x, y, col_index=j)
                        self.draw_symbol(parts, x, y, genotype, color='royalblue')
                else:
                    in_ancestor = (inferred_left <= i) & (i <= inferred_right)
                    if site_type[i] in [2,3] and in_ancestor: #inference or focal
                        self.draw_line(parts, x, y, col_index=j)
                        self.draw_cell(parts, x, y, genotype, fill={0: 'white', 1:'aliceblue'}, stroke_width=1)
                    else:
                        self.draw_cell(parts, x, y, genotype, fill={0: 'white', 1:'aliceblue'})
            x = x + self.cell_width
        x = x + 2*self.filler_width
        #parts.append(f'<rect x="{x}" y="{0}" width="{self.label_col_width}" height="{self.cell_height}" fill="white" stroke="black" />')
        parts.append(f'<text x="{x + self.cell_width/2}" y="{self.cell_height/2}" text-anchor="middle" alignment-baseline="middle">Inferred</text>')
        
        #Inferred ancestor
        for i in range(num_sites):
            y = self.cell_height + i * self.cell_height
            genotype = ancestor_haplotype[i]
            in_ancestor = (inferred_left <= i) & (i <= inferred_right)
            if genotype != tskit.MISSING_DATA and in_ancestor:
                self.draw_cell(parts, x, y, genotype=genotype, fill='#b6b8c2', stroke_width=2)
                if site_type[i] == 2: #inference site
                    self.draw_line(parts, x, y, col_index=-1)
                    self.draw_symbol(parts, x, y, genotype, color='orange')
                elif site_type[i] == 3: #focal site
                    self.draw_line(parts, x, y, col_index=-1)
                    self.draw_symbol(parts, x, y, genotype, color='royalblue')
                    
        #True ancestor
        parts.append(f'<text x="{x + self.label_col_width/2 + 1.1*self.cell_width}" y="{self.cell_height/2}" text-anchor="middle" alignment-baseline="middle">True</text>')
        x = x + self.cell_width + self.filler_width
        for i in range(num_sites):
            y = self.cell_height + i * self.cell_height
            genotype = true_haplotype[i]
            if genotype != tskit.MISSING_DATA:
                if errors[i]:
                    if site_type[i] == 2: #inference site
                        self.draw_cell(parts, x, y, genotype, fill='lightcoral', stroke_width=2)
                        self.draw_symbol(parts, x, y, genotype, color='red')
                    else:
                        self.draw_cell(parts, x, y, genotype, fill='lightcoral', stroke_width=2)
                else:
                    self.draw_cell(parts, x, y, genotype, fill='#89c48f', stroke_width=2)
                    if site_type[i] == 2: #inference site
                        self.draw_symbol(parts, x, y, genotype, color='orange')
                    elif site_type[i] == 3: #focal site
                        self.draw_symbol(parts, x, y, genotype, color='royalblue')

        #Prev method ancestor
        
        if self.inferred_haplotypes_old is not None:
            ancestor_haplotype_old = self.inferred_haplotypes_old[:, anc_id]
            parts.append(f'<text x="{x + self.cell_width/2}" y="{self.cell_height/2}" text-anchor="middle" alignment-baseline="middle">Inferred (old)</text>')
            x = x + self.cell_width + self.filler_width
            for i in range(num_sites):
                y = self.cell_height + i * self.cell_height
                genotype = ancestor_haplotype_old[i]
                #in_ancestor = (inferred_left <= i) & (i <= inferred_right)
                if genotype != tskit.MISSING_DATA:
                    self.draw_cell(parts, x, y, genotype=genotype, fill='mediumpurple', stroke_width=2)
                    if site_type[i] == 2: #inference site
                        self.draw_line(parts, x, y, col_index=-1)
                        self.draw_symbol(parts, x, y, genotype, color='orange')
                    elif site_type[i] == 3: #focal site
                        self.draw_symbol(parts, x, y, genotype, color='royalblue')
        
        parts.append("</svg>")
        svg_str = "".join(parts)
        return svg_str

    def visualise(self, height=800):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        indices = list(self.df.index.values)
        anc_id_widget = widgets.IntText(value=min(indices), description='Ancestor ID:')
        left_button = widgets.Button(description='Left')
        right_button = widgets.Button(description='Right')
        min_sample_widget = widgets.IntText(
            value=self.df.loc[anc_id_widget.value, 'min_sample_count'],
            description='Min Sample Count:')
        svg_out = widgets.HTML(value="")
        scroll_container = widgets.Box([svg_out],
            layout=widgets.Layout(overflow_y='auto', border='1px solid gray', height=f"{height}px"))
        controls = widgets.HBox([left_button, anc_id_widget, right_button, min_sample_widget])
        container = widgets.VBox([controls, scroll_container])
        
        def update_svg(*args):
            default_min = self.df.loc[anc_id_widget.value, 'min_sample_count']
            focal_AC = self.df.loc[anc_id_widget.value, 'focal_AC']
            if min_sample_widget.value < default_min:
                min_sample_widget.value = default_min
            elif min_sample_widget.value > focal_AC:
                min_sample_widget.value = focal_AC
            svg_str = self.visualise_ancestor(anc_id_widget.value, min_sample_count=min_sample_widget.value)
            svg_out.value = svg_str if svg_str is not None else ""
        
        def on_left_clicked(b):
            current_index = indices.index(anc_id_widget.value)
            new_index = max(0, current_index - 1)
            anc_id_widget.value = indices[new_index]
            min_sample_widget.value = self.df.loc[anc_id_widget.value, 'min_sample_count']
            update_svg()
        
        def on_right_clicked(b):
            current_index = indices.index(anc_id_widget.value)
            new_index = min(len(indices) - 1, current_index + 1)
            anc_id_widget.value = indices[new_index]
            min_sample_widget.value = self.df.loc[anc_id_widget.value, 'min_sample_count']
            update_svg()
        
        anc_id_widget.observe(lambda change: update_svg() if change['name'] == 'value' else None, names='value')
        min_sample_widget.observe(lambda change: update_svg() if change['name'] == 'value' else None, names='value')
        left_button.on_click(on_left_clicked)
        right_button.on_click(on_right_clicked)
        
        display(container)
        update_svg()
