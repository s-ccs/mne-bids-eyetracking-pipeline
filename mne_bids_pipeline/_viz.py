from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def plot_auto_scores(
    auto_scores: dict[str, Any], *, ch_types: list[str]
) -> list[Figure]:
    # Plot scores of automated bad channel detection.
    import matplotlib.pyplot as plt
    import seaborn as sns

    ch_types_ = list(ch_types)
    if "meg" in ch_types_:  # split it
        idx = ch_types_.index("meg")
        ch_types_[idx] = "grad"
        ch_types_.insert(idx + 1, "mag")

    figs: list[Figure] = []
    for ch_type in ch_types_:
        # Only select the data for mag or grad channels.
        ch_subset = auto_scores["ch_types"] == ch_type
        if not ch_subset.any():
            continue  # e.g., MEG+EEG data with finding bads with MF enabled
        ch_names = auto_scores["ch_names"][ch_subset]
        scores = auto_scores["scores_noisy"][ch_subset]
        limits = auto_scores["limits_noisy"][ch_subset]
        bins = auto_scores["bins"]  # The the windows that were evaluated.

        # We will label each segment by its start and stop time, with up to 3
        # digits before and 3 digits after the decimal place (1 ms precision).
        bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

        # We store the data in a Pandas DataFrame. The seaborn heatmap function
        # we will call below will then be able to automatically assign the
        # correct labels to all axes.
        data_to_plot = pd.DataFrame(
            data=scores,
            columns=pd.Index(bin_labels, name="Time (s)"),
            index=pd.Index(ch_names, name="Channel"),
        )

        # First, plot the "raw" scores.
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(
            f"Automated noisy channel detection: {ch_type}",
            fontsize=16,
            fontweight="bold",
        )
        sns.heatmap(
            data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0]
        )
        [
            ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
            for x in range(1, len(bins))
        ]
        ax[0].set_title("All Scores", fontweight="bold")

        # Now, adjust the color range to highlight segments that exceeded the
        # limit.
        sns.heatmap(
            data=data_to_plot,
            vmin=np.nanmin(limits),  # input data may contain NaNs
            cmap="Reds",
            cbar_kws=dict(label="Score"),
            ax=ax[1],
        )
        [
            ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
            for x in range(1, len(bins))
        ]
        ax[1].set_title("Scores > Limit", fontweight="bold")

        # The figure title should not overlap with the subplots.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figs.append(fig)
    assert figs

    return figs

def visualize_bad_breaks(raw, break_start_regex=None, break_end_regex=None, modify_break_regex_func=None):
    """Visualization of BAD_break, BAD_ending and BAD_beginning annotations + underlying events"""

    events = raw.annotations.to_data_frame(time_format=None)
    first_time = raw.first_time

    # If breaks were annotated using break start and end regex, extract potentially modified regex
    if break_start_regex and break_end_regex:
        if modify_break_regex_func:
            break_start_regex_new, break_end_regex_new = modify_break_regex_func(raw, break_start_regex, break_end_regex)
        else:
            break_start_regex_new = break_start_regex
            break_end_regex_new = break_end_regex

        # Extract break annotations together with the underlying break events
        annotations = raw.annotations[events[events['description'].astype(str).str.contains(
            f"BAD_break|{break_start_regex_new}|{break_end_regex_new}|BAD_beginning|BAD_ending",
            regex=True)].index]
    else: # In the case that breaks were identified using gaps, only use resulting break annotations
        annotations = raw.annotations[events[events['description'].astype(str).str.contains(
            f"BAD_break",
            regex=True)].index]

    
    fig, ax = plt.subplots(3, 1,figsize=(10, 7), height_ratios=[1,4,1.5])

    # Simple visualisation of break segments in the data (including beginning and end)
    for ann in annotations:
        if ann['description'] in ['BAD_break', 'BAD_beginning', 'BAD_ending']:
            # Draw rectangle for BAD segments
            rect = Rectangle((ann['onset'], 0), ann['duration'], 1, 
                        facecolor='red', alpha=0.2, label='Annotated breaks')
            ax[0].add_patch(rect)
        else:
            # Draw vertial line for underlying "break" events
            ax[0].vlines(ann['onset'], 0, 1, colors='black', label='Underyling break events')
    
    
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(raw.times[0]+first_time,raw.times[-1]+first_time) # Add first_time to align with annotation time
    ax[0].set_xlabel('Time (s)')
    ax[0].set_yticks([])
    ax[0].set_title('Break visualisation', loc='left', pad=12)

    
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # to avoid duplicate labels
    ax[0].legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        bbox_to_anchor=(1.,1),
        frameon=False
    )

    # Table with additional information (e.g. break onset, duration etc)
    df_breaks = events[events['description'].astype(str).str.contains(
        "BAD_break|BAD_beginning|BAD_ending",
        regex=True)]
    
    # Extract table data from DataFrame
    table_data = df_breaks.values  # Convert DataFrame to NumPy array
    table_columns = df_breaks.columns  # Get column names
    
    # Add a header row to the table data
    table_data_with_header = [list(table_columns)] + [list(row) for row in table_data]
    
    # Add table to the plot
    table = ax[1].table(
        cellText=table_data_with_header,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Additional information', loc='left', pad=12)

    if break_start_regex and break_end_regex:
        # Add the regular expressions that were used for the break annotation
        ax[2].text(
        0., 1,
        f"(Modified) break start regex:\n{break_start_regex_new} \n \n (Modified) break end regex:\n{break_end_regex_new}",
        fontsize=10,
        verticalalignment="top",
    )
        ax[2].set_axis_off()
        ax[2].set_title("Regular expressions used for break annotation", loc='left', pad=12)
    
    fig.tight_layout()
    
    return fig