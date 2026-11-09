import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from collections import Counter
import matplotlib.colors as mcolors

def plot_num_orders_vs_runtime(
        results:pd.DataFrame
):
    # Calculate median and quartiles for each instance size
    df_plot = (
        results.groupby("num_orders")["m_logruntime"]
        .agg(
            median="median",
            lq=lambda x: x.quantile(0.25),
            uq=lambda x: x.quantile(0.75)
        )
        .reset_index()
        .sort_values("num_orders")
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot median
    ax.plot(
        df_plot["num_orders"],
        df_plot["median"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Median"
    )

    # Shade interquartile range
    ax.fill_between(
        df_plot["num_orders"],
        df_plot["lq"],
        df_plot["uq"],
        color="steelblue",
        alpha=0.2,
        label="Interquartile range"
    )

    # Formatting
    ax.set_xlabel("Number of orders", fontsize=12)
    ax.set_ylabel("Log runtime", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        "src/slap/plots/figures/num_orders_runtime.pdf",
        bbox_inches="tight"
    )

def plot_num_units_distance_reductions(
        results:pd.DataFrame
) -> pd.DataFrame:
    
    # Copy the data to avoid modifying the original dataframe
    m = results["m_dreduction"].copy()
    ms = results["ms_dreduction"].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # M: Larger filled blue circles
    ax.scatter(
        results["total_ordered_units"],
        m,
        color="steelblue",
        s=80,
        marker="o",
        label="M",
        zorder=3
    )

    # MS: Smaller filled orange circles
    ax.scatter(
        results["total_ordered_units"],
        ms,
        color="darkorange",
        s=35,
        marker="o",
        label="MS",
        zorder=4
    )

    # Formatting
    ax.set_xlabel("Total ordered units", fontsize=12)
    ax.set_ylabel("Distance reduction (%)", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Save for publication
    fig.savefig(
        "src/slap/plots/figures/distance_reduction_scatter.pdf",
        bbox_inches="tight"
    )

def plot_num_orders_vs_optgap(
        results:pd.DataFrame
):
    # Calculate median and quartiles for each instance size
    df_plot = (
        results.groupby("num_orders")["m_optgap"]
        .agg(
            median="median",
            lq=lambda x: x.quantile(0.25),
            uq=lambda x: x.quantile(0.75)
        )
        .reset_index()
        .sort_values("num_orders")
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot median
    ax.plot(
        df_plot["num_orders"],
        df_plot["median"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Median"
    )

    # Shade interquartile range
    ax.fill_between(
        df_plot["num_orders"],
        df_plot["lq"],
        df_plot["uq"],
        color="steelblue",
        alpha=0.2,
        label="Interquartile range"
    )

    # Formatting
    ax.set_xlabel("Number of orders", fontsize=12)
    ax.set_ylabel("Optimality Gap", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        "src/slap/plots/figures/num_orders_optgap.pdf",
        bbox_inches="tight"
    )


def heatmap_df(
        results:pd.DataFrame
) -> pd.DataFrame:
    
    # Extract all aisle visits
    m_all_aisles = [
        aisle
        for x in results["m_aisles"]
        for aisles in literal_eval(x).values()
        for aisle in aisles
    ]

    ms_all_aisles = [
        aisle
        for x in results["ms_aisles"]
        for aisles in literal_eval(x).values()
        for aisle in aisles
    ]

    h_all_aisles = [
        aisle
        for x in results["h_aisles"]
        for aisles in literal_eval(x).values()
        for aisle in aisles
    ]

    # Count visits to each aisle
    m_counts = Counter(m_all_aisles)
    ms_counts = Counter(ms_all_aisles)
    h_counts = Counter(h_all_aisles)

    # Create dataframe
    df = pd.DataFrame({
        "aisle": range(70),
        "m_visits": [m_counts[i] for i in range(70)],
        "ms_visits": [ms_counts[i] for i in range(70)],
        "h_visits": [h_counts[i] for i in range(70)],
        "lenth": 1
    })

    return df


def plot_aisle_visits_heatmap(
        df:pd.DataFrame
):
    
    # Extract visit counts
    data = df[
        ["m_visits", "ms_visits", "h_visits"]
    ].to_numpy().T

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 2.5))

    # Shared colour scale
    norm = mcolors.Normalize(
        vmin=0,
        vmax=data.max()
    )

    # Plot heatmap
    im = ax.imshow(
        data,
        cmap="cividis",
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

    # Formatting
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Model", "Model (S)", "Heuristic"])

    ax.set_xticks(range(0, 70, 5))
    ax.set_xlabel("Aisle")

    ax.axhline(y=0.5, color="black", linewidth=1)
    ax.axhline(y=1.5, color="black", linewidth=1)

    # Colour bar
    fig.colorbar(
        im,
        ax=ax,
        label="Number of visits",
        pad=0.02
    )

    fig.savefig(
        "src/slap/plots/figures/aisle_visits_heatmap.pdf",
        bbox_inches="tight"
    )

    return 

def plot_num_batches_runtime(
        results:pd.DataFrame
):
    # Calculate median and quartiles for each instance size
    df_plot = (
        results.groupby("m_numbatches")["m_runtime"]
        .agg(
            median="median",
            lq=lambda x: x.quantile(0.25),
            uq=lambda x: x.quantile(0.75)
        )
        .reset_index()
        .sort_values("m_numbatches")
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot median
    ax.plot(
        df_plot["m_numbatches"],
        df_plot["median"],
        color="steelblue",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Median"
    )

    # Shade interquartile range
    ax.fill_between(
        df_plot["m_numbatches"],
        df_plot["lq"],
        df_plot["uq"],
        color="steelblue",
        alpha=0.2,
        label="Interquartile range"
    )

    # Formatting
    ax.set_xlabel("Number of Batches", fontsize=12)
    ax.set_ylabel("Runtime", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    fig.savefig(
        "src/slap/plots/figures/num_batches_runtime.pdf",
        bbox_inches="tight"
    )


def plot_num_units_runtime(
        results:pd.DataFrame
) -> pd.DataFrame:
    
    # Copy the data to avoid modifying the original dataframe
    m = results["m_runtime"].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # M: Larger filled blue circles
    ax.scatter(
        results["total_ordered_units"],
        m,
        color="steelblue",
        s=80,
        marker="o",
        label="M",
        zorder=3
    )

    # Formatting
    ax.set_xlabel("Total ordered units", fontsize=12)
    ax.set_ylabel("Distance reduction (%)", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Save for publication
    fig.savefig(
        "src/slap/plots/figures/num_units_runtime.pdf",
        bbox_inches="tight"
    )


def plot_num_unique_products_runtime(
        results:pd.DataFrame
) -> pd.DataFrame:
    
    # Copy the data to avoid modifying the original dataframe
    m = results["m_runtime"].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # M: Larger filled blue circles
    ax.scatter(
        results["total_unique_prods"],
        m,
        color="steelblue",
        s=80,
        marker="o",
        label="M",
        zorder=3
    )

    # Formatting
    ax.set_xlabel("Total ordered units", fontsize=12)
    ax.set_ylabel("Distance reduction (%)", fontsize=12)

    ax.tick_params(axis="both", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    # Save for publication
    fig.savefig(
        "src/slap/plots/figures/num_unique_products_runtime.pdf",
        bbox_inches="tight"
    )