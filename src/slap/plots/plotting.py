import pandas as pd
import matplotlib.pyplot as plt

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

