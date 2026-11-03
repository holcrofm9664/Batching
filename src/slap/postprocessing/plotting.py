import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def generate_plots(
        results:pd.DataFrame
) -> dict[str,go.figure]:
    """Produces a dictionary of figures from the results.

    Args:
        results: Results data frame.

    Returns:
        A dictionary of figures.
    """

    figures = {}

    # preprocessing
    num_success = results[
        results["status"]==2
    ].shape[0]

    num_timeout = results[
        results["status"]==9
    ].shape[0]

    num_fail = results[
        (results["status"] != 2) & (results["status"] != 9)
    ].shape[0]

    df_ran = results[
        (results["status"] != 2) | (results["status"] != 9)
    ]

    # figure 1: runtime histogram
    figures["runtime_hist"] = px.histogram(
        data_frame=df_ran,
        x="runtime",
        nbins=20,
        labels={
            "runtime":"Runtime"
        }
    )

    # figure 2: runtime vs number of orders
    figures["runtime_num_orders_box"] = px.box(
        data_frame=df_ran,
        x="num_orders",
        y="runtime",
        labels={
            "num_orders":"Number of Orders",
            "runtime":"Runtime"
        }
    )

    # figure 3: runtime vs order size
    figures["runtime_size_orders_box"] = px.box(
        data_frame=df_ran,
        x="num_orders",
        y="runtime",
        labels={
            "num_orders":"Number of Orders",
            "runtime":"Runtime"
        }
    )

    # figure 4: runtime vs (order size) x (order size)
    figures["order_size_num_scatter"] = px.scatter(
        data_frame=df_ran,
        x="(order_size x num_orders)",
        y="runtime",
        labels={
            "(order_size x num_orders)":"(Order Size) x (Number of Orders)",
            "runtime":"Runtime"
        }
    )

    # figure 5: runtime vs number of trips
    figures["num_trips_box"] = px.box(
        data_frame=df_ran,
        x="num_trips",
        y="runtime",
        labels={
            "num_trips":"Number of Trips",
            "runtime":"Runtime"
        }
    )

    # return dictionary
    return figures