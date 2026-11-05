import pandas as pd
import numpy as np
from ast import literal_eval

def postprocessing(
        results:pd.DataFrame
) -> pd.DataFrame:
    """Postprocesses the results dataframe.
    
    Args:
        results: Results data frame testing scalability.

    Returns:
        A post-processed data frame ready for plotting and analysis.
    """
    df = results.copy()

    # put columns into a convenient order for analysis
    df=df[
        [
            "num_orders",
            "max_orders_per_cage",
            "repetition",
            "m_status",
            "ms_status",
            "m_optgap",
            "ms_optgap",
            "m_runtime",
            "ms_runtime",
            "m_distance",
            "ms_distance",
            "h_distance",
            "m_aisles",
            "ms_aisles",
            "h_aisles",
            "orders",
            "m_batches",
            "ms_batches",
            "h_batches"
        ]
    ]

    # total ordered units
    df["total_ordered_units"] = df["orders"].apply(
        lambda x:sum(
            len(order) 
            for order in literal_eval(x).values()
        )
    )
    
    # units in each order
    df["units_per_order"] = df["orders"].apply(
        lambda x:[
            len(order) 
            for order in literal_eval(x).values()
        ]
    )

    # total unique products across orders
    df["total_unique_prods"] = df["orders"].apply(
        lambda x:len(
            set(
                prod 
                for order in [
                    order 
                    for order in literal_eval(x).values()
                ] 
                for prod in order
            )
        )
    )

    # unique products in each order
    df["unique_prods_per_order"] = df["orders"].apply(
        lambda x:[
            len(set(order)) 
            for order in literal_eval(x).values()
        ]
    )

    df["m_dreduction"] = df.apply(
        lambda row:100*(1-row["m_distance"]/row["h_distance"]),
        axis=1
    )

    df["ms_dreduction"] = df.apply(
        lambda row:100*(1-row["ms_distance"]/row["h_distance"]),
        axis=1
    )

    # number of batches used by the model
    df["m_numbatches"] = df["m_batches"].apply(
        lambda x:len(
            [
                (batch) 
                for batch in literal_eval(x).values() 
                if len(
                    [
                        prod 
                        for cage in batch.values()
                        for prod in cage
                    ]
                ) > 0
            ]
        )
    )

    # number of batches used by the model under single-placement
    df["ms_numbatches"] = df["ms_batches"].apply(
        lambda x:len(
            [
                (batch) 
                for batch in literal_eval(x).values() 
                if len(
                    [
                        prod 
                        for cage in batch.values()
                        for prod in cage
                    ]
                ) > 0
            ]
        )
    )

    # number of batches used by the heuristic
    df["h_numbatches"] = df["h_batches"].apply(
        lambda x:len(
            [
                (batch) 
                for batch in literal_eval(x).values()
                if len(
                    [
                        prod
                        for cage in batch.values()
                        for prod in cage
                    ]
                ) > 0
            ]
        )
    )

    df["m_logruntime"] = df["m_runtime"].apply(
        lambda x:np.log(x)
    )

    df["ms_logruntime"] = df["ms_runtime"].apply(
        lambda x:np.log(x)
    )

    df.to_csv("/home/holcrofm/holcrofm/Batching/data/tesco_data/processed_df.csv")
    return df
