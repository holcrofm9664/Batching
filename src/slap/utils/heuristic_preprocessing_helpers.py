import pandas as pd
import numpy as np
from ast import literal_eval
from slap.comp_tests.dataclasses import (
    DataFrames
)

def clean_dataframes(
    data_frames:DataFrames
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans the dataframes by dropping NA values and renaming columns.

    Args:
        data_frames: Data frames containing assignments and pick data.

    Returns:
        The cleaned data frames.
    """

    column_headers_dict = {
        "tpnd":"product",
        "Store":"store_id",
        "store":"store_id",
        "shop":"store_id",
        "Shop":"store_id",
        "assignment":"trip",
        "assignment_number":"trip",
        "batch":"trip",
        "Batch":"trip",
        "uod_number":"cage",
        "uod":"cage",
        "Cage":"cage",
        "datetime":"time",
        "qty":"qty_to_pick",
        "pick_qty":"qty_to_pick",
        "Aisle":"aisle"
    }
    # extract the dataframes, so we don't modify the dataclass
    pick_data, solution_allocation = data_frames.pick_data, data_frames.solution_allocation

    # ensure slot_pair is a tuple
    solution_allocation["Slot_Pair"] = (
        data_frames.solution_allocation["Slot_Pair"].
        apply(literal_eval)
    )

    # rename columns in both dataframes
    pick_data = (
        pick_data
        .rename(columns = column_headers_dict)
    )

    solution_allocation = (
        solution_allocation
        .rename(columns = column_headers_dict)
    )

    # drop NA values from the solution_allocation dataframe
    solution_allocation = solution_allocation.dropna()

    return pick_data, solution_allocation


def weights_and_volumes(
    solution_allocation:pd.DataFrame, 
    prod_subset:list[int]
) -> tuple[dict[int,float], dict[int,float]]:
    """Constructs a weights dictionary and a volumes dictionary from the dataframe.

    Args:
        solution_allocation: Dataframe containing the weights and volumes.
        prod_subset: Products remaining after we have filtered for time and a chosen 
            product subset.

    Returns:
        The dictionaries of weights and volumes.
    """

    # construct the weights and volumes dictionaries
    df = (
        solution_allocation[
            solution_allocation["tpnd"].isin(prod_subset)
        ]
        .drop_duplicates(subset="tpnd")
        .set_index("tpnd")
    )

    weight_dict = df["weight"].to_dict()
    volume_dict = df["volume"].to_dict()
    
    return weight_dict, volume_dict


def demands(
    pick_data:pd.DataFrame, 
    prod_subset:list[int], 
    stores_subset:list[int]
) -> tuple[dict[tuple[int,int],int], dict[int,int], dict[int,int]]:
    """Creates the shop_prod_dem dict.

    Args:
        pick_data: Data frame containing the pick data.
        prod_subset: The products in the warehouse.
        store_subset: The stores used in the instance.
    
    Returns:
        Demand for each (shop,product) pair.
    """

    # demand of each shop-prod combination
    shop_prod_series = (
        pick_data
        .groupby(["store_id", "product"])["qty_to_pick"]
        .sum()
        .clip(lower=0)
    )

    # add in missing combinations with zero demand
    shop_prod_index = pd.MultiIndex.from_product(
        [stores_subset, prod_subset],
        names=["store_id", "product"]
    )

    shop_prod_dem = (
        shop_prod_series
        .reindex(shop_prod_index, fill_value=0)
        .to_dict()
    )

    return shop_prod_dem


def sample_products_stores(
    num_prods:int, 
    num_stores:int, 
    data_frames:DataFrames
) -> list[int]:
    """Samples products and stores to be kept in the instance.

    Args:
        num_prods: Number of products to include in the instance.
        num_stores: Number of stores to include in the instance.
        data_frames: Data frames containg assignments and pick data.

    Returns:
        All the products, the sampled products and the sampled stores.
    """
    
    pick_data, solution_allocation  = data_frames.pick_data, data_frames.solution_allocation
    all_prods = solution_allocation["product"].unique()
    all_stores = pick_data["store_id"].unique()

    if num_prods >= len(all_prods):
        num_prods = len(all_prods)
    if num_stores >= len(all_stores):
        num_stores = len(all_stores)

    unit_qty_dict = (
        dict(
            pick_data
            .groupby("product")["qty_to_pick"]
            .sum()
        )
    )

    store_qty_dict = (
        dict(
            pick_data
            .groupby("store_id")["qty_to_pick"]
            .sum()
        )
    )

    items, frequencies = list(unit_qty_dict.keys()), list(unit_qty_dict.values())
    stores, store_frequencies = list(store_qty_dict.keys()), list(store_qty_dict.values())

    sampled_prods = np.random.choice(
        a = items,
        size = num_prods,
        replace = False,
        p = frequencies/np.sum(frequencies)
    )

    sampled_stores = np.random.choice(
        a = stores,
        size = num_stores,
        replace = False,
        p = store_frequencies/np.sum(store_frequencies)
    )

    all_prods = list(int(x) for x in all_prods)
    sampled_prods = list(int(x) for x in sampled_prods)
    sampled_stores = list(int(x) for x in sampled_stores)

    return all_prods, sampled_prods, sampled_stores
