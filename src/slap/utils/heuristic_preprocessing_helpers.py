import pandas as pd
import numpy as np
from ast import literal_eval

def clean_dataframes(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the dataframes by dropping NA values and renaming columns

    Inputs:
    - pick_data: the dataframe containing the pick data
    - solution_allocation: the dataframe containing storage assignments and product weights and volumes

    Outputs:
    - pick_data: the cleaned pick_data dataframe
    - solution_allocation: the cleaned solution_allocation dataframe
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

    # ensure slot_pair is a tuple
    solution_allocation["Slot_Pair"] = (
        solution_allocation["Slot_Pair"].
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
    """ 
    Constructs a weights dictionary and a volumes dictionary from the dataframe

    Inputs:
    - solution_allocation: the dataframe containing the weights and volumes
    - prod_subset: the products remaining after we have filtered for time and a chosen product subset

    Outputs:
    - volume_dict: the product volumes
    - weight_dict: the product weights 
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
    
    return volume_dict, weight_dict


def demands(
    pick_data:pd.DataFrame, 
    prod_subset:list[int], 
    stores_subset:list[int]
) -> tuple[dict[tuple[int,int],int], dict[int,int], dict[int,int]]:
    """ 
    Creates the shop_prod_dem dict

    Inputs:
    - pick_data: a dataframe containing the pick data
    - prod_subset: the remaining products we are optimising 
    - store_subset: the remaining stores we are optimising
    
    Outputs:
    - shop_prod_dem: the demand for each (shop,product) pair
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
    solution_allocation:pd.DataFrame, 
    pick_data:pd.DataFrame
) -> list[int]:
    """
    Samples products and stores to be kept in the instance

    Inputs:
    - num_prods: the number of products to include in the instance
    - num_stores: the number of stores to include in the instance
    - solution_allocation: the dataframe containing all products
    - pick_data: the pick data dataframe

    Outputs:
    - all_prods: a list of all products 
    - sampled_prods: the products we will use in our instance
    - sampled_stores: a stores we will use in our instance
    """
    
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
