import pandas as pd
from src.slap.utils.heuristic_preprocessing_helpers import (
    clean_dataframes, 
    sample_products_stores, 
    weights_and_volumes, 
    demands
)

def preprocessing_function_heuristic(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame, 
    num_aisles:int, 
    num_bays:int, 
    cage_weight_capacity:float, 
    cage_vol_capacity:float, 
    num_prods:int, 
    num_stores:int, 
    fill_percent:float=0.85, 
    cages_per_trip:int=5, 
    look_ahead:int=75
):

    pick_data, solution_allocation = clean_dataframes(
        pick_data=pick_data,
        solution_allocation=solution_allocation
    )
    
    _, prod_subset, stores_subset = sample_products_stores(
        num_prods=num_prods,
        num_stores=num_stores,
        solution_allocation=solution_allocation,
        pick_data=pick_data
    )
    
    weights_dict, vol_dict = weights_and_volumes(
        solution_allocation=solution_allocation,
        prod_subset=prod_subset
    )
    
    shop_prod_dem = demands(
        pick_data=pick_data,
        prod_subset=prod_subset,
        stores_subset=stores_subset
    )
    
    stored_product = (
        solution_allocation.dropna(subset=["Slot_Pair"])
        .groupby("Slot_Pair")["product"]
        .agg(lambda x: tuple(x.dropna().astype(int).unique()))
        .to_dict()
    )

    instance = {
        "shop_prod_dem":shop_prod_dem,
        "stored_product":stored_product,
        "weights_dict":weights_dict,
        "vol_dict":vol_dict,
        "num_aisles":num_aisles,
        "num_bays":num_bays,
        "cage_weight_capacity":cage_weight_capacity,
        "cage_vol_capacity":cage_vol_capacity,
        "fill_percent":fill_percent,
        "cages_per_trip":cages_per_trip,
        "look_ahead":look_ahead
    }
    
    return instance