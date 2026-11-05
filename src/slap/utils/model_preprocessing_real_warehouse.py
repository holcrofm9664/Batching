import pandas as pd
from typing import Any
from slap.utils.helpers import (
    weights_and_volumes, 
    sample_orders, 
    create_max_batches,
    get_prod_assignments
)

from slap.utils.dataclasses import (
    CageData,
    WarehouseData
)

def batching_preprocessing(
    pick_data:pd.DataFrame, 
    assignments:pd.DataFrame, 
    num_samples:int,  
    cage_data:CageData,
    time_limit:int,
    repetition:int,
    cages_per_batch:int=5, 
    buffer:float=0.25,
    **unused:Any
) -> tuple[list[list[int]], dict[int,list[int]], int, dict[int,float], dict[int,float], int, int, int, int, float, int, int, int]:

    all_products = assignments["tpnd"].unique()

    # get the product weights and volumes
    weights_dict, volumes_dict = weights_and_volumes(
        assignments=assignments,
        prod_subset=all_products
    )

    orders = sample_orders(
        pick_data=pick_data,
        num_samples=num_samples,
        seed=repetition
    )

    max_batches = create_max_batches(
        weights_dict=weights_dict,
        volumes_dict=volumes_dict,
        orders=orders,
        cage_data=cage_data,
        buffer=buffer,
        cages_per_batch=cages_per_batch
    )
    grouped = assignments.groupby("aisle",as_index=False)

    aisle_assignments_df = (
        grouped.agg(
            products_list = ("tpnd",list)
        )
    )

    aisle_assignments = dict(zip(aisle_assignments_df["aisle"],aisle_assignments_df["products_list"]))

    warehouse_data = WarehouseData(
        num_aisles=70,
        num_bays=60
    )

    prod_assignments = get_prod_assignments(
        assignments=assignments
    )

    instance = {
        "orders":orders,
        "aisle_assignments":aisle_assignments,
        "max_batches":max_batches,
        "weights_dict":weights_dict,
        "volumes_dict":volumes_dict,
        "warehouse_data":warehouse_data,
        "cage_data":cage_data,
        "time_limit":time_limit,
        "cages_per_batch":cages_per_batch,
        "prod_assignments":prod_assignments
    }
    
    return instance