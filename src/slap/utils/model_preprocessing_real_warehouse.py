import pandas as pd
from slap.utils.helpers import (
    clean_dataframes, 
    weights_and_volumes, 
    create_orders, 
    create_max_batches
)

from slap.utils.dataclasses import (
    CageData,
    OrdersData,
    WarehouseData
)

def batching_preprocessing(
    pick_data:pd.DataFrame, 
    assignments:pd.DataFrame, 
    orders_data:OrdersData,  
    cage_data:CageData,
    time_limit:int,
    cages_per_batch:int=5, 
    buffer:float=0.25
) -> tuple[list[list[int]], dict[int,list[int]], int, dict[int,float], dict[int,float], int, int, int, int, float, int, int, int]:
    """Creates an instance for the batching model.

    Args:
        pick_data: Data frame containing pick data.
        assignments: Data frame containing assignments.
        max_orders_per_cage: Maximum-permitted orders assigned to each cage.
        num_samples: Number of orders sampled for the instance.

    Returns:
        An instance for the batching heuristic.
    """
    # clean pick data and solution allocation dataframes
    pick_data, assignments = clean_dataframes(
        pick_data=pick_data,
        assignments=assignments
    )

    all_products = assignments["tpnd"].unique()

    # get the product weights and volumes
    weights_dict, volumes_dict = weights_and_volumes(
        assignments=assignments,
        prod_subset=all_products
    )

    orders, product_demands_dict = create_orders(
        pick_data=pick_data,
        prod_subset=all_products,
        orders_data=orders_data
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
        num_bays=122
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
    }
    
    return instance
