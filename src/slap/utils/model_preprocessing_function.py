import pandas as pd
from slap.utils.model_preprocessing_helpers import (
    clean_dataframes, 
    sample_products, 
    weights_and_volumes, 
    create_orders, 
    create_max_batches, 
    create_aisle_assignments
)

from slap.utils.dataclasses import (
    WarehouseData,
    CageData,
    OrdersData
)

def preprocessing_function_batching(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame, 
    num_products:int, 
    orders_data:OrdersData, 
    warehouse_data:WarehouseData,  
    cage_data:CageData,
    cages_per_batch:int=5, 
    buffer:float=0.25
) -> tuple[list[list[int]], dict[int,list[int]], int, dict[int,float], dict[int,float], int, int, int, int, float, int, int, int]:

    # clean pick data and solution allocation dataframes
    pick_data, solution_allocation = clean_dataframes(
        pick_data=pick_data,
        solution_allocation=solution_allocation
    )

    # take a sample of products
    _, prod_subset = sample_products(
        num_prods=num_products,
        solution_allocation=solution_allocation,
        pick_data=pick_data
    )

    # get the product weights and volumes
    weights_dict, volumes_dict = weights_and_volumes(
        solution_allocation=solution_allocation,
        prod_subset=prod_subset
    )

    orders, product_demands_dict = create_orders(
        pick_data=pick_data,
        prod_subset=prod_subset,
        orders_data=orders_data,
        volume_dict=volumes_dict,
        weight_dict=weights_dict,
        cage_data=cage_data
    )

    max_batches = create_max_batches(
        weights_dict=weights_dict,
        volumes_dict=volumes_dict,
        orders=orders,
        cage_data=cage_data,
        buffer=buffer,
        cages_per_batch=cages_per_batch
    )
    
    aisle_assignments = create_aisle_assignments(
        solution_allocation=solution_allocation,
        warehouse_data=warehouse_data,
        prod_subset=prod_subset,
        num_zones=warehouse_data.num_zones,
        product_demands_dict=product_demands_dict,
    )

    instance = {
        "orders":orders,
        "aisle_assignments":aisle_assignments,
        "max_batches":max_batches,
        "weights_dict":weights_dict,
        "volumes_dict":volumes_dict,
        "warehouse_data":warehouse_data,
        "cage_data":cage_data,
        "cages_per_batch":cages_per_batch,
    }
    
    return instance

    
