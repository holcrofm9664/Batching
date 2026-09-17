import pandas as pd
from slap.utils.model_preprocessing_helpers import (
    clean_dataframes, 
    sample_products, 
    weights_and_volumes, 
    create_orders, 
    create_max_batches, 
    create_aisle_assignments
)

def preprocessing_function_batching(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame, 
    num_products:int, 
    num_orders:int, 
    min_order_size:int, 
    max_order_size:int, 
    num_aisles:int, 
    num_bays:int, 
    num_zones:int=3, 
    slot_capacity:int=2, 
    between_aisle_dist:int=1, 
    between_bay_dist:int=1, 
    cages_per_batch:int=5, 
    cage_weight_capacity:int=400, 
    cage_volume_capacity:int=45, 
    fill_percent:float=0.85, 
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
        num_orders=num_orders,
        min_order_size=min_order_size,
        max_order_size=max_order_size,
        volume_dict=volumes_dict,
        weight_dict=weights_dict,
        cage_weight_capacity=cage_weight_capacity,
        cage_volume_capacity=cage_volume_capacity,
        fill_percent=fill_percent
    )

    max_batches = create_max_batches(
        weights_dict=weights_dict,
        volumes_dict=volumes_dict,
        orders=orders,
        cage_weight_capacity=cage_weight_capacity,
        cage_volume_capacity=cage_volume_capacity,
        fill_percent=fill_percent,
        buffer=buffer,
        cages_per_batch=cages_per_batch
    )
    
    aisle_assignments = create_aisle_assignments(
        prod_subset=prod_subset,
        num_zones=num_zones,
        num_aisles=num_aisles,
        num_bays=num_bays,
        solution_allocation=solution_allocation,
        product_demands_dict=product_demands_dict,
        slot_capacity=slot_capacity
    )

    instance = {
        "orders":orders,
        "aisle_assignments":aisle_assignments,
        "max_batches":max_batches,
        "weights_dict":weights_dict,
        "volumes_dict":volumes_dict,
        "num_aisles":num_aisles,
        "num_bays":num_bays,
        "cage_weight_capacity":cage_weight_capacity,
        "cage_volume_capacity":cage_volume_capacity,
        "fill_percent":fill_percent,
        "cages_per_batch":cages_per_batch,
        "between_aisle_dist":between_aisle_dist,
        "between_bay_dist":between_bay_dist
    }
    
    return instance

    
