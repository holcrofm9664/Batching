import random
import pandas as pd
from typing import Any
from slap.models.comp_tests_assignments import aisle_directionality_assignments
from slap.comp_tests.helpers import (
    create_orders, 
    weights_and_volumes, 
    create_max_batches
)

def aisle_directionality_instance(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame, 
    num_aisles:int, 
    num_bays:int, 
    slot_capacity:int, 
    num_orders:int, 
    min_order_size:int, 
    max_order_size:int, 
    same_direction:bool
) -> dict[str,Any]:
    """
    Creates an instance for the aisle directionality computational tests

    Inputs:
    - pick_data: the pick data dataframe, containing product demands
    - solution_allocation: the solution allocation dataframe, containing assignments, 
      product weights and volumes
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in the warehouse
    - slot_capacity: the number of unique products that can fit into each (aisle,bay) pair
    - num_orders: the number of orders used for testing
    - min_order_size: the minimum allowed size of generated orders
    - max_order_size: the maximum allowed size of generated orders
    - same_direction: if products are to be scattered across aisles wuth the same 
      direction. If set of False, they will be placed only in aisles with a different 
      direction

    Outputs:
    - instance: a kwargs instance ready for input into the batching model
    """

    all_prods = solution_allocation.dropna()["tpnd"].to_list()

    num_prods = (num_aisles*num_bays*slot_capacity)//2
    prod_subset = random.sample(all_prods, num_prods)

    weight_dict, volume_dict = weights_and_volumes(
        solution_allocation=solution_allocation,
        prod_subset=prod_subset
    )
    
    orders, _ = create_orders(
        pick_data=pick_data,
        prod_subset = prod_subset,
        num_orders=num_orders,
        min_order_size=min_order_size,
        max_order_size=max_order_size,
        volume_dict=volume_dict,
        weight_dict=weight_dict
    )
    
    aisle_assignments = aisle_directionality_assignments(
        prods=prod_subset,
        num_aisles=num_aisles,
        num_bays=num_bays,
        slot_capacity=slot_capacity,
        same_direction=same_direction
    )

    max_batches = create_max_batches(
        weights_dict=weight_dict,
        volumes_dict=volume_dict,
        orders=orders
    )

    instance = {
        "orders":orders,
        "aisle_assignments":aisle_assignments,
        "max_batches":max_batches,
        "weights_dict":weight_dict,
        "volumes_dict":volume_dict,
        "num_aisles":num_aisles,
        "num_bays":num_bays
    }
    
    return instance