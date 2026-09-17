import random
import pandas as pd
from typing import Any
from slap.models.comp_tests_assignments import aisle_directionality_assignments
from slap.comp_tests.helpers import (
    create_orders, 
    weights_and_volumes, 
    create_max_batches
)
from dataclasses import dataclass

@dataclass
class DataFrames:
    pick_data:pd.DataFrame
    solution_allocation:pd.DataFrame

@dataclass
class WarehouseData:
    """
    warehouse-related input data.

    attributes:
        num_aisles: the number of aisles in the warehouse
        num_bays: the number of bays in the warehouse
        slot_capacity: the number of unique products that can fit into each (aisle,bay) pair
        between_aisle_dist: the distance between consecutive aisles
        between_bay_dist: the distance between consecutive bays
    """
    num_aisles:int
    num_bays:int
    slot_capacity:int
    between_aisle_dist:int
    between_bay_dist:int

@dataclass
class OrdersData:
    num_orders:int
    min_order_size:int
    max_order_size:int


def aisle_directionality_instance(
    data_frames:DataFrames, 
    warehouse_data:WarehouseData, 
    orders_data:OrdersData, 
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

    all_prods = data_frames.solution_allocation.dropna()["tpnd"].to_list()

    num_prods = (warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity)//2
    prod_subset = random.sample(all_prods, num_prods)

    weight_dict, volume_dict = weights_and_volumes(
        solution_allocation=data_frames.solution_allocation,
        prod_subset=prod_subset
    )
    
    orders, _ = create_orders(
        pick_data=data_frames.pick_data,
        prod_subset = prod_subset,
        orders_data=orders_data,
        volume_dict=volume_dict,
        weight_dict=weight_dict
    )
    
    aisle_assignments = aisle_directionality_assignments(
        prods=prod_subset,
        warehouse_data=warehouse_data,
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
        "num_aisles":warehouse_data.num_aisles,
        "num_bays":warehouse_data.num_bays
    }
    
    return instance