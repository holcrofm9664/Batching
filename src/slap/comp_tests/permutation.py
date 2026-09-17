import pandas as pd
from typing import Any
from slap.models.comp_tests_assignments import permutation_assignments
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

def permutation_instance(
    data_frames:DataFrames, 
    orders_data:OrdersData,
    warehouse_data:WarehouseData
) -> dict[str,Any]:
    """
    Creates an instance for the batching model, where instances are a permutation of 
    true assignments with added random scattering of most-demanded products.

    Inputs:
    - pick_data: the pick data dataframe, containing demands
    - solution_allocation: the solution allocation dataframe, containing assignments, 
      weights and volumes
    - num_orders: the number of orders we want to use in the test
    - min_order_size: the minimum permitted size of orders
    - max_order_size: the maximum permitted size of orders

    Outputs:
    - instance: a kwargs instance ready for input into the batching model
    """

    solution_allocation = data_frames.solution_allocation.dropna(subset=["tpnd", "aisle"])
    current_assignments = dict(zip(solution_allocation["tpnd"], solution_allocation["aisle"]))

    # prod_subset is all_prods here
    prod_subset = [k for k in current_assignments.keys()]

    aisle_assignments = {a: [] for a in range(70)}
    for k in prod_subset:
        a = current_assignments[k]
        aisle_assignments[a].append(k)

    aisle_space_used = {a:len(s) for a,s in aisle_assignments.items()}
    
    weight_dict, volume_dict = weights_and_volumes(
        solution_allocation=solution_allocation,
        prod_subset=prod_subset
    )

    orders, prods_by_dem = create_orders(
        pick_data=data_frames.pick_data,
        prod_subset=prod_subset,
        orders_data=orders_data,
        volume_dict=volume_dict,
        weight_dict=weight_dict
    )

    aisle_assignments = permutation_assignments(
        prods_by_dem=prods_by_dem,
        aisle_space_used=aisle_space_used,
        warehouse_data=warehouse_data
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