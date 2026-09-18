import random
from typing import Any
from slap.models.comp_tests_assignments import scalability_assignments
from slap.comp_tests.helpers import (
    create_orders, 
    weights_and_volumes, 
    create_max_batches
)
from slap.comp_tests.dataclasses import (
    DataFrames,
    WarehouseData,
    OrdersData
)


def scalability_instance(
    data_frames:DataFrames, 
    warehouse_data:WarehouseData, 
    orders_data:OrdersData, 
    top_frac:float
    ) -> dict[str,Any]:
    """Creates an instance for the scalability computational tests.

    Args:
        data_frames: Data frames containing assignments and pick data.
        warehouse_data: Warehouse-related data.
        orders_data: Orders-related data.
        top_frac: Fraction of products to be scattered.

    Returns:
        A kwargs instance ready for input into the batching model
    """

    all_prods = data_frames.solution_allocation.dropna()["tpnd"].to_list()

    if (warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity)//(1+top_frac) > len(all_prods):
        print(
            f"The requested number of {(warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity)//(1+top_frac)}"
            f"exceeds the available number of products."
            f"Using {len(all_prods)} products instead."
        )
        prod_subset = all_prods
    else:
        num_prods = int((warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity)//(1+top_frac))
        prod_subset = random.sample(all_prods, num_prods)


    weight_dict, volume_dict = weights_and_volumes(
        solution_allocation=data_frames.solution_allocation,
        prod_subset=prod_subset
    )
    
    orders, prods_by_dem = create_orders(
        pick_data=data_frames.pick_data,
        prod_subset = prod_subset,
        orders_data=orders_data,
        volume_dict=volume_dict,
        weight_dict=weight_dict
    )

    
    aisle_assignments = scalability_assignments(
        prods_by_dem=prods_by_dem,
        warehouse_data=warehouse_data,
        top_frac=top_frac
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