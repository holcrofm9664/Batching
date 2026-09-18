import random   
from typing import Any
from slap.models.comp_tests_assignments import scatter_frequency_assignments
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

def scatter_frequency_instance(
    data_frames:DataFrames,
    warehouse_data:WarehouseData,
    orders_data:OrdersData,
    scatter_frequency:int
) -> dict[str,Any]:
    """Creates an instance for the scatter frequency computational tests.

    Args:
        data_frames: Data frames containing assignment and pick data.
        warehouse_data: Warehouse-related data.
        orders_data: Orders-related data.
        scatter_frequency: Number of unique aisles each product is assigned to.

    Returns:
        A kwargs instance ready for input into the batching model.
    """

    all_prods = data_frames.solution_allocation.dropna()["tpnd"].to_list()

    num_prods = (warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity)//scatter_frequency
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

    aisle_assignments = scatter_frequency_assignments(
        scatter_frequency=scatter_frequency,
        prods=prod_subset,
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