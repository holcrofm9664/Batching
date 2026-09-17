import pandas as pd
from typing import Any
from slap.models.comp_tests_assignments import permutation_assignments
from slap.comp_tests.helpers import (
    create_orders, 
    weights_and_volumes, 
    create_max_batches
)


def permutation_instance(
    pick_data:pd.DataFrame, 
    solution_allocation:pd.DataFrame, 
    num_orders:int, 
    min_order_size:int, 
    max_order_size:int
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

    solution_allocation = solution_allocation.dropna(subset=["tpnd", "aisle"])
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
        pick_data=pick_data,
        prod_subset=prod_subset,
        num_orders=num_orders,
        min_order_size=min_order_size,
        max_order_size=max_order_size,
        volume_dict=volume_dict,
        weight_dict=weight_dict
    )

    aisle_assignments = permutation_assignments(
        prods_by_dem=prods_by_dem,
        aisle_space_used=aisle_space_used,
        num_aisles=70,
        num_bays=122,
        slot_capacity=2
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
        "num_aisles":70,
        "num_bays":122
    }
    
    return instance