import pandas as pd
from typing import Any
from slap.models.comp_tests_assignments import permutation_assignments
from slap.utils.helpers import (
    create_orders, 
    weights_and_volumes, 
    create_max_batches
)
from slap.utils.dataclasses import (
    WarehouseData,
    OrdersData,
    CageData
)

def permutation_instance(
    pick_data:pd.DataFrame,
    solution_allocation:pd.DataFrame, 
    orders_data:OrdersData,
    warehouse_data:WarehouseData,
    cage_data:CageData
) -> dict[str,Any]:
    """Creates a permutation of true assignments with random scattering of 
    most-demanded products.

    Args:
        pick_data: Dataframe containing orders data.
        solution_allocation: Dataframe containing storage assignments.
        num_orders: Number of orders used in the tests.
        orders_data: Orders-related data.
        warehouse_data: Warehouse-related data.

    Returns:
        A kwargs instance ready for input into the batching model.
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
        orders=orders,
        cage_data=cage_data
    )

    instance = {
        "orders":orders,
        "aisle_assignments":aisle_assignments,
        "max_batches":max_batches,
        "weights_dict":weight_dict,
        "volumes_dict":volume_dict,
        "warehouse_data":warehouse_data,
        "cage_data":cage_data
    }
    
    return instance