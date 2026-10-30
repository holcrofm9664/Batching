import pandas as pd
from typing import Any
from slap.utils.dataclasses import CageData
from slap.utils.helpers import sample_orders, weights_and_volumes, get_prod_assignments

def heuristic_preprocessing(
    pick_data:pd.DataFrame,
    assignments:pd.DataFrame, 
    max_orders_per_cage:int,
    num_samples:int,
    cages_per_batch:int,
    seed:int    
) -> dict[str,Any]:
    """Creates an instance for the batching heuristic.

    Args:
        pick_data: Data frame containing pick data.
        assignments: Data frame containing assignments.
        max_orders_per_cage: Maximum-permitted orders assigned to each cage.
        num_samples: Number of orders sampled for the instance.
        cages_per_batch: Number of cages per batch.
        seed: Seed for controlling randomness in orders generation.

    Returns:
        An instance for the batching heuristic.
    """

    orders = sample_orders(
        pick_data = pick_data,
        num_samples=num_samples,
        seed=seed
    )

    weights_dict, volumes_dict = weights_and_volumes(
        assignments=assignments,
        prod_subset=assignments["tpnd"].drop_duplicates().to_list()
    )

    cage_data = CageData(
        max_orders_per_cage=max_orders_per_cage
    )

    prod_assignments = get_prod_assignments(
        assignments=assignments
    )

    instance = {
        "orders":orders,
        "weights_dict":weights_dict,
        "volumes_dict":volumes_dict,
        "cage_data":cage_data,
        "prod_assignments":prod_assignments,
        "cages_per_batch":cages_per_batch
    }

    return instance