from typing import Any
from slap.utils.dataclasses import WarehouseData

def calculate_distance_one_batch(
    aisle_assignments: dict[int, list[int]],
    batch: dict[Any, list[int]],
    warehouse_data: WarehouseData
) -> float:
    """
    Calculate the minimum picking distance for one batch.

    Uses dynamic programming over occupied aisle subsets,
    avoiding enumeration of all product-to-aisle assignments.
    """

    # Identify the distinct products in the batch
    products = {
        prod
        for cage in batch.values()
        for prod in cage
    }

    if not products:
        return 0.0

    # Preserve the original aisle ordering
    aisles = list(aisle_assignments)

    # Map each product to a bitmask of its possible aisles
    product_masks = {prod: 0 for prod in products}

    for i, aisle in enumerate(aisles):
        bit = 1 << i

        for prod in aisle_assignments[aisle]:
            if prod in product_masks:
                product_masks[prod] |= bit

    # Check that every product has at least one location
    missing = [
        prod
        for prod, mask in product_masks.items()
        if mask == 0
    ]

    if missing:
        raise ValueError(
            f"Products without aisle assignments: {missing}"
        )

    # Dynamic programming:
    # Each state represents a feasible set of occupied aisles
    reachable = {0}

    for options in product_masks.values():

        new_reachable = set()

        for mask in reachable:

            remaining = options

            while remaining:
                bit = remaining & -remaining

                new_reachable.add(mask | bit)

                remaining ^= bit

        reachable = new_reachable

    # Warehouse distances
    M = warehouse_data.between_aisle_dist
    N = warehouse_data.between_bay_dist

    L = (warehouse_data.num_bays + 1) * N

    min_dist = float("inf")

    # Evaluate each distinct occupied-aisle set
    for mask in reachable:

        visited = [
            aisle
            for i, aisle in enumerate(aisles)
            if mask & (1 << i)
        ]

        if not visited:
            continue

        num_occupied_aisles = len(visited)

        # Consecutive aisle direction mismatches
        mismatch = sum(
            (abs(a - b) + 1) % 2
            for a, b in zip(visited, visited[1:])
        )

        F_down = min(visited) % 2
        L_up = (max(visited) + 1) % 2

        H = max(visited) * M

        distance = (
            L * (
                num_occupied_aisles
                + mismatch
                + F_down
                + L_up
            )
            + 2 * max(M, H)
        )

        min_dist = min(min_dist, distance)

    return min_dist
    

def calculate_distance_all_batches(
    batches:dict[Any,dict[Any,list[int]]], 
    aisle_assignments:dict[int,list[int]], 
    warehouse_data:WarehouseData
) -> tuple[int, dict[Any,int]]:
    """Calculates the distance for all batches.

    Args:
        batches: Set of batches returned by the optimisation.
        aisle_assignments: Set of aisle assignments, including scattered storage.
        warehouse_data: Warehouse-related data.

    Returns:
        The total distance and distance by batch.
    """
    
    dist_by_batch = {}

    for i, batch in batches.items():
        distance = calculate_distance_one_batch(aisle_assignments=aisle_assignments,
                                               batch=batch,
                                               warehouse_data=warehouse_data)
        dist_by_batch[i] = distance

    total_distance = sum(dist_by_batch.values())

    return total_distance, dist_by_batch