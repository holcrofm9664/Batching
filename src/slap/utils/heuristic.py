from collections import deque, Counter
from itertools import product
import math
import time
from typing import Any

from slap.utils.dataclasses import CageData, WarehouseData
from slap.eval.evaluation import calculate_distance_all_batches

warehouse_data = WarehouseData(
    num_aisles=70,
    num_bays=60
)

def batching_heuristic(
        orders: dict[int, list[int]],
        weights_dict: dict[int, float],
        volumes_dict: dict[int, float],
        cage_data: CageData,
        prod_assignments: dict[int, tuple[int, int]],
        aisle_assignments:dict[int,list[int]],
        cages_per_batch: int = 5,
        verbose: bool = False,
        **unused: Any
):

    start_time = time.perf_counter()

    def debug(message):
        if verbose:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{elapsed:.3f}s] {message}",
                flush=True
            )

    debug("Starting batching heuristic.")

    # --------------------------------------------------
    # 1. Initialise capacities
    # --------------------------------------------------

    cage_weight_capacity = (
        cage_data.cage_weight_capacity * cage_data.fill_frac
    )

    cage_volume_capacity = (
        cage_data.cage_volume_capacity * cage_data.fill_frac
    )

    if cages_per_batch <= 0:
        raise ValueError("cages_per_batch must be positive.")

    if cage_data.max_orders_per_cage < 1:
        raise ValueError("max_orders_per_cage must be positive.")

    debug(
        f"Cage capacities: "
        f"weight={cage_weight_capacity:.2f}, "
        f"volume={cage_volume_capacity:.2f}"
    )

    # --------------------------------------------------
    # 2. Construct storage location indices
    # --------------------------------------------------

    sol_prods = set(prod_assignments.keys())

    aisles = list(range(70))
    bays = list(range(1, 61))

    slot_idx = {}

    for aisle, bay in product(aisles, bays):

        slot_idx[(aisle, bay)] = (
            60 * aisle + bay
            if aisle % 2 == 0
            else 60 * aisle + (61 - bay)
        )

    prod_idx = {
        k: slot_idx[prod_assignments[k]]
        for k in sol_prods
    }

    debug("Storage location indices constructed.")

    # --------------------------------------------------
    # 3. Create product-order pairs
    # --------------------------------------------------

    # Each occurrence is represented separately.
    # For example, if order 1 contains [5, 5, 8],
    # we create [(5, 1), (5, 1), (8, 1)].

    prods_in_orders = [
        (prod, order_id)
        for order_id, products in orders.items()
        for prod in products
    ]

    num_prods = len(prods_in_orders)

    debug(
        f"Number of orders: {len(orders)}"
    )

    debug(
        f"Number of product occurrences: {num_prods}"
    )

    debug(
        f"Number of distinct products: "
        f"{len(set(prod for prod, _ in prods_in_orders))}"
    )

    # --------------------------------------------------
    # 4. Sort products by storage location
    # --------------------------------------------------

    prods_sorted = deque(
        sorted(
            prods_in_orders,
            key=lambda x: prod_idx[x[0]]
        )
    )

    debug("Products sorted by storage location.")

    # --------------------------------------------------
    # 5. Initialise cages
    # --------------------------------------------------

    cage_idx = 1

    cages = {cage_idx: []}
    cage_orders = {cage_idx: []}

    cage_weights = {cage_idx: 0}
    cage_volumes = {cage_idx: 0}

    # NEW: Track product-order occurrences in each cage.
    cage_contents = {cage_idx: []}

    # Used only to validate the solution.
    assigned_occurrences = Counter()

    num_placed = 0

    debug("Entering main placement loop.")

    # --------------------------------------------------
    # 6. Main placement loop
    # --------------------------------------------------

    while prods_sorted:

        # Each iteration handles exactly one occurrence.
        prod, order = prods_sorted.popleft()

        placed = False

        prod_weight = weights_dict[prod]
        prod_volume = volumes_dict[prod]

        # A product that exceeds cage capacity
        # cannot be assigned to any feasible cage.

        if (
            prod_weight > cage_weight_capacity
            or prod_volume > cage_volume_capacity
        ):
            raise ValueError(
                f"Product {prod} from order {order} "
                f"exceeds cage capacity."
            )

        # Search existing cages.

        for cage in cages:

            weight_feasible = (
                cage_weights[cage] + prod_weight
                <= cage_weight_capacity
            )

            volume_feasible = (
                cage_volumes[cage] + prod_volume
                <= cage_volume_capacity
            )

            if not (weight_feasible and volume_feasible):
                continue

            # Case 1: Order already present in cage.

            if order in cage_orders[cage]:

                cages[cage].append(prod)

                # NEW: Record originating order.
                cage_contents[cage].append((prod, order))

                cage_weights[cage] += prod_weight
                cage_volumes[cage] += prod_volume

                placed = True
                break

            # Case 2: New order can be added to cage.

            elif (
                len(cage_orders[cage])
                < cage_data.max_orders_per_cage
            ):

                cage_orders[cage].append(order)

                cages[cage].append(prod)

                # NEW: Record originating order.
                cage_contents[cage].append((prod, order))

                cage_weights[cage] += prod_weight
                cage_volumes[cage] += prod_volume

                placed = True
                break

        # Case 3: No existing cage is suitable.

        if not placed:

            cage_idx += 1

            cages[cage_idx] = [prod]
            cage_orders[cage_idx] = [order]

            # NEW: Initialise product-order contents.
            cage_contents[cage_idx] = [(prod, order)]

            cage_weights[cage_idx] = prod_weight
            cage_volumes[cage_idx] = prod_volume

            placed = True

            debug(
                f"Created cage {cage_idx} "
                f"for product {prod}, order {order}."
            )

        # Record successful placement.

        assigned_occurrences[(prod, order)] += 1

        num_placed += 1

        # Periodic progress reporting.

        if num_placed % 100 == 0 or num_placed == num_prods:

            debug(
                f"Progress: {num_placed}/{num_prods} "
                f"occurrences placed "
                f"({100 * num_placed / num_prods:.1f}%). "
                f"Cages: {len(cages)}. "
                f"Remaining: {len(prods_sorted)}."
            )

    debug("Main placement loop completed.")

    # --------------------------------------------------
    # 7. Validate product assignments
    # --------------------------------------------------

    debug("Validating product assignments.")

    original_occurrences = Counter(prods_in_orders)

    assert assigned_occurrences == original_occurrences, (
        "Product-order occurrences do not match "
        "the original orders."
    )

    assert (
        sum(len(cage) for cage in cages.values())
        == num_prods
    ), "Incorrect total number of placed products."

    # NEW: Validate recorded cage contents.
    recorded_occurrences = Counter(
        occurrence
        for contents in cage_contents.values()
        for occurrence in contents
    )

    assert recorded_occurrences == original_occurrences, (
        "Recorded cage contents do not match "
        "the original product-order occurrences."
    )

    # Validate cage capacities and order limits.

    for cage in cages:

        assert (
            cage_weights[cage]
            <= cage_weight_capacity + 1e-9
        ), f"Cage {cage} exceeds weight capacity."

        assert (
            cage_volumes[cage]
            <= cage_volume_capacity + 1e-9
        ), f"Cage {cage} exceeds volume capacity."

        assert (
            len(cage_orders[cage])
            <= cage_data.max_orders_per_cage
        ), f"Cage {cage} exceeds maximum orders."

    debug("All validation checks passed.")

    # --------------------------------------------------
    # 8. Create batches
    # --------------------------------------------------

    num_batches = math.ceil(
        len(cages) / cages_per_batch
    )

    batches = {}

    for b in range(num_batches):

        batches[b+1] = {
            k: v
            for k, v in cages.items()
            if k in range(
                b * cages_per_batch + 1,
                (b + 1) * cages_per_batch + 1
            )
        }

    # --------------------------------------------------
    # 9. Calculate picking distances for the original batches
    # --------------------------------------------------

    _, batch_distances, _ = calculate_distance_all_batches(
        batches=batches,
        aisle_assignments=aisle_assignments,
        warehouse_data=warehouse_data
    )

    # Sort batches in decreasing order of picking distance.
    # Keep their original batch IDs.
    batches_sorted = sorted(
        batches.items(),
        key=lambda item: batch_distances[item[0]],
        reverse=True
    )

    batches_sorted = {
        new_id: batch
        for new_id, (_, batch) in enumerate(batches_sorted, start=1)
    }

    # --------------------------------------------------
    # 10. Sort and renumber cages, keeping contents aligned
    # --------------------------------------------------

    batches_cage_sorted = {}
    batch_contents = {}
    aisles_visited = {}

    for batch_id, batch in batches_sorted.items():
        # Retain original cage IDs until the contents have been copied.
        sorted_cages = sorted(
            batch.items(),
            key=lambda item: cage_weights[item[0]],
            reverse=True
        )

        batches_cage_sorted[batch_id] = {}
        batch_contents[batch_id] = {}

        for new_cage_id, (old_cage_id, products) in enumerate(
            sorted_cages, start=1
        ):
            batches_cage_sorted[batch_id][new_cage_id] = products
            batch_contents[batch_id][new_cage_id] = dict(
                Counter(cage_contents[old_cage_id])
            )

        # Aisles are properties of batches, not cage indices.
        aisles_visited[batch_id] = list({
            prod_assignments[prod][0]
            for products in batch.values()
            for prod in products
        })

    # Check the renumbering has preserved every product-order occurrence.
    reordered_occurrences = Counter()
    for batch in batch_contents.values():
        for cage in batch.values():
            reordered_occurrences.update(cage)
    assert reordered_occurrences == original_occurrences, (
        "Renumbered cage contents do not match original orders."
    )

    # --------------------------------------------------
    # 11. Final summary
    # --------------------------------------------------

    elapsed = time.perf_counter() - start_time
    debug(
        f"HEURISTIC COMPLETE: "
        f"{num_prods} occurrences, "
        f"{len(cages)} cages, "
        f"{num_batches} batches, "
        f"{elapsed:.4f} seconds."
    )

    return batches_cage_sorted, aisles_visited, batch_contents