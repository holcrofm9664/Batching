from collections import deque
from itertools import product
import math
from slap.utils.dataclasses import CageData
from typing import Any

def batching_heuristic(
        orders:dict[int,list[int]],
        weights_dict:dict[int,float],
        volumes_dict:dict[int,float],
        cage_data:CageData,
        prod_assignments:dict[int:tuple[int,int]],
        cages_per_batch:int=5,
        **unused:Any
):

    cage_weight_capacity = cage_data.cage_weight_capacity * cage_data.fill_frac
    cage_volume_capacity = cage_data.cage_volume_capacity * cage_data.fill_frac

    sol_prods = {x for x in prod_assignments.keys()}

    aisles = [x for x in range(70)]
    bays = [x+1 for x in range(60)]

    slot_idx = {}
    for aisle,bay in product(aisles,bays):
        slot_idx[(aisle,bay)] = (
            60*aisle + bay if aisle % 2 == 0
            else 60*aisle + (61-bay)
        )

    prod_idx = {k:slot_idx[prod_assignments[k]] for k in sol_prods}

    cage_idx = 1
    
    cages, cage_orders = {cage_idx:[]}, {cage_idx:[]}
    cage_weights, cage_volumes = {cage_idx:0}, {cage_idx:0}

    prods_in_orders = [prod for order in orders.values() for prod in order]

    prods_sorted = deque(sorted(prods_in_orders,key = lambda x:prod_idx[x]))

    orders = {k:v.copy() for k,v in orders.items()}

    while prods_sorted:
        prod = prods_sorted[0]
        for order in [order for order in orders if prod in orders[order]]:
            placed = False
            for cage in cages:
                if (
                    order in cage_orders[cage]
                    and cage_weights[cage] + weights_dict[prod] <= cage_weight_capacity
                    and cage_volumes[cage] + volumes_dict[prod] <= cage_volume_capacity
                ):
                    cages[cage].append(prod)
                    cage_weights[cage] += weights_dict[prod]
                    cage_volumes[cage] += volumes_dict[prod]
                    prods_sorted.popleft()
                    orders[order].remove(prod)
                    placed = True
                    break
                elif (
                    len(cage_orders[cage]) < cage_data.max_orders_per_cage
                    and cage_weights[cage] + weights_dict[prod] <= cage_weight_capacity
                    and cage_volumes[cage] + volumes_dict[prod] <= cage_volume_capacity
                ):
                    cage_orders[cage].append(order)
                    cages[cage].append(prod)
                    cage_weights[cage] += weights_dict[prod]
                    cage_volumes[cage] += volumes_dict[prod]
                    prods_sorted.popleft()
                    orders[order].remove(prod)
                    placed = True
                    break
                else:
                    continue

            if placed == False:
                cage_idx += 1
                cages[cage_idx] = [prod]
                cage_orders[cage_idx] = [order]
                cage_weights[cage_idx] = weights_dict[prod]
                cage_volumes[cage_idx] = volumes_dict[prod]
                prods_sorted.popleft()
                orders[order].remove(prod)
                placed = True
                break

    # creating the batches

    num_batches = math.ceil(len(cages)/cages_per_batch)
    batches = {}
    for b in range(num_batches):
        batches[b] = {
            k:v 
            for k,v 
            in cages.items() 
            if k in range(
                b * cages_per_batch + 1,
                (b+1) * cages_per_batch + 1
            )
        }

    aisles_visited = {
        batch_id: list({
            prod_assignments[prod][0]
            for cage in batch.values()
            for prod in cage
        })
        for batch_id, batch in batches.items()
    }

    return batches, aisles_visited
            