from itertools import product
from typing import Any
import numpy as np
from slap.comp_tests.dataclasses import WarehouseData

def calculate_distance_one_batch(
    aisle_assignments:dict, 
    batch:dict[str,list[int]], 
    warehouse_data:WarehouseData
) -> float:
    """Calculates the distance travelled in picking a single order

    Args:
        aisle_assignments: a dictionary containing the products (keys) and locations (values). Locations can either be slot tuples or aisle integers
        batch: Single batch of several cages.
        warehouse_data: Warehouse-related data.

    Returns:
        The total distance travelled in picking the order.
    """
   
    # protect against infeasibility
    if len(aisle_assignments) == 0:
        return 0
    
    M, N = warehouse_data.between_aisle_dist, warehouse_data.between_bay_dist

    min_dist = np.inf

    prods_in_batch = set([prod for cage in batch for prod in batch[cage]])

    aisle_assignments_batch = {k:[x for x in aisle_assignments[k] if x in prods_in_batch] for k,_ in aisle_assignments.items()}

    # Product -> possible aisles
    product_dict = {}

    for aisle, products in aisle_assignments_batch.items():
        for prod in products:
            product_dict.setdefault(prod, []).append(aisle)

    # Generate all possible choices
    all_assignments = []

    for choices in product(*product_dict.values()):

        assignment = {aisle: [] for aisle in aisle_assignments_batch}
        for prod, aisle in zip(product_dict, choices):
            assignment[aisle].append(prod)
        all_assignments.append(assignment)

    for assignment in all_assignments:
        aisles = [a for a in assignment.keys() if len(assignment[a])>0]

        # calculate the number of aisles
        num_occupied_aisles = len(aisles)
    
        if num_occupied_aisles == 0:
            return 0
    
        # calculate the number of consecutive aisle direction mismatches
        aisle_pairs = list(zip(aisles,aisles[1:]))          
        pens = []
        for pair in aisle_pairs:
            pens.append((abs(pair[0]-pair[1])+1)%2)
        mismatch = sum(pens)
    
        # calculating the indicator variables
        F_down = (min(aisles)) % 2 # if first aisle is a down
        L_up = (max(aisles)+1) % 2 # if last aisle is an up

        # calculate the distance travelled traversing a single aisle
        L = (warehouse_data.num_bays+1)*N # distance travelled traversing one aisle

        # calculate the horizontal distance to the final aisle
        H = (max(aisles))*M  # horizontal distance to final aisle

        # calculate the total distance for that order
        distance = L*(num_occupied_aisles + mismatch + F_down + L_up) + 2*max(M,H)

        if distance < min_dist:
            min_dist = distance
            
    return distance
    

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