from itertools import product
from typing import Tuple, Any
import numpy as np

def calculate_distance_one_trip(aisle_assignments:dict, trip:dict[str,list[int]], between_aisle_dist:float, between_bay_dist:float, num_bays:int) -> float:
    """
    Calculates the distance travelled in picking a single order

    Inputs:
    - aisle_assignments: a dictionary containing the products (keys) and locations (values). Locations can either be slot tuples or aisle integers
    - trip: a single trip of several cages
    - between_aisle_dist: the distance between two consecutive aisles in the warehouse
    - between_bay_distance: the distance between two consecutive bays in the warehouse
    - num_bays: the number of bays in the warehouse

    Outputs:
    - distance: the total distance travelled in picking the order
    """
   
    # protect against infeasibility
    if len(aisle_assignments) == 0:
        return 0
    
    M, N = between_aisle_dist, between_bay_dist

    min_dist = np.inf

    prods_in_trip = set([prod for cage in trip for prod in trip[cage]])

    aisle_assignments_trip = {k:[x for x in aisle_assignments[k] if x in prods_in_trip] for k,_ in aisle_assignments.items()}

    # Product -> possible aisles
    product_dict = {}

    for aisle, products in aisle_assignments_trip.items():
        for prod in products:
            product_dict.setdefault(prod, []).append(aisle)

    # Generate all possible choices
    all_assignments = []

    for choices in product(*product_dict.values()):

        assignment = {aisle: [] for aisle in aisle_assignments_trip}
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
        L = (num_bays+1)*N # distance travelled traversing one aisle

        # calculate the horizontal distance to the final aisle
        H = (max(aisles))*M  # horizontal distance to final aisle

        # calculate the total distance for that order
        distance = L*(num_occupied_aisles + mismatch + F_down + L_up) + 2*max(M,H)

        if distance < min_dist:
            min_dist = distance
            
    return distance
    

def calculate_distance_all_trips(trips:dict[Any,dict[Any,list[int]]], aisle_assignments:dict[int,list[int]], between_aisle_dist:int, between_bay_dist:int, num_bays:int) -> Tuple[int, dict[Any,int]]:
    """
    Calculates the distance for all trips

    Inputs:
    - trips: the set of trips returned by the optimisation
    - aisle_assignments: the set of aisle assignments (can include scattered storage)
    - between_aisle_dist: the distance between consecutive aisles
    - between_bay_dist: the distance between consecutive bays
    - num_bays: the number of bays in each aisle

    Outputs:
    - total_distance: the total distance across all trips
    - dist_by_trip: the distance of each trip
    """
    
    dist_by_trip = {}

    for i, trip in trips.items():
        distance = calculate_distance_one_trip(aisle_assignments=aisle_assignments,
                                               trip=trip,
                                               between_aisle_dist=between_aisle_dist,
                                               between_bay_dist=between_bay_dist,
                                               num_bays=num_bays)
        dist_by_trip[i] = distance

    total_distance = sum(dist_by_trip.values())

    return total_distance, dist_by_trip