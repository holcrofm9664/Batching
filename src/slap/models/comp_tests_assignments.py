import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass

@dataclass
class WarehouseData:
    """
    warehouse-related input data.

    attributes:
        num_aisles: the number of aisles in the warehouse
        num_bays: the number of bays in the warehouse
        slot_capacity: the number of unique products that can fit into each (aisle,bay) pair
        between_aisle_dist: the distance between consecutive aisles
        between_bay_dist: the distance between consecutive bays
    """
    num_aisles:int
    num_bays:int
    slot_capacity:int
    between_aisle_dist:int
    between_bay_dist:int

@dataclass
class OrdersData:
    num_orders:int
    min_order_size:int
    max_order_size:int

def scatter_frequency_assignments(
    warehouse_data:WarehouseData,
    prods:list[int],
    scatter_frequency:int
) -> dict[int,list[int]]:
    """
    Creates assignments for the scatter frequency computational tests

    Inputs:
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in each aisle
    - slot_capacity: the number of unique products that can be placed into each (aisle,bay) slot
    - prods: a list of products to be assigned to aisles
    - scatter_frequency: the number of aisles each product is assigned to

    Outputs:
    - aisle_assignments: the products assigned to each aisle
    """
    
    # aisle capacity
    C = warehouse_data.num_bays*warehouse_data.slot_capacity

    # sets 
    A = [a for a in range(warehouse_data.num_aisles)]
    P = prods

    model = gp.Model("scatter_frequency_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x") # if product k is assigned to aisle a

    for k in P:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == scatter_frequency,
            name = f"assign_product_{k}_to_exactly_{scatter_frequency}_aisles"
        )

    for a in A:
        model.addConstr(
            gp.quicksum(x[a,k] for k in P) <= C,
            name = f"capacity_of_aisle_{a}"
        )

    model.setObjective(a, GRB.MINIMIZE)

    model.optimize()

    aisle_assignments = {a:[] for a in A}

    for a in A:
        for k in P:
            if x[a,k].X > 0.5:
                aisle_assignments[a].append(k)

    return aisle_assignments


def aisle_directionality_assignments(
    warehouse_data:WarehouseData,
    prods:list[int], 
    same_direction:bool=True
) -> dict[int,list[int]]:
    """
    Creates assignments for the aisle directionality computational tests

    Inputs:
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in each aisle
    - slot_capacity: the number of unique products that can be placed into each (aisle,bay) slot
    - prods: a list of products to be assigned to aisles
    - same_direction: if products are to be scattered across aisles wuth the same 
      direction. If set of False, they will be placed only in aisles with a different 
      direction

    Outputs:
    - aisle_assignments: the products assigned to each aisle
    """

    C = warehouse_data.num_bays*warehouse_data.slot_capacity

    A = [a for a in range(warehouse_data.num_aisles)]

    A_even = [
        a
        for a in A 
        if a%2==0
    ]

    A_odd = [
        a 
        for a in A 
        if a%2==1
    ]

    P = prods

    model = gp.Model("aisle_directionality_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")
    z = model.addVars(P, vtype=GRB.BINARY, name="z") # auxiliary variable to ensure either all assigned to odd or even-indexed aisles

    for k in P:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == 2,
            name = f"assign_product_{k}_to_exactly_two_aisles"
        )

    for a in A:
        model.addConstr(
            gp.quicksum(x[a,k] for k in P) <= C,
            name = f"capacity_of_aisle_{a}"
        )

    if same_direction:
        for k in P:
            model.addConstr(
                gp.quicksum(x[a,k] for a in A_even) <= 2*z[k],
                name = f"same_aisle_index_1"
            )

            model.addConstr(
                gp.quicksum(x[a,k] for a in A_odd) <= 2*(1-z[k]),
                name = f"same_aisle_index_2"
            )

    else:
        for k in P:
            model.addConstr(
                gp.quicksum(x[a,k] for a in A_even) == gp.quicksum(x[a,k] for a in A_odd),
                name = f"different_aisle_index"
        )

    model.setObjective(a, GRB.MINIMIZE)

    model.optimize()
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        return None, None
    
    aisle_assignments = {a:[] for a in A}

    for a in A:
        for k in P:
            if x[a,k].X > 0.5:
                aisle_assignments[a].append(k)

    return aisle_assignments


def scalability_assignments( 
    warehouse_data:WarehouseData, 
    prods_by_dem:list[int],
    top_frac:float
) -> dict[int,list[int]]:
    """
    Creates assignments for the scalability computational tests

    Inputs:
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in each aisle
    - slot_capacity: the number of unique products that can be placed into each 
      (aisle,bay) slot
    - prods_by_dem: a list of products to be assigned to aisles, sorted in decreasing 
      order of demand
    - top_frac: the fraction of products that are to be scattered across two aisles

    Outputs:
    - aisle_assignments: the products assigned to each aisle
    """

    num_prods = len(prods_by_dem)
    num_scattered_prods = math.floor(top_frac*num_prods)

    C = warehouse_data.num_bays*warehouse_data.slot_capacity

    P = prods_by_dem
    S = prods_by_dem[:num_scattered_prods]
    U = prods_by_dem[num_scattered_prods:]
    A = [a for a in range(warehouse_data.num_aisles)]

    model = gp.Model("scalability_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")

    for k in S:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == 2,
            name = f"assign_product_{k}_to_two_aisles"
        )

    for k in U:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == 1,
            name = f"assign_product_{k}_to_one_aisle"
        )

    for a in A:
        model.addConstr(
            gp.quicksum(x[a,k] for k in P) <= C,
            name = f"capacity_of_aisle_{a}"
        )

    model.setObjective(a, GRB.MINIMIZE)

    model.optimize()

    aisle_assignments = {a:[] for a in A}

    for a in A:
        for k in P:
            if x[a,k].X > 0.5:
                aisle_assignments[a].append(k)

    return aisle_assignments


def permutation_assignments(
    warehouse_data:WarehouseData,
    prods_by_dem:list[int], 
    aisle_space_used:dict[int,int], 
) -> dict[int,list[int]]:
    """
    Creates assignments for the scalability computational tests

    Inputs:
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in each aisle
    - slot_capacity: the number of unique products that can be placed into each 
      (aisle,bay) slot
    - prods_by_dem: a list of products to be assigned to aisles, sorted in decreasing 
      order of demand
    - aisle_space_used: the aisle space already used by products assigned there

    Outputs:
    - aisle_assignments: the products assigned to each aisle
    """

    C = warehouse_data.num_bays*warehouse_data.slot_capacity

    warehouse_capacity = warehouse_data.num_aisles*warehouse_data.num_bays*warehouse_data.slot_capacity
    num_scattered_prods = warehouse_capacity - len(prods_by_dem)

    A = [a for a in range(warehouse_data.num_aisles)]
    P = prods_by_dem
    S = prods_by_dem[:num_scattered_prods]

    model = gp.Model("permutation_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")

    for k in S:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == 2,
            name = f"scatter_product_{k}"
        )

    for a in A:
        used_capacity = aisle_space_used[a]
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) <= C - used_capacity,
            name = f"capacity_of_aisle_{a}"
        )

    model.setObjective(a, GRB.MINIMIZE)

    model.optimize()

    aisle_assignments = {a:[] for a in A}

    for a in A:
        for k in P:
            if x[a,k].X > 0.5:
                aisle_assignments[a].append(k)

    return aisle_assignments