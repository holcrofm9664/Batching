import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def scatter_frequency_assignments(scatter_frequency:int, prods:pd.DataFrame, num_aisles:int, num_bays:int, slot_capacity:int) -> dict[int,list[int]]:
    
    C = num_aisles*num_bays*slot_capacity

    A = [a for a in range(num_aisles)]
    P = prods

    model = gp.Model("scatter_frequency_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")

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



def aisle_directionality_assignments(prods:pd.DataFrame, num_aisles:int, num_bays:int, slot_capacity:int, same_direction:bool=False, different_direction:bool=False) -> dict[int,list[int]]:

    if same_direction==True and different_direction==True:
        print(f"cannot force aisles with a product to be both same and different direction")
        return None
    
    C = num_aisles*num_bays*slot_capacity

    A = [a for a in range(num_aisles)]
    A_even = [a for a in A if a%2==0]
    A_odd = [a for a in A if a%2==1]
    P = prods

    model = gp.Model("aisle_directionality_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")
    z = model.addVar(vtype=GRB.BINARY, name="z") # auxiliary variable to ensure either all assigned to odd or even-indexed aisles

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
                gp.quicksum(x[a,k] for a in A_even) <= 2*z,
                name = f"same_aisle_index_1"
            )

            model.addConstr(
                gp.quicksum(x[a,k] for a in A_odd) <= 2*(1-z),
                name = f"same_aisle_index_2"
            )

    if different_direction:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A_even) == gp.quicksum(x[a,k] for a in A_odd),
            name = f"different_aisle_index"
        )

    model.setObjective(a, GRB.MINIMIZE)

    model.optimize()

    aisle_assignments = {a:[] for a in A}

    for a in A:
        for k in P:
            if x[a,k].X > 0.5:
                aisle_assignments[a].append(k)

    return aisle_assignments


def scalability_assignments(prods_by_dem:list[int], num_aisles:int, num_bays:int, slot_capacity:int, top_frac:float) -> dict[int,list[int]]:
    """
    Inputs:
    - prods_by_dem: the products to be assigned, sorted in DECREASING order of demand
    
    """

    num_prods = len(prods_by_dem)
    num_scattered_prods = math.floor(top_frac*num_prods)

    C = num_bays*slot_capacity

    P = prods_by_dem
    S = prods_by_dem[:num_scattered_prods]
    U = prods_by_dem[num_scattered_prods:]
    A = [a for a in range(num_aisles)]

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


def permutation_assignments(prods_by_dem:list[int], current_assignments:dict[int,int], num_aisles:int, num_bays:int, slot_capacity:int)->dict[int,list[int]]:
    """
    Inputs:
    - current_assignments: the assignments used by Tesco, key is product, value is aisle

    
    """

    C = num_bays*slot_capacity

    warehouse_capacity = num_aisles*num_bays*slot_capacity
    num_scattered_prods = warehouse_capacity - len(prods_by_dem)

    A = [a for a in range(num_aisles)]
    P = prods_by_dem
    S = prods_by_dem[:num_scattered_prods]

    model = gp.Model("permutation_assignments")

    x = model.addVars(A, P, vtype=GRB.BINARY, name="x")

    for k in P:
        model.addConstr(
            x[current_assignments[k],k]==1,
            name = f"fix_product_{k}_to_aisle_{current_assignments[k]}"
        )

    for k in S:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) == 2,
            name = f"scatter_product_{k}"
        )

    for a in A:
        model.addConstr(
            gp.quicksum(x[a,k] for a in A) <= C,
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