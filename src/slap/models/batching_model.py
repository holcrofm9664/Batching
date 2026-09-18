import gurobipy as gp
from gurobipy import GRB
from typing import Any
from slap.comp_tests.dataclasses import (
    WarehouseData,
    CageData
)
 
def batching_model(
    orders:dict[int,list[int]], 
    aisle_assignments:dict[int,list[int]], 
    max_batches:int, 
    weights_dict:dict[int,int], 
    volumes_dict:dict[int,int], 
    warehouse_data:WarehouseData, 
    cage_data:CageData, 
    cages_per_batch:int=5,  
    **unused:Any
    ) -> tuple[float, dict[Any,dict[Any,list[int]]]]:
    """Constructs a set of batches from a set of orders to minimise distance.
    
    Args:
        orders: Orders constituting items which cannot be picked in different trips.
        aisle_assignments: Current assignment of products to aisles.
        max_batches: Maximum permitted number of batches.
        weights_dict: Product weights.
        volumes_dict: Product volumes.
        warehouse_data: Warehouse-related data.
        cage_data: Cage-related data.
        cages_per_batch: the number of cages each split can be split into
    
    Returns:
        The final batches and corresponding total distance.
    """

    W, V = weights_dict, volumes_dict

    # adjust for prudential liquid-fill
    cage_weight_capacity = cage_data.fill_percent*cage_data.cage_weight_capacity
    cage_volume_capacity = cage_data.fill_percent*cage_data.cage_volume_capacity

    # orders need to be in a dictionary such that we can access the keys
    if type(orders) == list:
        orders = {i+1:order for i,order in enumerate(orders)}

    # filter out empty orders
    orders = {k:v for k,v in orders.items() if len(v) > 0}

    # the sets
    P = set(prod for aisle in aisle_assignments for prod in aisle_assignments[aisle]) # the set of products
    T = range(1, max_batches + 1) # set of batches
    O = orders.keys() # set of orders (unbreakable groups of orders)
    A = range(warehouse_data.num_aisles) # set of aisles
    C = range(1, cages_per_batch + 1) # the set of cages

    # if aisle a contains product k
    A_dict = {
        (k, a): int(k in aisle_assignments[a])
        for k in P
        for a in A
    }

    # if order o contains product k
    B_dict = {
        (o, k): int(k in orders[o])
        for o in orders
        for k in P
    }

    W = {
        o: sum(weights_dict[prod] for prod in order)
        for o, order in orders.items()
    }

    V = {
        o: sum(volumes_dict[prod] for prod in order)
        for o, order in orders.items()
    }

    # calculating remaining parameters
    L = warehouse_data.between_bay_dist*(warehouse_data.num_bays + 1) 
    M = warehouse_data.between_aisle_dist

    # our model
    model = gp.Model("batching_strict_s")

    zp_combs = [
        (o,a,k)
        for o in O 
        for a in A 
        for k in P 
        if (
            B_dict[(o,k)]==1 
            and A_dict[(k,a)]==1
        )
    ]
    
    p_combs = [
        (t,a,b) 
        for t in T 
        for a in A 
        for b in A 
        if b > a
    ]

    A_even = [
        a 
        for a in A 
        if a%2 == 0
    ]

    A_odd = [
        a 
        for a in A
        if a%2 == 1
    ]

    # the variables
    y = model.addVars(O, T, C, vtype = GRB.BINARY, name = "y") # if order o is assigned to cage c of batch t
    z = model.addVars(T, A, vtype = GRB.BINARY, name = "z") # if batch t visits aisle a
    zp = model.addVars(zp_combs, vtype = GRB.BINARY, name="zp") # if product k is picked from aisle a for order o
    f = model.addVars(T, A, vtype = GRB.BINARY, name = "f") # if aisle a is the first aisle with a pick for batch t
    q = model.addVars(T, A, vtype = GRB.BINARY, name = "q") # if aisle a is the last aisle with a pick for batch t
    q_idx = model.addVars(T, lb=0, ub = warehouse_data.num_aisles, vtype = GRB.INTEGER, name = "q_index") # the index of the last picked-from aisle for batch t
    F = model.addVars(T, vtype = GRB.BINARY, name = "F") # if the index of the first picked-from aisle is odd
    Q = model.addVars(T, vtype = GRB.BINARY, name = "Q") # if the index of the last aisle picked-from aisle is even
    w = model.addVars(T, vtype = GRB.BINARY, name = "w") # auxiliary variable which takes the value 1 if fewer than two aisles are picked from for batch t
    Pen = model.addVars(T, vtype = GRB.BINARY, name = "First_aisle_only_penalty") # an indicator for whether only the first aisle contains a pick (and a horizontal penalty of the 2M needs to be applied)
    p = model.addVars(p_combs, vtype = GRB.BINARY, name = "p") # if aisles a and b share a direction and form a consecutive pair of picked-from aisles for batch t
    E = model.addVars(T, vtype = GRB.BINARY, name = "E") # if batch t is non-empty
    dist = model.addVars(T, vtype=GRB.CONTINUOUS, name="dist") # the distance travelled for batch T

    # the constraints

    for o in O:
        model.addConstr(
            gp.quicksum(y[o,t,c] for t in T for c in C) == 1,
            name = f"assign_order_{o}_to_exactly_one_cage_of_one_batch"
        )

    for t in T:
        model.addConstr(
            q_idx[t] == gp.quicksum(a*q[t,a] for a in A),
            name = f"retrieving_the_index_of_the_last_aisle_with_a_pick_for_batch_{t}"
        )

        model.addConstr(
            F[t] == gp.quicksum(f[t,a] for a in A_odd),
            name = f"if_the_first_aisle_with_a_pick_in_batch_{t}_is_odd_indexed"
        )

        model.addConstr(
            Q[t] == gp.quicksum(q[t,a] for a in A_even),
            name = f"if_the_last_aisle_with_a_pick_for_batch_{t}_is_even_indexed"
        )

        model.addConstr(
            gp.quicksum(z[t,a] for a in A) >= 2*(1-w[t]),
            name = f"if_fewer_than_two_aisles_visited_then_w_forced_to_one_trip_{t}"
        )

        model.addGenConstrAnd(
            Pen[t], [w[t],z[t,0]],
            name = f"enforce_first_aisle_penalty_trip_{t}_if_fewer_than_two_aisles_contain_picks_and_first_aisle_contains_a_pick"
        )

        model.addConstr(
            gp.quicksum(f[t,a] for a in A) + E[t] == 1,
            name = f"exactly_one_aisle_is_the_first_aisle_with_a_pick_for_batch_{t}_if_the_batch_is_non_empty"
        )

        model.addConstr(
            gp.quicksum(q[t,a] for a in A) + E[t] == 1,
            name = f"exactly_one_aisle_is_the_last_aisle_with_a_pick_for_batch_{t}_if_the_batch_is_non_empty"
        )

        model.addConstr(
            gp.quicksum(y[o,t,c] for o in O for c in C) + E[t] >= 1,
            name = f"if_no_products_assigned_to_trip_{t}_then_E_{t}_equals_1"
        )
        
        for o in O:
            model.addConstr(
                E[t] <= 1 - gp.quicksum(y[o,t,c] for c in C),
                name = f"set_E_to_0_if_batch_{t}_is_non-empty"
            )

        for c in C:
            model.addConstr(
                gp.quicksum(y[o,t,c]*V[o] for o in O) <= cage_volume_capacity,
                name = f"cage_volume_capacity"
            )

            model.addConstr(
                gp.quicksum(y[o,t,c]*W[o] for o in O) <= cage_weight_capacity,
                name = f"cage_weight_capacity"
            )

        model.addConstr(
            dist[t] == L * (gp.quicksum(z[t,a] for a in A) + gp.quicksum(p[t,a,b] for a in A for b in A if b > a if (t,a,b) in p) + F[t] + Q[t] - E[t]) 
                    + 2 * M * q_idx[t]
                    + 2 * M * Pen[t],
            name = f"the_distance_for_trip_{t}"
        )

    for t, a, b in p_combs:
        if (b-a)%2 == 0:
            model.addConstr(
                p[t,a,b] >= z[t,a] + z[t,b] - gp.quicksum(z[t,k] for k in range(a+1,b)) - 1,
                name = f"enforce_p_{t}_{a}_{b}"
            )
        
    for t in T:
        for a in A:
            model.addConstr(
                f[t,a] <= z[t,a],
                name = f"aisle_{a}_can_only_be_the_first_aisle_with_a_pick_for_batch_{t}_if_it_contains_a_pick"
            )

            model.addConstr(
                q[t,a] <= z[t,a],
                name = f"aisle_{a}_can_only_be_the_last_aisle_with_a_pick_for_batch_{t}_if_it_contains_a_pick"
            )

            model.addGenConstrIndicator(
                f[t,a], 1, gp.quicksum(z[t,k] for k in range(a)) == 0,
                name = f"if_aisle_{a}_is_the_first_aisle_to be_visited_for_batch_{t}_then_no_previous_aisles_have_been_visited_in_that_batch"
            )

            model.addGenConstrIndicator(
                q[t,a], 1, gp.quicksum(z[t,k] for k in range(a+1, warehouse_data.num_aisles)) == 0,
                name = f"if_aisle_{a}_is_the_last_aisle_to_be_visited_for_batch_{t}_then_no_further_aisles_will_be_visited_in_that_batch"
            )

    for t in T:
        for c in C:
            for o,a,k in zp_combs:
                model.addConstr(
                    z[t,a] >= y[o,t,c] + zp[o,a,k] - 1,
                    name = f"if_aisle_{a}_entered_for_order_{o}_and_the_order_is_in_batch_{t}_then_the_batch_must_also_enter_that_aisle"
                )

    for o, a, k in zp_combs:
        model.addConstr(
            gp.quicksum(zp[o,a,k] for a in A if (o,a,k) in zp) == 1,
            name = f"pick_product_{k}_from_exactly_one_aisle_for_order_{o}"
        )
    
    for t in T:
        if t > 1:
            model.addConstr(
                dist[t] >= dist[t-1],
                name = f"trip_symmetry_breaking_trip_{t}"
            )

        for c in C:
            if c > 1:
                model.addConstr(
                    gp.quicksum(W[o]*y[o,t,c] for o in O) <= gp.quicksum(W[o]*y[o,t,c-1] for o in O),
                    name = f"cage_symmetry_breaking_cage_{c}_trip_t"
                )

    model.setObjective(
        gp.quicksum(dist[t] for t in T), GRB.MINIMIZE
    )

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        return None, None

    else:
        distance = model.ObjVal
        trips_dict = {}

        for trip in T:
            trip_dict = {}
            for cage in C:
                products = [
                    prod
                    for order in O
                    if y[order, trip, cage].X > 0.5
                    for prod in orders[order]
                ]
                trip_dict[f"cage{cage}"] = list(products)

            trips_dict[f"trip{trip}"] = trip_dict

    return distance, trips_dict