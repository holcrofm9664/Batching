import gurobipy as gp
from gurobipy import GRB
from typing import Any
from slap.utils.dataclasses import (
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
    time_limit:int,
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
        time_limit: Model time limit.
        cages_per_batch: the number of cages each split can be split into
    
    Returns:
        The solve status, time and gap, and final batches and corresponding total distance.
    """

    W, V = weights_dict, volumes_dict

    # adjust for prudential liquid-fill
    cage_weight_capacity = cage_data.fill_frac*cage_data.cage_weight_capacity
    cage_volume_capacity = cage_data.fill_frac*cage_data.cage_volume_capacity

    # orders need to be in a dictionary such that we can access the keys
    if type(orders) == list:
        orders = {i+1:order for i,order in enumerate(orders)}

    # filter out empty orders
    orders = {k:v for k,v in orders.items() if len(v) > 0}

    # create the demands dict (number of each product in each order)
    dem_dict = {}
    for order_num, order in orders.items():
        all_prods = set(order)
        for prod in all_prods:
            dem_dict[(order_num,prod)] = len([p for p in order if p == prod])

    # create the m dict (max of each product from each order that can be assigned to one cage)
    m_dict = {}

    for (o,k) in dem_dict.keys():
        m = min(dem_dict[(o,k)], cage_weight_capacity//W[k], cage_volume_capacity//V[k])
        m_dict[(o,k)] = m
    
    # the sets
    P = set(prod for aisle in aisle_assignments for prod in aisle_assignments[aisle]) # the set of products
    T = range(1, max_batches + 1) # set of batches
    O = orders.keys() # set of orders (unbreakable groups of orders)
    A = range(warehouse_data.num_aisles) # set of aisles
    C = range(1, cages_per_batch + 1) # the set of cages

    # create the order constituents dictionary 
    order_const_dict = {}
    for o in O:
        prods = [k for k in P if (o,k) in dem_dict]
        order_const_dict[o] = prods

    # if aisle a contains product k
    A_dict = {
        (k, a): int(k in aisle_assignments[a])
        for k in P
        for a in A
    }

    # create R_k, the set of aisles k is stored in
    R = {}
    for k in P:
        aisles = [a for a in A if A_dict[(k,a)] == 1]
        R[k] = aisles

    # if order o contains product k
    B_dict = {
        (o, k): int(k in orders[o])
        for o in orders
        for k in P
    }

    # calculating remaining parameters
    L = warehouse_data.between_bay_dist*(warehouse_data.num_bays + 1) 
    M = warehouse_data.between_aisle_dist

    # our model
    model = gp.Model("batching_strict_s")

    model.setParam("TimeLimit", time_limit)
    
    zp_combs = [
        (k,t,a)
        for k in P
        for t in T
        for a in A 
        if A_dict[(k,a)]==1
    ]
    
    p_combs = [
        (t,a,b) 
        for t in T 
        for a in A 
        for b in A 
        if b > a
    ]

    yp_combs = [
        (o,k,t,c)
        for o in O
        for k in order_const_dict[o]
        for t in T
        for c in C
    ]

    y_combs = [
        (o,k,t,c)
        for o in O
        for k in order_const_dict[o]
        for t in T
        for c in C
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
    y = model.addVars(y_combs, vtype=GRB.INTEGER, name="y") # the quantity of product k from order o assigned to cage c of trip t
    yp = model.addVars(yp_combs, vtype = GRB.BINARY, name = "yp") # if any products of type k from order o are assigned to cage c of batch t
    Y = model.addVars(O, T, C, vtype=GRB.BINARY, name = "Y") # if any products from order o are assigned to cage c of trip t
    YP = model.addVars(P, T, vtype=GRB.BINARY, name = "YP") # if any of product k assigned to batch t
    z = model.addVars(T, A, vtype = GRB.BINARY, name = "z") # if batch t visits aisle a
    zp = model.addVars(zp_combs, vtype = GRB.BINARY, name="zp") # if products of type k are picked from aisle a for order o
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
        for k in order_const_dict[o]:
            model.addConstr(
                gp.quicksum(y[o,k,t,c] for c in C for t in T) == dem_dict[(o,k)],
                name = f"assign_all_units_of_SKU_{k}_from_order_{o}"
            )
    
    for o, k, t, c in yp_combs:
        model.addConstr(
            Y[o,t,c] >= yp[o,k,t,c],
            name = f"define_Y"
        )

        m = m_dict[(o,k)]
        model.addConstr(
            y[o,k,t,c] <= m*yp[o,k,t,c],
            name = f"flag_yp_when_product_{k}_from_order_{o}_assigned_to_cage_{c}_of_trip_{t}"
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
            gp.quicksum(Y[o,t,c] for o in O for c in C) + E[t] >= 1,
            name = f"if_no_products_assigned_to_trip_{t}_then_E_{t}_equals_1"
        )

        model.addConstr(
            dist[t] == L * (gp.quicksum(z[t,a] for a in A) + gp.quicksum(p[t,a,b] for a in A for b in A if b > a if (t,a,b) in p) + F[t] + Q[t] - E[t]) 
                    + 2 * M * q_idx[t]
                    + 2 * M * Pen[t],
            name = f"the_distance_for_trip_{t}"
        )
        
    for o in O:
        for t in T:    
            for c in C:
                model.addConstr(
                    E[t] <= 1 - Y[o,t,c],
                    name = f"set_E_to_0_if_batch_{t}_is_non-empty"
                )

    for t in T:
        for c in C:
            model.addConstr(
                gp.quicksum(y[o,k,t,c]*W[k] for o in O for k in order_const_dict[o]) <= cage_weight_capacity,
                name = f"cage_weight_capacity"
            )

            model.addConstr(
                gp.quicksum(y[o,k,t,c]*V[k] for o in O for k in order_const_dict[o]) <= cage_volume_capacity,
                name = f"cage_volume_capacity"
            )

            model.addConstr(
                gp.quicksum(Y[o,t,c] for o in O) <= cage_data.max_orders_per_cage,
                name = f"permit_up_to_{cage_data.max_orders_per_cage}_orders_assigned_to_cage_{c}_of_trip_{t}"
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

    for o in O:
        for c in C:
            for k,t,a in zp_combs:
                if B_dict[(o,k)] == 1:
                    model.addConstr(
                        z[t,a] >= yp[o,k,t,c] + zp[k,t,a] - 1,
                        name = f"if_product_{k}_from_order_{o}_assigned_to_cage_{c}_of_trip_{t}_and_we_pick_that_product_from_aisle_{a}_then_that_trip_must_enter_that_aisle"
                    )
    for k in P:
        for t in T:
            model.addConstr(
                YP[k,t] == gp.quicksum(zp[k,t,a] for a in R[k]),
                name = f"if_product_{k}_assigned_to_batch_{t}_then_that_batch_must_pick_that_product"
            )
    
    for t in T:
        for c in C:
            for o in O:
                for k in order_const_dict[o]:
                    model.addConstr(
                        YP[k,t] >= yp[o,k,t,c],
                        name = f"flag_YP_if_batch_{t}_contains_any_units_of_product_{k}"
                    )
        if t > 1:
            model.addConstr(
                dist[t] >= dist[t-1],
                name = f"trip_symmetry_breaking_trip_{t}"
            )

        for c in C:
            if c > 1:
                model.addConstr(
                    gp.quicksum(W[k]*y[o,k,t,c] for o in O for k in order_const_dict[o]) <= gp.quicksum(W[k]*y[o,k,t,c-1] for o in O for k in order_const_dict[o]),
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
        trips = set(t for _, _, t, _ in y_combs)
        cages = set(c for _, _, _, c in y_combs)

        distance = model.ObjVal
        trips_dict = {}

        for t in trips:
            trip_dict = {}

            for c in cages:
                products = []

                for o in O:
                    for k in P:
                        if (o,k,t,c) in y_combs:
                            if y[o,k,t,c].X > 0:
                                products.extend([k] * int(y[o,k,t,c].X))

                trip_dict[f"cage {c}"] = products

            trips_dict[f"trip {t}"] = trip_dict

    return model.Status, model.Runtime, model.MIPGap, distance, trips_dict