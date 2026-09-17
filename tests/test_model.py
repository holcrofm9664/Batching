import pytest
import random
import pandas as pd
import numpy as np
from itertools import product
from slap.utils.model_preprocessing_function import preprocessing_function_batching
from slap.eval.evaluation import calculate_distance_all_trips
from slap.models.batching_model import batching_model


pick_data = pd.read_csv("tests/pick_data.csv")
solution_allocation = pd.read_csv("tests/solution_allocation.csv")

random.seed(123)
np.random.seed(123)

num_products_combs = [10,20]
num_orders_combs = [3,12]
min_order_size_combs = [2,2]
max_order_size_combs = [3,4]
num_aisles_combs = [5,10]
num_bays_combs = [10,20]

instances = []

for (
    num_products, 
    num_orders, 
    min_order_size, 
    max_order_size, 
    num_aisles, 
    num_bays
) in product(
    num_products_combs, 
    num_orders_combs, 
    min_order_size_combs, 
    max_order_size_combs, 
    num_aisles_combs, 
    num_bays_combs
):
    instance = preprocessing_function_batching(
        pick_data=pick_data,
        solution_allocation=solution_allocation,
        num_products=num_products,
        num_orders = num_orders,
        min_order_size=min_order_size,
        max_order_size=max_order_size,
        num_aisles=num_aisles,
        num_bays=num_bays
    )
    
    instances.append(instance)


@pytest.mark.parametrize(
    "instance",
    
        instances
    ,
)
def test_model_outputs(
    instance
):

    distance, trips_dict = batching_model(
        **instance
    )

    # ----- check output types ----------------------------------------------------------------
    
    assert type(distance) == float or distance == None
    assert type(trips_dict) == dict or trips_dict == None

    if type(trips_dict) == dict:

        # test that each trip is a dictionary
        for trip in trips_dict.values():
            assert type(trip) == dict


            for cage in trip.values():
                assert type(cage) == list

                for prod in cage:
                    assert type(prod) == int

    # ----- check distances -------------------------------------------------------------------
    
    distance_eval, dist_by_trip = calculate_distance_all_trips(
        trips = trips_dict,
        aisle_assignments=instance["aisle_assignments"],
        between_aisle_dist=instance["between_aisle_dist"],
        between_bay_dist=instance["between_bay_dist"],
        num_bays=instance["num_bays"]
    )
    
    assert distance == distance_eval

    print(f"dist_by_trip: {dist_by_trip}")
    # ----- check cage weight and volume capacities -------------------------------------------
    
    weights_dict, volumes_dict = instance["weights_dict"], instance["volumes_dict"]
    cage_weight_capacity = instance["cage_weight_capacity"]
    cage_volume_capacity = instance["cage_volume_capacity"]

    for trip in trips_dict.values():
        for cage in trip:
            # check cage weight capacity not exceeded
            assert sum(
                [weights_dict[p] for p in trip[cage]]
                ) <= cage_weight_capacity
            
            # check cage volume capacity not exceeded
            assert sum(
                [volumes_dict[p] for p in trip[cage]]
                ) <= cage_volume_capacity

    # ----- check that all products in oredrs are assigned ------------------------------------
    
    orders = instance["orders"]

    prods_orders = sorted(
        [
            p
            for o in orders 
            for p in o
        ]
    )

    prods_trips = sorted(
        [
            p for trip in trips_dict
            for cage in trips_dict[trip]
            for p in trips_dict[trip][cage]
        ]
    )

    # assert that all products in orders are assigned to batches
    assert prods_orders == prods_trips

    # test that individual distances match (i.e. the set of distances found by model is the set of distances found by function)

    # test that, if a batch has one cage free, then another batch doesn't use only one batch
    # - is this ever optimal (maybe at the edges?)
