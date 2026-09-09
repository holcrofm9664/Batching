import pandas as pd
import numpy as np
from typing import Tuple
import random
import math

def clean_dataframes(pick_data:pd.DataFrame, solution_allocation:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the dataframes by dropping NA values and renaming columns

    Inputs:
    - pick_data: the dataframe containing the pick data
    - solution_allocation: the dataframe containing storage assignments and product weights and volumes

    Outputs:
    - pick_data: the cleaned pick_data dataframe
    - solution_allocation: the cleaned solution_allocation dataframe
    """

    column_headers_dict = {"tpnd":"product",
                           "Store":"store_id",
                           "store":"store_id",
                           "shop":"store_id",
                           "Shop":"store_id",
                           "assignment":"trip",
                           "assignment_number":"trip",
                           "batch":"trip",
                           "Batch":"trip",
                           "uod_number":"cage",
                           "uod":"cage",
                           "Cage":"cage",
                           "datetime":"time",
                           "qty":"qty_to_pick",
                           "pick_qty":"qty_to_pick",
                           "Aisle":"aisle"
                           }
    
    # rename columns in both dataframes
    pick_data = pick_data.rename(columns = column_headers_dict)
    solution_allocation = solution_allocation.rename(columns = column_headers_dict)

    # drop NA values from the solution_allocation dataframe
    solution_allocation = solution_allocation.dropna()

    return pick_data, solution_allocation


def sample_products(num_prods:int, solution_allocation:pd.DataFrame, pick_data:pd.DataFrame) -> list[int]:
    """
    Samples products to be kept in the instance

    Inputs:
    - num_prods: the number of products to include in the instance
    - solution_allocation: the dataframe containing all products
    - pick_data: the pick data dataframe

    Outputs:
    - all_prods: a list of all products 
    - sampled_prods: the products we will use in our instance
    """
    
    all_prods = solution_allocation["product"].unique()

    if num_prods >= len(all_prods):
        print(f"Too many products requested. Returning all products")
        return list(int(x) for x in all_prods), list(int(x) for x in all_prods)

    unit_qty_dict = dict(pick_data.groupby("product")["qty_to_pick"].sum())

    items, frequencies = list(unit_qty_dict.keys()), list(unit_qty_dict.values())

    sampled_prods = np.random.choice(
        a = items,
        size = num_prods,
        replace = False,
        p = frequencies/np.sum(frequencies)
    )

    return list(int(x) for x in all_prods), list(int(x) for x in sampled_prods)


def weights_and_volumes(solution_allocation:pd.DataFrame, prod_subset:list[int]) -> Tuple[dict[int,float], dict[int,float]]:
    """ 
    Constructs a weights dictionary and a volumes dictionary from the dataframe

    Inputs:
    - solution_allocation: the dataframe containing the weights and volumes
    - prod_subset: the products remaining after we have filtered for time and a chosen product subset

    Outputs:
    - volume_dict: the product volumes
    - weight_dict: the product weights 
    """

    # construct the weights and volumes dictionaries
    weight_dict, volume_dict = {}, {}
    for prod in prod_subset:
        weight_dict[prod], volume_dict[prod] = solution_allocation[solution_allocation["product"]==prod]["weight"].to_list()[0], solution_allocation[solution_allocation["product"]==prod]["volume"].to_list()[0]

    return volume_dict, weight_dict


def create_orders(pick_data:pd.DataFrame, prod_subset:list[int], num_orders:int, min_order_size:int, max_order_size:int, volume_dict:dict[int,float], weight_dict:dict[int,float], cage_weight_capacity:int, cage_volume_capacity:int, fill_percent:float) -> list[list[int]]:
    """
    Creates a set of synthetic orders based on user specifications and historical product demands

    inputs:
    - pick_data: the pick data dataframe
    - prod_subset: the products remaining after we have filtered for time and a chosen product subset
    - num_orders: the number of orders to be generated
    - min_order_size: the minimum permitted order size
    - max_order_size: the maximum permitted order size
    - volume_dict: the product weights
    - weight_dict: the product volumes
    - cage_weight_capacity: the weight capacity of each cage
    - cage_volume_capacity: the volume capacity of each cage
    - fill_percent: the liquid-fill assumption


    outputs:
    - orders: the orders to be used in the optimisation
    """
    if len(prod_subset) == 0:
        return [], {}
    
    product_demands_dict = pick_data.groupby("product")["qty_to_pick"].sum().reindex(prod_subset,fill_value=0).to_dict()

    for prod in prod_subset:
        if prod not in product_demands_dict.keys():
            product_demands_dict[prod] = 0

    orders = []

    while len(orders) < num_orders:
        order_size = random.randint(min_order_size,max_order_size)

        order = random.choices(
            population = list(product_demands_dict.keys()),
            weights = list(product_demands_dict.values()),
            k=order_size
        )

        order_weight = sum([weight_dict[k] for k in order])
        order_volume = sum([volume_dict[k] for k in order])

        if order_weight < cage_weight_capacity*fill_percent and order_volume < cage_volume_capacity*fill_percent:
            orders.append(order)

    return orders, product_demands_dict


def create_max_batches(weights_dict:dict[int,float], volumes_dict:dict[int,float], orders:list[list[int]], cage_weight_capacity:int=400, cage_volume_capacity:int=45, fill_percent:float=0.85, buffer:float=0.25, cages_per_batch:int=5) -> int:
    """
    Takes in the orders and outputs a maximum number of batches the model is permitted to use

    Inputs:
    - weights_dict: a dictionary of the product weights
    - volumes_dict: a dictionary of the product volumes
    - orders: the orders generated for the instance
    - cage_weight_capacity: the weight capacity of the cage 
    - cage_volume_capacity: the volume capacity of the cage
    - fill_percent: the assumed liquid-fill percentage
    - buffer: the percentage buffer to allow the model access to more than the minimum number of cages needed for feasibility
    - cages_per_batch: the number of cages each batch is comprised of
    
    Outputs:
    - max_batches: the maximum allowed number of batches
    """

    cage_weight_capacity, cage_volume_capacity = cage_weight_capacity*fill_percent, cage_volume_capacity*fill_percent

    W = {
        o: sum(weights_dict[prod] for prod in order)
        for o, order in enumerate(orders)
    }

    V = {
        o: sum(volumes_dict[prod] for prod in order)
        for o, order in enumerate(orders)
    }

    num_cages, current_weight, current_volume = 1, 0, 0

    for order, _ in enumerate(orders):
        w, v = W[order], V[order]

        if (
            current_weight + w <= cage_weight_capacity
            and current_volume + v <= cage_volume_capacity
        ):
            current_weight += w
            current_volume += v
        else:
            num_cages += 1
            current_weight = w
            current_volume = v

    # offer more cages than required - the model may find a better solution with more than the minimum
    max_batches = math.floor((1+buffer)*math.ceil(num_cages / cages_per_batch) + 1)

    return max_batches


def create_aisle_assignments(prod_subset:list[int], num_zones:int, num_aisles:int, num_bays:int, solution_allocation:pd.DataFrame, product_demands_dict:dict[int,int], slot_capacity:int=2) -> Tuple[dict[int,list[int]],dict[int,list[int]]]:
    """
    Takes the zones, warehouse dimensions and product weights and demands and creates a synthetic set of assignments

    Inputs:
    - prod_subset: the products stored in the warehouse
    - num_zones: the number of zones
    - num_aisles: the number of aisles in the warehouse
    - num_bays: the number of bays in each aisle
    - solution_allocation: the allocation dataframe
    - product_demands_dict: the demands of each product
    - slot_capacity: the capacity of each (aisle,bay) pair

    Outputs:
    - aisle_assignments: the {aisle:list[products]} assignments dictionary
    """

    aisles = list(range(num_aisles))
    zones = np.array_split(aisles, num_zones)

    aisle_capacity = num_bays * slot_capacity

    # Products ordered from heaviest to lightest
    weights = (
        solution_allocation
        .set_index("product")["weight"]
        .to_dict()
    )

    prods_by_weight = sorted(
        prod_subset,
        key=lambda p: weights[int(p)],
        reverse=True
    )

    # Products ordered from highest to lowest demand
    prods_by_demand = sorted(
        product_demands_dict,
        key=product_demands_dict.get,
        reverse=True
    )

    current_assignments = {}
    pos = 0

    for zone in zones:
        num_slots = len(zone) * aisle_capacity

        zone_prods = set(prods_by_weight[pos:pos + num_slots])
        zone_prods_by_demand = [p for p in prods_by_demand if p in zone_prods]

        for idx, aisle in enumerate(zone):
            start = idx * aisle_capacity
            end = start + aisle_capacity

            for prod in zone_prods_by_demand[start:end]:
                current_assignments[prod] = aisle

        pos += len(zone_prods_by_demand)

    aisle_assignments = {v: [k for k, val in current_assignments.items() if val == v] for v in aisles}

    return aisle_assignments