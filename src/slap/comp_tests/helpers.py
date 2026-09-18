import random
import math
import pandas as pd
from slap.comp_tests.dataclasses import (
    OrdersData,
    CageData
)

def weights_and_volumes(
    solution_allocation:pd.DataFrame, 
    prod_subset:list[int]
) -> tuple[dict[int,float], dict[int,float]]:
    """Constructs a weights dictionary and a volumes dictionary from the dataframe.

    Args:
        solution_allocation: Dataframe containing the weights and volumes.
        prod_subset: Products remaining after we have filtered for time and a chosen 
            product subset.

    Returns:
        The dictionaries of weights and volumes.
    """

    # construct the weights and volumes dictionaries
    df = (
        solution_allocation[
            solution_allocation["tpnd"].isin(prod_subset)
        ]
        .drop_duplicates(subset="tpnd")
        .set_index("tpnd")
    )

    weight_dict = df["weight"].to_dict()
    volume_dict = df["volume"].to_dict()
    
    return weight_dict, volume_dict

def create_orders(
        pick_data:pd.DataFrame, 
        prod_subset:list[int], 
        orders_data:OrdersData, 
        volume_dict:dict[int,float], 
        weight_dict:dict[int,float], 
        cage_data:CageData
    ) -> list[list[int]]:
    """Creates a set of synthetic orders based on user specifications and historical 
    product demands.

    Args:
        pick_data: Data frame containing pick data.
        prod_subset: Products to assign to slots.
        orders_data: Orders-related data.
        volume_dict: Product weights.
        weight_dict: Product volumes.
        cage_data: Cage-related data.

    Returns:
        Orders to be used in the optimisation.
    """
    
    if len(prod_subset) == 0:
        return [], {}
    
    product_demands_dict = (
        pick_data
        .groupby("tpnd")["qty_to_pick"]
        .sum()
        .reindex(prod_subset,fill_value=0)
        .to_dict()
    )

    for prod in prod_subset:
        if prod not in product_demands_dict.keys():
            product_demands_dict[prod] = 0

    orders = []

    while len(orders) < orders_data.num_orders:
        order_size = random.randint(orders_data.min_order_size,orders_data.max_order_size)

        order = random.choices(
            population = list(product_demands_dict.keys()),
            weights = list(product_demands_dict.values()),
            k=order_size
        )

        order_weight = sum([weight_dict[k] for k in order])
        order_volume = sum([volume_dict[k] for k in order])

        if order_weight < cage_data.cage_weight_capacity*cage_data.fill_percent and order_volume < cage_data.cage_volume_capacity*cage_data.fill_percent:
            orders.append(order)

    # Products ordered from highest to lowest demand
    prods_by_demand = sorted(
        product_demands_dict,
        key=product_demands_dict.get,
        reverse=True
    )

    return orders, prods_by_demand


def create_max_batches(
    weights_dict:dict[int,float], 
    volumes_dict:dict[int,float], 
    orders:list[list[int]], 
    cage_data:CageData, 
    buffer:float=0.25, 
    cages_per_batch:int=5
) -> int:
    """Finds a maximum number of batches the model is permitted to use.

    Args:
        weights_dict: Product weights.
        volumes_dict: Product volumes.
        orders: Orders generated for the instance.
        cage_data: Cage-related data.
        buffer: Percentage above minimum number of cages offered.
        cages_per_batch: Number of cages comprising each batch.
    
    Returns:
        The maximum allowed number of batches.
    """

    cage_weight_capacity = cage_weight_capacity*cage_data.fill_percent
    cage_volume_capacity = cage_volume_capacity*cage_data.fill_percent

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

