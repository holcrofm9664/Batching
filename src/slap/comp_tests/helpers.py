import random
import math
import pandas as pd

def weights_and_volumes(
    solution_allocation:pd.DataFrame, 
    prod_subset:list[int]
) -> tuple[dict[int,float], dict[int,float]]:
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
    df = (
        solution_allocation[
            solution_allocation["tpnd"].isin(prod_subset)
        ]
        .drop_duplicates(subset="tpnd")
        .set_index("tpnd")
    )

    weight_dict = df["weight"].to_dict()
    volume_dict = df["volume"].to_dict()
    
    return volume_dict, weight_dict

def create_orders(
        pick_data:pd.DataFrame, 
        prod_subset:list[int], 
        num_orders:int, 
        min_order_size:int, 
        max_order_size:int, 
        volume_dict:dict[int,float], 
        weight_dict:dict[int,float], 
        cage_weight_capacity:int=400, 
        cage_volume_capacity:int=45, 
        fill_percent:float=0.85
    ) -> list[list[int]]:
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
    cage_weight_capacity:int=400, 
    cage_volume_capacity:int=45, 
    fill_percent:float=0.85, 
    buffer:float=0.25, 
    cages_per_batch:int=5
) -> int:
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

    cage_weight_capacity = cage_weight_capacity*fill_percent
    cage_volume_capacity = cage_volume_capacity*fill_percent

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

