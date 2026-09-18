import pandas as pd
import numpy as np
import random
import math
from ast import literal_eval
from slap.comp_tests.dataclasses import (
    DataFrames,
    OrdersData,
    CageData,
    WarehouseData
)

def clean_dataframes(
    data_frames:DataFrames
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans the dataframes by dropping NA values and renaming columns.

    Args:
        data_frames: Data frames containing assignments and pick data.

    Returns:
        The cleaned data frames.
    """

    column_headers_dict = {
        "tpnd":"product",
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
    # extract the dataframes, so we don't modify the dataclass
    pick_data, solution_allocation = data_frames.pick_data, data_frames.solution_allocation

    # ensure slot_pair is a tuple
    solution_allocation["Slot_Pair"] = (
        data_frames.solution_allocation["Slot_Pair"].
        apply(literal_eval)
    )

    # rename columns in both dataframes
    pick_data = (
        pick_data
        .rename(columns = column_headers_dict)
    )

    solution_allocation = (
        solution_allocation
        .rename(columns = column_headers_dict)
    )

    # drop NA values from the solution_allocation dataframe
    solution_allocation = solution_allocation.dropna()

    return pick_data, solution_allocation


def sample_products(
    num_prods:int, 
    data_frames:DataFrames
) -> list[int]:
    """.Samples products to be kept in the instance.

    Args:
        num_prods: the number of products to include in the instance
        data_frames: data frames containing assignments and pick data.

    Returns:
        All the products and the products that have been sampled.
    """

    pick_data, solution_allocation = data_frames.pick_data, data_frames.solution_allocation

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

    all_prods = list(int(x) for x in all_prods)
    sampled_prods = list(int(x) for x in sampled_prods)

    return all_prods, sampled_prods


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
    cage_data:CageData,
) -> list[list[int]]:
    """Creates a set of synthetic orders.

    Args:
        pick_data: Data frame with pick data.
        prod_subset: Products used in the instance.
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
        .groupby("product")["qty_to_pick"]
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

        if (
            order_weight < cage_data.cage_weight_capacity*cage_data.fill_percent 
            and order_volume < cage_data.cage_volume_capacity*cage_data.fill_percent
        ):
            orders.append(order)

    return orders, product_demands_dict


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


def create_aisle_assignments(
    solution_allocation:pd.DataFrame,
    warehouse_data:WarehouseData,
    prod_subset:list[int], 
    num_zones:int,  
    product_demands_dict:dict[int,int] 
) -> tuple[dict[int,list[int]],dict[int,list[int]]]:
    """Creates a set of synthetic assignments.

    Args:
        solution_allocation: Data frame containing assignments and weights.
        warehouse_data: Warehouse-related data.
        prod_subset: Products stored in the warehouse.
        num_zones: Number of zones.
        product_demands_dict: Demands of each product.

    Returns:
        The aisle assignments.
    """

    aisles = list(range(warehouse_data.num_aisles))
    zones = np.array_split(aisles, num_zones)

    aisle_capacity = warehouse_data.num_bays*warehouse_data.slot_capacity

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