import pandas as pd
import numpy as np
import random
import math
from slap.utils.dataclasses import OrdersData, CageData, WarehouseData

def clean_dataframes(
    pick_data:pd.DataFrame,
    assignments:pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans the dataframes by dropping NA values and renaming columns.

    Args:
        pick_data: Dataframe containing orders data.
        assignments: Dataframe containing storage assignments.

    Returns:
        The cleaned data frames.
    """

    column_headers_dict = {
        "tpnd":"tpnd",
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
    pick_data = (
        pick_data
        .rename(columns = column_headers_dict)
    )

    assignments = (
        assignments
        .rename(columns = column_headers_dict)
    )

    # drop NA values from the assignments dataframe
    assignments = assignments.dropna()

    return pick_data, assignments

def weights_and_volumes(
    assignments:pd.DataFrame, 
    prod_subset:list[int]
) -> tuple[dict[int,float], dict[int,float]]:
    """Constructs a weights dictionary and a volumes dictionary from the dataframe.

    Args:
        assignments: Dataframe containing the weights and volumes.
        prod_subset: Products remaining after we have filtered for time and a chosen 
            product subset.

    Returns:
        The dictionaries of weights and volumes.
    """

    # construct the weights and volumes dictionaries
    df = (
        assignments[
            assignments["tpnd"].isin(prod_subset)
        ]
        .drop_duplicates(subset="tpnd")
        .set_index("tpnd")
    )

    weight_dict = df["weight"].to_dict()
    volume_dict = df["volume"].to_dict()
    
    return weight_dict, volume_dict


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
        orders: Orders used for the instance.
        cage_data: Cage-related data.
        buffer: Percentage above minimum number of cages offered.
        cages_per_batch: Number of cages comprising each batch.
    
    Returns:
        The maximum allowed number of batches.
    """

    num_cages, current_weight, current_volume, current_orders = 1, 0, 0, 0

    for order in orders.values(): 

        current_orders += 1

        for prod in order:
            w, v = weights_dict[prod], volumes_dict[prod]
            if (
                current_weight + w <= cage_data.cage_weight_capacity
                and current_volume + v <= cage_data.cage_volume_capacity
                and current_orders <= cage_data.max_orders_per_cage
            ):
                current_weight += w
                current_volume += v
            else:
                num_cages += 1
                current_weight = w
                current_volume = v
                current_orders = 1

    # offer more cages than required - the model may find a better solution with more than the minimum
    max_batches = math.floor((1+buffer)*math.ceil(num_cages / cages_per_batch) + 1)

    return max_batches

def sample_orders(
    pick_data:pd.DataFrame,
    num_samples:int,
    seed:int
) -> dict[tuple[str,int],list[int]] :
    """Samples orders to be used in the instance.

    Args:
        pick_data: Dataframe containing orders data.
        num_samples: Number of orders sampled.
        seed: Seed for controlling randomness.

    Returns:
        Sampled orders dictionary.
    """
    
    groups_df = pick_data.groupby(
        ["time","store_id"],
        as_index=False
    ).agg(
        qty = ("qty_to_pick",sum)
    )

    orders_dict = {(t,s):q for t,s,q in zip(groups_df["time"].to_list(),groups_df["store_id"].to_list(),groups_df["qty"].to_list())}

    random.seed(seed)

    sample = random.sample(
        [k for k in orders_dict.keys()],
        counts = orders_dict.values(),
        k = num_samples
    )

    orders_dict = {}

    for s in sample:

        # the dataframe for just that order
        new_df = pick_data[(pick_data["time"]==s[0]) & (pick_data["store_id"] == s[1])]

        prods = new_df["tpnd"].to_list()
        qtys = new_df["qty_to_pick"].to_list()

        qty_dict = {k:q for k,q in zip(prods,qtys)}

        for k in prods:
            orders_dict.setdefault((s[0],s[1]),[]).extend([k for _ in range(qty_dict[k])])

    return orders_dict

def get_prod_assignments(
        assignments:pd.DataFrame
) -> dict[int,tuple[int,int]]:
    """Gets product assignments from assignments data frame, for single-placement.

    Args:
        assignments: The Assignments data frame.

    Returns:
        A dictionary containing products and their respective slots.
    """

    assignments = assignments.dropna(subset="tpnd")
    prod_assignments = {
        k:(a,b) 
        for k,a,b in zip(
            assignments["tpnd"].to_list(),
            assignments["aisle"].to_list(),
            assignments["bay"].to_list()
        )
    }
    return prod_assignments

def merge_and_clean_dfs(
        solution_allocation:pd.DataFrame,
        locations:pd.DataFrame
) -> pd.DataFrame:
    """Merges and filters the key dataframes to create the assignments dataframe.

    Args:
        solution_allocation: Data frame containing assignments.
        locations: Data frame containing location information.

    Returns:
        The assignments and single assignments dataframes.
    """

    # rename columns
    locations = locations.rename(
        columns={
            "Name":"location",
            "Aisle":"aisle"
        }
    )

    # merge the dataframes
    assignments = solution_allocation.merge(locations,how="outer",on="location")

    # retrieve the aisles
    assignments["aisle"] = np.ceil(
        assignments["MapCol"]/3
    ).astype(int)
    assignments["aisle"] = assignments["aisle"]-1

    # retrieve the bays
    assignments["bay"] = 62 - assignments["MapRow"]

    # calculate the volumes
    assignments["volume"] = assignments["weight"]/assignments["mass_factor"]

    # extract product numbers
    assignments["tpnd"] = assignments["tpnd"].str.replace(
        r"\D",
        "",
        regex=True
    )

    # reorder by location
    assignments=assignments.sort_values(
        by="location",
        key = lambda col:col.str.replace(
            r"\D",
            "",
            regex=True
        ).astype(int)
    )

    # reduce to relevant columns
    assignments = assignments[
        [
            "location",
            "tpnd",
            "aisle",
            "bay",
            "weight",
            "volume"
        ]
    ]

    single_assignments = assignments.drop_duplicates(
        subset="tpnd",
        keep="first"
    )

    return assignments, single_assignments

# -----------------------------------------------------------------------------------
# helpers specifically for synthetic instances
# -----------------------------------------------------------------------------------

def sample_products(
    num_prods:int, 
    pick_data:pd.DataFrame,
    assignments:pd.DataFrame
) -> list[int]:
    """Samples products to be kept in the instance.

    Args:
        num_prods: the number of products to include in the instance
        pick_data: Dataframe containing orders data.
        assignments: Dataframe containing storage assignments.

    Returns:
        All the products and the products that have been sampled.
    """

    all_prods = assignments["tpnd"].unique()
    
    unit_qty_dict = dict(pick_data.groupby("tpnd")["qty_to_pick"].sum())

    if num_prods >= len(unit_qty_dict):
        print(f"Too many products requested. Returning all products")
        return list(int(x) for x in all_prods), list(int(x) for x in all_prods)


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


def create_orders(
    pick_data:pd.DataFrame, 
    prod_subset:list[int], 
    orders_data:OrdersData
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

        orders.append(order)

    return orders, product_demands_dict


def create_aisle_assignments(
    assignments:pd.DataFrame,
    warehouse_data:WarehouseData,
    prod_subset:list[int], 
    num_zones:int,  
    product_demands_dict:dict[int,int] 
) -> tuple[dict[int,list[int]],dict[int,list[int]]]:
    """Creates a set of synthetic assignments.

    Args:
        assignments: Data frame containing assignments and weights.
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
        assignments
        .set_index("tpnd")["weight"]
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