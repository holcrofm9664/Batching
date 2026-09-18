from dataclasses import dataclass
import pandas as pd

@dataclass
class DataFrames:
    """Data frames containing assignments and pick data.

    Attributes:
        pick_data: Dataframe containing the pick data, order quantities.
        solution_allocation: Dataframe containg current assignments, product 
            weights and volumes.
    """
    pick_data:pd.DataFrame
    solution_allocation:pd.DataFrame

@dataclass
class WarehouseData:
    """Warehouse-related input data.

    Attributes:
        num_aisles: Number of aisles in the warehouse
        num_bays: Number of bays in the warehouse
        slot_capacity: Number of unique products that can fit into each (aisle,bay) pair
        between_aisle_dist: Distance between consecutive aisles
        between_bay_dist: Distance between consecutive bays
    """
    num_aisles:int
    num_bays:int
    slot_capacity:int
    between_aisle_dist:int
    between_bay_dist:int

@dataclass
class OrdersData:
    """Orders-related data.

    Attributes:
        num_orders: Number of orders.
        min_order_size: Minimum number of units per order.
        max_order_size: Maximum number of units per order
    """
    num_orders:int
    min_order_size:int
    max_order_size:int


@dataclass
class CageData:
    """Cage-related data.
    
    Attributes:
        cage_weight_capacity: Weight capacity of each cage.
        cage_volume_capacity: Volume capacity of each cage.
        fill_frac: the liquid-fill assumption.
    """
    cage_weight_capacity:float
    cage_volume_capacity:float
    fill_frac:float