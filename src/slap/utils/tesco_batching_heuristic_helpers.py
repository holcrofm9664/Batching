from numpy import floor
from typing import Tuple

def quantity_that_fits(product:int, demand:int, weights_dict:dict[int,float], vol_dict:dict[int,float], cage_weight_capacity:float, cage_vol_capacity:float, c_weight:float, c_vol:float) -> int:
    """
    Checks the quantity of a product that can fit into an already part-filled cage

    Inputs:
    - product: the product whose units we are aiming to fit into the cage
    - demand: the total number we wish to fit in
    - weights_dict: the dictionary of product weights
    - vol_dict: the dictionary of product volumes
    - cage_weight_capacity: the total weight capacity of the cage
    - cage_vol_capacity: the total volume capacity of the cage
    - c_weight: the used weight capacith from already-assigned products
    - c_vol: the used volume capacity from already assigned products
    
    Outputs:
    - quantity: the quantity of the product which will fit into the cage
    """

    weight, volume = weights_dict[product], vol_dict[product]

    remaining_weight, remaining_volume = cage_weight_capacity - c_weight, cage_vol_capacity - c_vol

    # the quantity that can fit based on weight
    by_weight = demand if weight == 0 else floor(remaining_weight/weight)

    # the quantity that can fit based on volume
    by_volume = demand if volume == 0 else floor(remaining_volume/volume)

    quantity = int(max(0,min(demand, by_weight, by_volume)))

    return quantity


def add_product(product:int, quantity:int, shop_dem:int, c_weight:float, c_vol:float, trips:dict[int,dict[int,list[int]]], trip:int, cage:int, weights_dict:dict[int,float], vol_dict:dict[int,float]) -> Tuple[float,float]:
    """
    Adds a pre-determined quantity of a product to a cage

    Inputs:
    - product: the product to be added
    - quantity: the number of units of that product which are to be added
    - shop_dem: the quantity of that product demanded by the shop
    - c_weight: the used weight capacith from already-assigned products
    - c_vol: the used volume capacity from already assigned products
    - trips: the current trips dictionary
    - trip: the trip number
    - cage: the cage number
    - weights_dict: the dictionary of product weights
    - vol_dict: the dictionary of product volumes
    
    Outputs:
    - c_weight: the updated weight capacity used by the cage from assigned-products
    - c_vol: the updated volume capacity used by the cage from assigned products
    """
    
    if quantity == 0:
        return c_weight, c_vol

    # assign the quantity of product
    trips[trip][cage].extend([product] * quantity)

    # update the weights and volumes
    c_weight += weights_dict[product] * quantity
    c_vol += vol_dict[product] * quantity

    # deduct the assigned quantity from remaining demand
    shop_dem[product] -= quantity

    # remove demand altogether if all products have been assigned
    if shop_dem[product] == 0:
        del shop_dem[product]

    return c_weight, c_vol


def open_new_cage(trip:int, cage:int, c_weight:float, c_vol:float, trips:dict[int,dict[int,list[int]]], cages_per_trip:int) -> Tuple[int,int,float,float]:
    """
    Opens a new cage and updates relevant outputs

    Inputs:
    - trip: the trip number
    - cage: the cage number
    - c_weight: the used weight capacith from already-assigned products
    - c_vol: the used volume capacity from already assigned products
    - trips: the current trips dictionary

    Outputs:
    - trip: the new trip number
    - cage: the new cage number
    - c_weight: the new used cage capacity (0, as the cage will be empty)
    - c_vol: the new used cage volume (0, as the cage will be empty)
    """
    
    
    if cage < cages_per_trip:
        cage += 1
    else:
        trip += 1
        cage = 1
        trips[trip] = {}

    trips[trip][cage] = []

    c_weight, c_vol = 0.0, 0.0

    return trip, cage, c_weight, c_vol