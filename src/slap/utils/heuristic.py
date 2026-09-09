from collections import defaultdict
from src.slap.utils.heuristic_helpers import quantity_that_fits, add_product, open_new_cage

def create_batches(
    shop_prod_dem:dict[tuple[int,int],int],
    stored_product:dict[str,tuple[int,int]],
    weights_dict:dict[int,float],
    vol_dict:dict[int,float],
    num_aisles:int,
    num_bays:int,
    cage_weight_capacity:float,
    cage_vol_capacity:float,
    fill_percent:float=0.85,
    cages_per_trip:int=5,
    look_ahead:int=75,
) -> dict[int:dict[int,list[int]]]:
    """
    Create a set of batches from a set of orders corresponding to a subset of shops and products, 
    using Tesco's current batching heuristic.

    Inputs:
    - shop_prod_dem: the demand for each shop-product pair
    - stored_product: the products stored in each (aisle,bay) pair
    - weights_dict: the product weights dictionary
    - vol_dict: the product volumes dictionary
    - num_aisles: the number of aisles
    - num_bays: the number of bays
    - cage_weight_capacity: the cage weight capacity
    - cage_vol_capacity: the cage volume capacity
    - cages_per_trip: the number of cages in each trip
    - look_ahead: the number of slots look-ahead when one item cannot fit into a cage

    Outputs:
    - trips: the trips dictionary
    """

    # adjust capacities by fill-percent assumption
    cage_weight_capacity, cage_vol_capacity = fill_percent*cage_weight_capacity, fill_percent*cage_vol_capacity
    
    # create the slot ordering
    slots = []

    for aisle in range(num_aisles):

        bays = (
            range(num_bays)
            if aisle % 2 == 0
            else reversed(range(num_bays))
        )

        slots.extend((aisle, bay) for bay in bays)

    # ------------------------------------------------------------
    # 2. Check storage locations and identify stored products
    # ------------------------------------------------------------

    stored_skus = set()

    for location in slots:
        if location in stored_product:
            products = tuple(stored_product[location])
            stored_skus.update(products)

    # ------------------------------------------------------------
    # 3. Convert demand to {shop: {product: demand}}
    # ------------------------------------------------------------

    demand_by_shop = defaultdict(dict)

    for (shop, product), demand in shop_prod_dem.items():

        if not isinstance(demand, int) or isinstance(demand, bool) or demand < 0:
            raise ValueError(
                f"Demand for shop {shop}, product {product} "
                f"must be a non-negative integer."
            )

        if demand == 0:
            continue

        if product not in stored_skus:
            raise ValueError(
                f"Product {product}, demanded by shop {shop}, "
                "is not stored in the warehouse."
            )

        if product not in weights_dict:
            raise KeyError(f"No weight given for product {product}.")

        if product not in vol_dict:
            raise KeyError(f"No volume given for product {product}.")

        if (
            weights_dict[product] > cage_weight_capacity
            or vol_dict[product] > cage_vol_capacity
        ):
            raise ValueError(
                f"One unit of product {product} cannot fit "
                "in an empty cage."
            )

        demand_by_shop[shop][product] = demand

    if not demand_by_shop:
        return {}
    
    # ------------------------------------------------------------
    # 4. Initialise first trip/cage
    # ------------------------------------------------------------

    trip, cage = 1, 1

    trips = {trip: {cage: []}}

    c_weight, c_vol = 0.0, 0.0

    # ------------------------------------------------------------
    # 5. Process each shop in turn
    # ------------------------------------------------------------

    for shop in sorted(demand_by_shop):

        shop_dem = demand_by_shop[shop]

        slot = 0

        while shop_dem:

            if slot >= len(slots):
                raise RuntimeError(
                    f"Could not satisfy all demand for shop {shop}. "
                    f"Remaining demand: {shop_dem}"
                )

            location = slots[slot]
            if location in stored_product:
                products = tuple(stored_product[location])

                blocked = False

                # consider both products in the slot
                for product in products:
                    if product not in shop_dem:
                        continue

                    quantity = quantity_that_fits(
                        product=product,
                        demand=shop_dem[product],
                        weights_dict=weights_dict,
                        vol_dict=vol_dict,
                        cage_weight_capacity=cage_weight_capacity,
                        cage_vol_capacity=cage_vol_capacity,
                        c_weight=c_weight,
                        c_vol=c_vol
                    )

                    c_weight, c_vol = add_product(product=product,
                                                  quantity=quantity,
                                                  shop_dem=shop_dem,
                                                  c_weight=c_weight,
                                                  c_vol=c_vol,
                                                  trips=trips,
                                                  trip=trip,
                                                  cage=cage,
                                                  weights_dict=weights_dict,
                                                  vol_dict=vol_dict,
                                      )

                    # if a product is still in shop_dem then we couldn't fit all of it in
                    if product in shop_dem:
                        blocked = True

                # if we fit everything from this slot in
                if not blocked:

                    slot += 1
                    continue

                # the look-ahead aspect
                stop = min(slot + 1 + look_ahead, len(slots))

                for next_slot in range(slot + 1, stop):

                    next_location = slots[next_slot]
                    if next_location in stored_product:
                        next_products = tuple(stored_product[next_location])

                        # consider both products at each look-ahead slot.
                        for next_product in next_products:

                            if next_product not in shop_dem:
                                continue

                            quantity = quantity_that_fits(product=next_product,
                                                          demand=shop_dem[next_product],
                                                          weights_dict=weights_dict,
                                                          vol_dict=vol_dict,
                                                          cage_weight_capacity=cage_weight_capacity,
                                                          cage_vol_capacity=cage_vol_capacity,
                                                          c_weight=c_weight,
                                                          c_vol=c_vol
                                       )

                            c_weight, c_vol = add_product(product=product,
                                                          quantity=quantity,
                                                          shop_dem=shop_dem,
                                                          c_weight=c_weight,
                                                          c_vol=c_vol,
                                                          trips=trips,
                                                          trip=trip,
                                                          cage=cage,
                                                          weights_dict=weights_dict,
                                                          vol_dict=vol_dict,
                                                      )

                # this is a fail-safe. The blocking product should still remain, and thus shop_dem > 0
                if not shop_dem:
                    break
                
                # open a new cage and retry the original slot before the look-ahead
                trip, cage, c_weight, c_vol = open_new_cage(trip=trip,
                                                            cage=cage,
                                                            c_weight=c_weight,
                                                            c_vol=c_vol,
                                                            trips=trips,
                                                            cages_per_trip=cages_per_trip,
                                              )

            else:
                slot += 1

    return trips