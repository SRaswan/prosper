from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Dict, Any, List
import json
import collections
import copy

import numpy as np

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS': 0}


class Logger:

    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log self.POSITION_LIMIT[product]
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    position = copy.deepcopy(empty_dict)
    sf_cache = []
    ma_cache = 0
    sf_dim = 2

    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'STARFRUIT' : 100}
    
    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        _, best_sell_pr = self.values_extract(osell)
        _, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)
        pp = "poo"

        num = min(40, self.POSITION_LIMIT[product] - cpos)
        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num-1))
            cpos += num

        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] > 15):
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT[product]:
            orders.append(Order(product, bid_pr+2, 1))
            orders.append(Order(product, bid_pr, num-1))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        num = max(-40, -self.POSITION_LIMIT[product]-cpos)
        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] < -15):
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[product]:
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders, ""
    
    def calc_next_price_starfruit(self):
        ar_coef = -0.5
        diff = self.sf_cache[-1] - self.sf_cache[-2]
        return int(round((self.sf_cache[-1] + diff * ar_coef) * 2) / 2)
    
    """def calc_next_price_starfruit(self):
        ma_coef = -0.7096
        error = self.sf_cache[-1] - self.ma_cache
        return int(round((self.sf_cache[-1] + error * ma_coef) * 2) / 2)"""
    
    def compute_orders_starfruit(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < self.POSITION_LIMIT[product]:
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.POSITION_LIMIT[product]:
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product]-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        pp = ""
        if cpos > -self.POSITION_LIMIT[product]:
            num = max(-40, -self.POSITION_LIMIT[product]-cpos)
            orders.append(Order(product, sell_pr, num))
            pp = "selling"
            cpos += num

        return orders, pp

    def calc_next_price_orchids(self):
        ar_coef = -0.5
        diff = self.sf_cache[-1] - self.sf_cache[-2]
        return int(round((self.sf_cache[-1] + diff * ar_coef) * 2) / 2)

    def compute_orders_orchids(self, product, order_depth, acc_bid, acc_ask, conversionObservations):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        transportFee = conversionObservations.transportFees
        exportTar = conversionObservations.exportTariff
        importTar = conversionObservations.importTariff

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]
        duckcpos = self.position[product]

        mid_price['ORCHIDS'] = (best_sell_pr + best_buy_pr)/2

        res1 = mid_price['ORCHIDS']+transportFee+exportTar - mid_price['DUCKORCHIDS']
        res2 = mid_price['ORCHIDS'] - mid_price['DUCKORCHIDS']+transportFee+importTar

        if res > trade_at:
            vol = self.position['ORCHIDS'] + self.POSITION_LIMIT['ORCHIDS']
            if vol > 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', worst_buy['ORCHIDS'], -vol))
        elif res < -trade_at:
            vol = self.POSITION_LIMIT['ORCHIDS'] - self.position['ORCHIDS']
            if vol > 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', worst_sell['ORCHIDS'], vol))
        elif res < close_at and self.position['ORCHIDS'] < 0:
            vol = -self.position['ORCHIDS']
            if vol > 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', worst_sell['ORCHIDS'], vol))
        elif res > -close_at and self.position['ORCHIDS'] > 0:
            vol = self.position['ORCHIDS']
            if vol > 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', worst_buy['ORCHIDS'], -vol))

        return orders, conversions

        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""
        result = {}

        if len(self.sf_cache) == self.sf_dim:
            self.sf_cache.pop(0)

        _, bs_sf = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_sf = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)
        self.sf_cache.append((bs_sf + bb_sf) / 2)

        INF = 1e9
        sf_lb = -INF
        sf_ub = INF
        pp = ""


        acc_bid = {'AMETHYSTS' : 10_000, 'STARFRUIT' : sf_lb} # we want to buy at slightly below
        acc_ask = {'AMETHYSTS' : 10_000, 'STARFRUIT' : sf_ub} 
        
        for product in self.position.keys():
            self.position[product] = state.position.get(product, 0)

        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if (product == 'AMETHYSTS'):
                orders, pp = self.compute_orders_amethysts(product, order_depth, acc_bid[product], acc_ask[product])
            elif (product == 'STARFRUIT'):
                if len(self.sf_cache) == self.sf_dim:
                    next_price = self.calc_next_price_starfruit()
                    self.ma_cache = next_price
                    sf_lb = next_price - 1
                    sf_ub = next_price + 1
                    pp = next_price
                orders, pp = self.compute_orders_starfruit(product, order_depth, acc_bid[product], acc_ask[product])
            elif (product == 'ORCHIDS'):
                if len(self.sf_cache) == self.sf_dim:
                    next_price = self.calc_next_price_orchids() #DO THIS BRUH
                    self.ma_cache = next_price
                    sf_lb = next_price - 1
                    sf_ub = next_price + 1
                    pp = next_price
                orders, pp = self.compute_orders_orchids(product, order_depth, acc_bid[product], acc_ask[product], state.observations.conversionObservations)
            result[product] = orders
    
    
        traderData = str(pp) # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData
    