"""
🌙 Moon Dev's HyperLiquid Trading Functions
Focused functions for HyperLiquid perps trading
Built with love by Moon Dev 🚀
"""

# Standard library imports
import datetime
import json
import os
import time
import warnings

# Third-party imports
import eth_account
import pandas as pd
import requests

# Third-party from imports
from dotenv import load_dotenv
from eth_account.signers.local import LocalAccount
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from termcolor import colored

# Load environment variables
load_dotenv()

# Hide all warnings
warnings.filterwarnings('ignore')

def ask_bid(symbol):
    """Get ask and bid prices for a symbol"""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}

    data = {
        'type': 'l2Book',
        'coin': symbol
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    l2_data = response.json()
    l2_data = l2_data['levels']

    # get bid and ask
    bid = float(l2_data[0][0]['px'])
    ask = float(l2_data[1][0]['px'])

    return ask, bid, l2_data

def get_sz_px_decimals(symbol):
    """Get size and price decimals for a symbol"""
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': 'application/json'}
    data = {'type': 'meta'}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == symbol), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('Symbol not found')
            return 0, 0
    else:
        print('Error:', response.status_code)
        return 0, 0

    ask = ask_bid(symbol)[0]
    ask_str = str(ask)

    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0

    print(f'{symbol} price: {ask} | sz decimals: {sz_decimals} | px decimals: {px_decimals}')
    return sz_decimals, px_decimals

def get_position(symbol, account):
    """Get current position for a symbol"""
    print(f'{colored("Getting position for", "cyan")} {colored(symbol, "yellow")}')

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    positions = []
    for position in user_state["assetPositions"]:
        if position["position"]["coin"] == symbol and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            coin = position["position"]["coin"]
            pos_size = float(position["position"]["szi"])
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f'{colored(f"{coin} position:", "green")} Size: {pos_size} | Entry: ${entry_px} | PnL: {pnl_perc:.2f}%')

    im_in_pos = len(positions) > 0

    if not im_in_pos:
        print(f'{colored("No position in", "yellow")} {symbol}')
        return positions, im_in_pos, 0, symbol, 0, 0, True

    # Return position details
    pos_size = positions[0]["szi"]
    pos_sym = positions[0]["coin"]
    entry_px = float(positions[0]["entryPx"])
    pnl_perc = float(positions[0]["returnOnEquity"]) * 100
    is_long = float(pos_size) > 0

    if is_long:
        print(f'{colored("LONG", "green")} position')
    else:
        print(f'{colored("SHORT", "red")} position')

    return positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long

def set_leverage(symbol, leverage, account):
    """Set leverage for a symbol"""
    print(f'Setting leverage for {symbol} to {leverage}x')
    exchange = Exchange(account, constants.MAINNET_API_URL)

    # Update leverage (is_cross=True for cross margin)
    result = exchange.update_leverage(leverage, symbol, is_cross=True)
    print(f'✅ Leverage set to {leverage}x for {symbol}')
    return result

def adjust_leverage_usd_size(symbol, usd_size, leverage, account):
    """Adjust leverage and calculate position size"""
    print(f'Adjusting leverage for {symbol} to {leverage}x with ${usd_size} size')

    # Set the leverage
    set_leverage(symbol, leverage, account)

    # Get current price
    ask, bid, _ = ask_bid(symbol)
    mid_price = (ask + bid) / 2

    # Calculate position size in coins
    pos_size = usd_size / mid_price

    # Get decimals for rounding
    sz_decimals, _ = get_sz_px_decimals(symbol)
    pos_size = round(pos_size, sz_decimals)

    print(f'Position size: {pos_size} {symbol} (${usd_size} at ${mid_price:.2f})')

    return leverage, pos_size

def cancel_all_orders(account):
    """Cancel all open orders"""
    print(colored('🚫 Cancelling all orders', 'yellow'))
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    # Get all open orders
    open_orders = info.open_orders(account.address)

    if not open_orders:
        print(colored('   No open orders to cancel', 'yellow'))
        return

    # Cancel each order
    for order in open_orders:
        try:
            exchange.cancel(order['coin'], order['oid'])
            print(colored(f'   ✅ Cancelled {order["coin"]} order', 'green'))
        except Exception as e:
            print(colored(f'   ⚠️ Could not cancel {order["coin"]} order: {str(e)}', 'yellow'))

    print(colored('✅ All orders cancelled', 'green'))
    return

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """Place a limit order"""
    exchange = Exchange(account, constants.MAINNET_API_URL)

    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)

    print(f"🌙 Moon Dev placing order:")
    print(f"Symbol: {coin}")
    print(f"Side: {'BUY' if is_buy else 'SELL'}")
    print(f"Size: {sz}")
    print(f"Price: ${limit_px}")
    print(f"Reduce Only: {reduce_only}")

    order_result = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)

    if isinstance(order_result, dict) and 'response' in order_result:
        print(f"✅ Order placed with status: {order_result['response']['data']['statuses'][0]}")
    else:
        print(f"✅ Order placed")

    return order_result

def kill_switch(symbol, account):
    """Close position at market price"""
    print(colored(f'🔪 KILL SWITCH ACTIVATED for {symbol}', 'red', attrs=['bold']))

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    exchange = Exchange(account, constants.MAINNET_API_URL)

    # Get current position
    positions, im_in_pos, pos_size, _, _, _, is_long = get_position(symbol, account)

    if not im_in_pos:
        print(colored('No position to close', 'yellow'))
        return

    # Place market order to close
    side = not is_long  # Opposite side to close
    abs_size = abs(float(pos_size))

    print(f'Closing {"LONG" if is_long else "SHORT"} position: {abs_size} {symbol}')

    # Get current price for market order
    ask, bid, _ = ask_bid(symbol)

    # For closing positions with IOC orders:
    # - Closing long: Sell below bid (undersell)
    # - Closing short: Buy above ask (overbid)
    if is_long:
        close_price = bid * 0.999  # Undersell to close long
    else:
        close_price = ask * 1.001  # Overbid to close short

    # Round to appropriate decimals for BTC
    if symbol == 'BTC':
        close_price = round(close_price)
    else:
        close_price = round(close_price, 1)

    print(f'   Placing IOC at ${close_price} to close position')

    # Place reduce-only order to close
    order_result = exchange.order(symbol, side, abs_size, close_price, {"limit": {"tif": "Ioc"}}, reduce_only=True)

    print(colored('✅ Kill switch executed - position closed', 'green'))
    return order_result

def pnl_close(symbol, target, max_loss, account):
    """Close position if PnL target or stop loss is hit"""
    print(f'{colored("Checking PnL conditions", "cyan")}')
    print(f'Target: {target}% | Stop loss: {max_loss}%')

    # Get current position info
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, is_long = get_position(symbol, account)

    if not im_in_pos:
        print(colored('No position to check', 'yellow'))
        return False

    print(f'Current PnL: {colored(f"{pnl_perc:.2f}%", "green" if pnl_perc > 0 else "red")}')

    # Check if we should close
    if pnl_perc >= target:
        print(colored(f'✅ Target reached! Closing position WIN at {pnl_perc:.2f}%', 'green', attrs=['bold']))
        kill_switch(symbol, account)
        return True
    elif pnl_perc <= max_loss:
        print(colored(f'🛑 Stop loss hit! Closing position LOSS at {pnl_perc:.2f}%', 'red', attrs=['bold']))
        kill_switch(symbol, account)
        return True
    else:
        print(f'Position still open. PnL: {pnl_perc:.2f}%')
        return False

def get_current_price(symbol):
    """Get current price for a symbol"""
    ask, bid, _ = ask_bid(symbol)
    mid_price = (ask + bid) / 2
    return mid_price

def get_account_value(account):
    """Get total account value"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    account_value = float(user_state["marginSummary"]["accountValue"])
    print(f'Account value: ${account_value:,.2f}')
    return account_value

def market_buy(symbol, usd_size, account):
    """Market buy using HyperLiquid"""
    print(colored(f'🛒 Market BUY {symbol} for ${usd_size}', 'green'))

    # Get current ask price
    ask, bid, _ = ask_bid(symbol)

    # Overbid by 0.1% to ensure fill (market buy needs to be above ask)
    buy_price = ask * 1.001

    # Round to appropriate decimals for BTC (whole numbers)
    if symbol == 'BTC':
        buy_price = round(buy_price)
    else:
        buy_price = round(buy_price, 1)

    # Calculate position size
    pos_size = usd_size / buy_price

    # Get decimals and round
    sz_decimals, _ = get_sz_px_decimals(symbol)
    pos_size = round(pos_size, sz_decimals)

    # Ensure minimum order value
    order_value = pos_size * buy_price
    if order_value < 10:
        print(f'   ⚠️ Order value ${order_value:.2f} below $10 minimum, adjusting...')
        pos_size = 11 / buy_price  # $11 to have buffer
        pos_size = round(pos_size, sz_decimals)

    print(f'   Placing IOC buy at ${buy_price} (0.1% above ask ${ask})')
    print(f'   Position size: {pos_size} {symbol} (value: ${pos_size * buy_price:.2f})')

    # Place IOC order above ask to ensure fill
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order_result = exchange.order(symbol, True, pos_size, buy_price, {"limit": {"tif": "Ioc"}}, reduce_only=False)

    print(colored(f'✅ Market buy executed: {pos_size} {symbol} at ${buy_price}', 'green'))
    return order_result

def market_sell(symbol, usd_size, account):
    """Market sell using HyperLiquid"""
    print(colored(f'💸 Market SELL {symbol} for ${usd_size}', 'red'))

    # Get current bid price
    ask, bid, _ = ask_bid(symbol)

    # Undersell by 0.1% to ensure fill (market sell needs to be below bid)
    sell_price = bid * 0.999

    # Round to appropriate decimals for BTC (whole numbers)
    if symbol == 'BTC':
        sell_price = round(sell_price)
    else:
        sell_price = round(sell_price, 1)

    # Calculate position size
    pos_size = usd_size / sell_price

    # Get decimals and round
    sz_decimals, _ = get_sz_px_decimals(symbol)
    pos_size = round(pos_size, sz_decimals)

    # Ensure minimum order value
    order_value = pos_size * sell_price
    if order_value < 10:
        print(f'   ⚠️ Order value ${order_value:.2f} below $10 minimum, adjusting...')
        pos_size = 11 / sell_price  # $11 to have buffer
        pos_size = round(pos_size, sz_decimals)

    print(f'   Placing IOC sell at ${sell_price} (0.1% below bid ${bid})')
    print(f'   Position size: {pos_size} {symbol} (value: ${pos_size * sell_price:.2f})')

    # Place IOC order below bid to ensure fill
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order_result = exchange.order(symbol, False, pos_size, sell_price, {"limit": {"tif": "Ioc"}}, reduce_only=False)

    print(colored(f'✅ Market sell executed: {pos_size} {symbol} at ${sell_price}', 'red'))
    return order_result

def close_position(symbol, account):
    """Close any open position for a symbol"""
    positions, im_in_pos, pos_size, _, _, pnl_perc, is_long = get_position(symbol, account)

    if not im_in_pos:
        print(f'No position to close for {symbol}')
        return None

    print(f'Closing {"LONG" if is_long else "SHORT"} position with PnL: {pnl_perc:.2f}%')
    return kill_switch(symbol, account)

# Additional helper functions for agents
def get_balance(account):
    """Get USDC balance"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    # Get withdrawable balance (free balance)
    balance = float(user_state["withdrawable"])
    print(f'Available balance: ${balance:,.2f}')
    return balance

def get_all_positions(account):
    """Get all open positions"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)

    positions = []
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            positions.append({
                'symbol': position["position"]["coin"],
                'size': float(position["position"]["szi"]),
                'entry_price': float(position["position"]["entryPx"]),
                'pnl_percent': float(position["position"]["returnOnEquity"]) * 100,
                'is_long': float(position["position"]["szi"]) > 0
            })

    return positions