from ib_insync import *
from os import listdir, remove
from time import sleep
import pandas as pd
import pickle

# Define your variables here ###########################################################################################
sampling_rate = 1 # How often, in seconds, to check for inputs from Dash?
# For TWS Paper account, default port is 7497
# For IBG Paper account, default port is 4002
port = 7497
# choose your master id. Mine is 10645. You can use whatever you want, just set it in API Settings within TWS or IBG.
master_client_id = 10645
# choose your dedicated id just for orders. I picked 1111.
orders_client_id = 1111
# account number: you'll need to fill in yourself. The below is one of my paper trader account numbers.
acc_number = 'DU3526622'
########################################################################################################################

# Create an IB app; i.e., an instance of the IB() class from the ib_insync package
ib = IB()
# Connect your app to a running instance of IBG or TWS
ib.connect(host='127.0.0.1', port=port, clientId=master_client_id)

# Make sure you're connected -- stay in this while loop until ib.isConnected() is True.
while not ib.isConnected():
    sleep(.01)

# If connected, script proceeds and prints a success message.
print('Connection Successful!')

# Main while loop of the app. Stay in this loop until the app is stopped by the user.
while True:
    # If the app finds a file named 'currency_pair.txt' in the current directory, enter this code block.

    # If there's a file named trade_order.p in listdir(), then enter the loop below.
    if 'trade_order.p' in listdir():

        # Create a special instance of IB() JUST for entering orders.
        # The reason for this is because the way that Interactive Brokers automatically provides valid order IDs to
        #   ib_insync is not trustworthy enough to really rely on. It's best practice to set aside a dedicated client ID
        #   to ONLY be used for submitting orders, and close the connection when the order is successfully submitted.

        # your code goes here
        f1 = open("trade_order.p", 'rb')
        trd_ordr = pickle.load(f1)
        contract = Stock(trd_ordr['trade_currency'],"SMART","USD")
        if trd_ordr['action'] == 'BUY':
            order = MarketOrder(trd_ordr['action'],trd_ordr['trade_amt'])
            order.account = acc_number
        else:
            order = LimitOrder(trd_ordr['action'],trd_ordr['trade_amt'],trd_ordr['price'])
            order.account = acc_number
        ib_orders = IB()
        ib_orders.connect(host='127.0.0.1', port=port, clientId=orders_client_id)
        new_order = ib_orders.placeOrder(contract, order)
        # The new_order object returned by the call to ib_orders.placeOrder() that you've written is an object of class
        #   `trade` that is kept continually updated by the `ib_insync` machinery. It's a market order; as such, it will
        #   be filled immediately.
        # In this while loop, we wait for confirmation that new_order filled.
        while not new_order.orderStatus.status == 'Filled':
            ib_orders.sleep(0) # we use ib_orders.sleep(0) from the ib_insync module because the async socket connection
                               # is not built for the normal time.sleep() function.

        # your code goes here
        f1.close()
        remove('trade_order.p')
        # pass: same reason as above.
        pass

    # sleep, for the while loop.
    ib.sleep(sampling_rate)
