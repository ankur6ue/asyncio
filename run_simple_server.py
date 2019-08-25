#!/usr/bin/env python3

# Echo server. Will reverse everything we throw at it.

import asyncio
import websockets
import dill
from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import argparse
import signal

def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    logging.error(f"Caught exception: {msg}")
    logging.info("Shutting down...")
    asyncio.create_task(shutdown(loop))

async def shutdown(loop, signal=None):
    """Cleanup tasks tied to the service's shutdown."""
    """Cleanup tasks tied to the service's shutdown."""
    if signal:
        logging.info(f"Received exit signal {signal.name}...")
    print('shutting down')
    logging.info("Nacking outstanding messages")
    tasks = [t for t in asyncio.all_tasks() if t is not
             asyncio.current_task()]

    [task.cancel() for task in tasks]

    logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    logging.info(f"Flushing metrics")
    loop.stop()


def generate_data():
    x = np.arange(0, 10, 0.1)
    m = 2
    b = 0.5
    y = m*x + b + np.random.normal(0,1,100)
    ind = np.random.permutation(np.arange(0, 100))
    return x[ind], y[ind]

# Importing the dataset
data = pd.read_csv('xclara.csv')

data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
K = 3
numClients = 2

def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )

    args = parser.parse_args()
    cnum = args.port - 8777
    X_ = X[cnum: -1: numClients]

    # This is so that Ctrl-C can be used to interrupt the server
    # See: https://stackoverflow.com/questions/24774980/why-cant-i-catch-sigint-when-asyncio-event-loop-is-running/24775107#24775107
    def wakeup():
        # Call again
        loop.call_later(0.1, wakeup)

    async def echo(websocket, path):
        round_num = 0
        async for message in websocket:
            func = dill.loads(message)
            centers, clusters = func(X_, K)
            #await asyncio.sleep(random.random())
            await websocket.send(dill.dumps([cnum, round_num, centers, clusters]))
            round_num = round_num + 1
    try:
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, wakeup)
        loop.set_exception_handler(handle_exception)
        loop.run_until_complete(websockets.serve(echo, 'localhost', args.port))
        loop.run_forever()
    finally:
        loop.close()
        logging.info("Successfully shutdown the Mayhem service.")

if __name__ == "__main__":
    main()
