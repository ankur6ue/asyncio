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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets, transforms
import signal

async def consume(websocket, queue):
    """Consumer client to simulate subscribing to a publisher.
    Args:
        queue (asyncio.Queue): Queue from which to consume messages.
    """
    msg_per_round = 0
    curr_round = 0
    global model
    async for message in websocket:
        msg = await queue.get()
        if msg is None:
            # the producer emits None to indicate that it is done
            queue.task_done()
            break
        # otherwise unpack the msg
        unpckd_msg = dill.loads(msg)
        cnum = unpckd_msg[0]
        round = unpckd_msg[1]
        model = unpckd_msg[2]
        print('recv: client#: {0}, round#: {1}'.format(cnum, round))
        if (round == curr_round):
            msg_per_round = msg_per_round + 1
            if (msg_per_round == num_clients):
                msg_per_round = 0

                curr_round = curr_round + 1
                event.set()
        queue.task_done()

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

no_cuda = True
seed = 0
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if use_cuda else "cpu")
def generate_data():
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

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

    # This is so that Ctrl-C can be used to interrupt the server
    # See: https://stackoverflow.com/questions/24774980/why-cant-i-catch-sigint-when-asyncio-event-loop-is-running/24775107#24775107
    def wakeup():
        # Call again
        loop.call_later(0.1, wakeup)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    async def FL_train(websocket, path):
        round_num = 0
        train_loader = generate_data()
        consumer_task = asyncio.get_event_loop().create_task(consumer, websocket)
        async for message in websocket:
            model = dill.loads(message)
            print('received model')
            lr = 0.01
            momentum = 0.5
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            train(model, device, train_loader, optimizer, round_num)
            msg = dill.dumps(model)
            #await asyncio.sleep(random.random())
            await websocket.send(dill.dumps([cnum, round_num, msg]))
            round_num = round_num + 1
    try:
        loop = asyncio.get_event_loop()
        loop.call_later(0.1, wakeup)
        loop.set_exception_handler(handle_exception)

        loop.run_until_complete(websockets.serve(FL_train, 'localhost', args.port))
        loop.run_forever()
    finally:
        loop.close()
        logging.info("Successfully shutdown the Mayhem service.")

if __name__ == "__main__":
    main()
