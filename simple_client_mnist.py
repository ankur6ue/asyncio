#!/usr/bin/env python3

# Client. Sends stuff from STDIN to the server.

import asyncio
import websockets
import numpy as np
import time

import torch
import dill
import torch.nn as nn
import torch.nn.functional as F
import zlib

import torch.optim as optim
from torchvision import datasets, transforms

#https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591

no_cuda = True
seed = 0
use_cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

async def handle_message(msg):
    """Kick off tasks for a given message.
    Args:
        msg (PubSubMessage): consumed message to process.
    """
    event = asyncio.Event()
    #asyncio.create_task(extend(msg, event))
    #asyncio.create_task(cleanup(msg, event))

    #await asyncio.gather(save(msg), restart_host(msg))
    event.set()

clients = set()
num_clients = 1

def register(websocket):
    clients.add(websocket)

def unregister(websocket):
    clients.remove(websocket)

model = Net()
async def consume(queue, event):
    """Consumer client to simulate subscribing to a publisher.
    Args:
        queue (asyncio.Queue): Queue from which to consume messages.
    """
    msg_per_round = 0
    curr_round = 0
    global model
    while True:
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

async def FL_main(uri, queue, event):
    async with websockets.connect(uri) as websocket:
        print('uri')
        register(websocket)
        model_srl = dill.dumps(model)
        model_srl_comp = zlib.compress(model_srl)
        await websocket.send(model_srl_comp)
        round = 0
        async for message in websocket:
            print(uri)
            event.clear()
            await queue.put(message)
            await event.wait()
            # check if we done
            if (round == 10):
                await websocket.close()
                # put none on the queue so consumers know when to exit
                await queue.put(None)
                unregister(websocket)
                return
            round = round + 1
            time.sleep(0.2)
            ser = dill.dumps(model)
            await websocket.send(ser)

connections = set()
connections.add('ws://localhost:8777')
#connections.add('ws://localhost:8778')

async def handler():
    event = asyncio.Event()
    queue = asyncio.Queue()
    consumer = consume(queue, event)
    consumer_task = asyncio.get_event_loop().create_task(consumer)
    # run the producer and wait for completion
    await asyncio.wait([FL_main(uri, queue, event) for uri in connections])
    # check size of clients: should be zero
    print(len(clients))
    # wait until the consumer has processed all items
    # await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer_task.cancel()
    print('done')

asyncio.get_event_loop().run_until_complete(handler())
asyncio.get_event_loop().close()