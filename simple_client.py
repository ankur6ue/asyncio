#!/usr/bin/env python3

# Client. Sends stuff from STDIN to the server.

import asyncio
import websockets
import numpy as np
import dill
import time
import cvxopt
#https://medium.com/@emlynoregan/serialising-all-the-functions-in-python-cd880a63b591

K = 3
def estimate_coef(y):
    x = y
    def test(y):
        return x+y
    return test

def model(_C):
    C = _C;
    def run_one_iteration(X, K):
        # Euclidean Distance Caculator
        def dist(a, b, ax=1):
            return np.linalg.norm(a - b, axis=ax)
        clusters = np.zeros(len(X))
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        # Finding the new centroids by taking the average value
        for i in range(K):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        return C, clusters
    return run_one_iteration

C_x = np.random.randint(0, 40-20, size=K)
# Y coordinates of random centroids
C_y = np.random.randint(0, 40-20, size=K)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

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

def register(websocket):
    clients.add(websocket)

def unregister(websocket):
    clients.remove(websocket)

async def consume(queue, event):
    """Consumer client to simulate subscribing to a publisher.
    Args:
        queue (asyncio.Queue): Queue from which to consume messages.
    """
    msg_per_round = 0
    num_clients = 2
    centers_sum = np.zeros((K,2))
    curr_round = 0
    global C
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
        centers = unpckd_msg[2]
        clusters = unpckd_msg[3]
        print('recv: client#: {0}, round#: {1}'.format(cnum, round))
        if (round == curr_round):
            msg_per_round = msg_per_round + 1
            centers_sum = centers_sum + centers
            if (msg_per_round == num_clients):
                msg_per_round = 0
                centers = centers_sum/num_clients
                C = centers
                centers_sum = np.zeros((K, 2))
                curr_round = curr_round + 1
                event.set()
        queue.task_done()

async def FL_main(uri, queue, event):
    async with websockets.connect(uri) as websocket:
        print('uri')
        register(websocket)
        ser = dill.dumps(model(C))
        await websocket.send(ser)
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
            print('centers: {0}'.format(C))
            time.sleep(0.2)
            ser = dill.dumps(model(C))
            await websocket.send(ser)

connections = set()
connections.add('ws://localhost:8777')
connections.add('ws://localhost:8778')

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