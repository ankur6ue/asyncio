import subprocess
import os
import signal
from torchvision import datasets
from torchvision import transforms


# Downloads MNIST dataset
mnist_trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

proc_list = []

call_alice = ["python", "run_simple_server_mnist.py", "--port", "8777", "--id", "alice"]

call_bob = ["python", "run_simple_server.py", "--port", "8778", "--id", "bob"]

print("Starting server for Alice")
p = subprocess.Popen(call_alice)
proc_list.append(p)

print("Starting server for Bob")
#p = subprocess.Popen(call_bob)
#proc_list.append(p)

input("Press Enter to kill servers")

for p in proc_list:
    os.kill(p.pid, signal.SIGINT)

