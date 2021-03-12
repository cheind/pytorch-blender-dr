import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    # A free port on the machine that will host the process with rank 0.
    os.environ['MASTER_ADDR'] = 'localhost'
    # IP address of the machine that will host the process with rank 0.
    os.environ['MASTER_PORT'] = '12355'
    # The total number of processes, so master knows how many workers to wait for.
    os.environ['WORLD_SIZE'] = str(world_size)
    # Rank of each process, so they will know whether it is the master of a worker.
    os.environ['RANK'] = str(rank)
    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def example(rank, world_size):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    setup(rank, world_size)

    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

    cleanup()

def main():
    world_size = 2

    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    main()