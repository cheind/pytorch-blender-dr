import torch
from torch import nn
from torch.utils import data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
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

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, mean, std):
        # construct data with given mean and std
        if isinstance(std, torch.Tensor):
            std = std[None, :, None, None]
        if isinstance(mean, torch.Tensor):
            mean = mean[None, :, None, None]
        self.data = torch.randn(1000, 3, 24, 24) * std + mean
        
    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

def iterate_over_dataloader(model, dl):
    device = next(model.parameters()).device
    for x in dl:
        x = x.to(device)
        out = model(x)

def main(rank, gpus):  # pkill -9 python; ps
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpus])
    world_size = len(gpus)
    setup(rank, world_size)

    ds = TestDataset(mean=0, std=1)
    sampler = DistributedSampler(ds, shuffle=True) 

    # assume dl1 loads training data
    dl1 = data.DataLoader(ds, batch_size=100, 
        num_workers=4, shuffle=(sampler is None), 
        sampler=sampler, drop_last=False, 
        pin_memory=False, persistent_workers=False)
    
    # assume dl2 loads validation data
    dl2 = data.DataLoader(ds, batch_size=100, 
        num_workers=4, shuffle=False, sampler=None,
        pin_memory=False, persistent_workers=False)

    # print(f'==> Rank: {rank} at loading barrier.')
    # dist.barrier()  # wait for all processes to reach barrier

    model = TestModel()
    device = torch.device(rank)
    model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    for epoch in range(4):
        model.train()
        iterate_over_dataloader(model, dl1)  # training

        if rank == 0:
            model.eval()
            with torch.no_grad():
                iterate_over_dataloader(model, dl2)  # validation

        # print(f'==> Rank: {rank} at epoch barrier.')
        # dist.barrier()  # wait for all processes to reach barrier
        # print('Jumped over barrier!')

    cleanup()

    # If I use barrier here the console output is like:
    # ==> Rank: 0 at loading barrier.
    # ==> Rank: 1 at loading barrier.
    # ==> Rank: 1 at epoch barrier.
    # Jumped over barrier!
    # ==> Rank: 0 at epoch barrier.
    # Jumped over barrier!
    # ...which shows that at some point we get stuck without
    # any warning or error message and that we are not stuck at the
    # barrier but somewhere else! Note: without any barriers the 
    # code runs through completely! 

    # ==> https://discuss.pytorch.org/t/why-the-second-barrier-is-used-in-the-ddp-tutorial/87849
    # Q: If I just want to save a model, I don’t need dist.barrier() , right?

    # A: Yep, that should be fine. If only rank 0 saves the model and that might take 
    # very long, you can set the timeout argument in init_process_group 1 to a 
    # larger value. The default is 30min.

    # ...he also emphasized that it is not necessary to use a barrier in that case
    # because after each backwards call the processes will also wait for each other!

def run(main_fn, gpus):
    mp.spawn(main_fn,
             args=(gpus,),
             nprocs=len(gpus),
             join=True)

if __name__ == '__main__':
    gpus = (0, 1)
    assert len(gpus) > 1

    run(main, gpus)
