import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.parallel
import numpy as np

from utils import Walker, exchange
from config import eq_limits, params, model_class
from config import paired_ranks_for_exchange


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    local_gpu = rank % 3
    device = torch.device(f"cuda:{local_gpu}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=600)
        # device_id = device    # setting this argument makes the code run much slower
    )

    np.random.seed(rank)
    torch.manual_seed(rank)

    limits = eq_limits[rank]
    walker = Walker(limits, params, device, rank, model_class)

    t0 = time.time()
    print_every = 5000
    exchange_every = 50000
    save_every = 40000

    exchange_count = 0

    it = 0

    print(f'Starting one walker per container in rank {rank}')
    while True:

        walker.step()
        it += 1

        if it % exchange_every == 0:
            exchange_direction = exchange_count % 2
            exchange_count += 1
            pair = paired_ranks_for_exchange[exchange_direction]

            paired_rank = pair[rank]
            print(f'\nit: {walker.it:,}. I will try to exchange with rank {paired_rank}')
            exchange(walker, paired_rank, params)

        if walker.it % print_every == 0:
            print('\nTime:', f'{time.time() - t0:.2f}', 'rank', f'{rank}', 'it:', f'{walker.it:,}')
            if walker.need_initialization:
                print('Not yet initialized.')
            else:
                h = walker.h.size - (walker.h > params.flatness * walker.h.mean()).sum()
                print('h_zeros:', (walker.h == 0).sum(), '  cond_h:', f'{h}')
                print(
                    f'reject_out rate: {walker.reject_out / print_every}  accept rate: {walker.random_accept / print_every}')
                walker.reject_out = 0
                walker.random_accept = 0

            t0 = time.time()

        if walker.it % save_every == 0:
            walker.save()
            # Add this line to copy back immediately after saving
            # It copies from the scratch results folder to your project's results folder
            os.system(f'cp -r {os.environ["WLRESULTS"]} {os.environ["HOME"]}/rewl_sim_A/')
            print(f"Rank {rank}: Copied results to {os.environ['HOME']}/rewl_sim_A/results/ at iteration {it}")

    dist.barrier()
    print(f'{rank} finished!')
    dist.destroy_process_group()


if __name__ == "__main__":
    main()





