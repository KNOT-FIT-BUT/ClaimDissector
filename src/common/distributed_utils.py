# Written by Martin Fajcik <martin.fajcik@vut.cz>
#
import logging
import os
import pickle
import socket
from random import randint

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_ddp_envinit(rank, world_size, backend, address=None, port=None):
    address = 'localhost' if address is None else address
    port = str(randint(3000, 15000)) if port is None else port
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port

    logger.info(f"Setup env master address/port {address}:{port}")
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # this helps to avoid strange (v1.10+) torch DDP behavior,
    # that allocates some extra memory on GPU 0 for every process
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()


def setup_ddp_tcpinit(rank, world_size, backend, address, port):
    method = f'tcp://{address}:{port}'

    logger.info(f"Setup tcp address/port {address}:{port}")
    # initialize the process group
    dist.init_process_group(backend, init_method=method, rank=rank, world_size=world_size)


def setup_ddp_fsinit(rank, world_size, backend, path):
    # method = f'file://{path}'
    #
    # logger.info(f"Setup fs ddp at {path}")
    # # initialize the process group
    # dist.init_process_group(backend, init_method=method, rank=rank, world_size=world_size)
    # logger.info(f"Finished setup fs ddp at {path}")
    store = dist.FileStore(path, world_size)
    logger.info(f"Setup fs ddp at {path}")
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


# def share_list(obj, rank, torch_device, world_size, distributed=True):
#     # Collect object from replicas and form a list
#     # Taken from https://github.com/pytorch/pytorch/issues/3473#issuecomment-627361795
#     if not distributed:
#         return [obj]
#     MAX_LENGTH = 2 ** 20  # 1M
#     assert 0 <= rank < world_size
#     result = []
#     for i in range(world_size):
#         if rank == i:
#             data = pickle.dumps(obj)
#             data_length = len(data)
#             data = data_length.to_bytes(4, "big") + data
#             assert len(data) < MAX_LENGTH
#             data += bytes(MAX_LENGTH - len(data))
#             data = np.frombuffer(data, dtype=np.uint8)
#             assert len(data) == MAX_LENGTH
#             tensor = torch.from_numpy(data).to(torch_device)
#
#             print(f"RANK {rank}, reduced data to tensor {tensor.shape}")
#         else:
#             tensor = torch.zeros(MAX_LENGTH, dtype=torch.uint8, device="cuda")
#             print(f"RANK {rank}, writing zeros to tensor {tensor.shape}")
#
#         print(f"RANK {rank}, reduceSUM...")
#         dist.barrier()
#         print(f"RANK {rank},AFTER BARRIER reduceSUM...")
#         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#         print(f"RANK {rank}, AFTER reduceSUM...")
#         data = tensor.cpu().numpy().tobytes()
#         length = int.from_bytes(data[:4], "big")
#         data = data[4:length + 4]
#         result.append(pickle.loads(data))
#     return result

# def share_list(l, _id, rank, world_size, log=True):
#     "Just use pickle"
#     dist.barrier()
#     assert type(l) == list
#     out_fn = f"__shared_list_ID{_id}_R{rank}_W{world_size}.pkl"
#     with open(out_fn, "wb") as f:
#         pickle.dump(l, f)
#     if log:
#         logging.debug(f"R {dist.get_rank()} Data dumped. Approaching barrier #1 {out_fn}")
#     dist.barrier()
#     if log:
#         logging.debug(f"R {dist.get_rank()} Reading {out_fn}")
#     all_files = [f"__shared_list_ID{_id}_R{r}_W{world_size}.pkl" for r in range(world_size)]
#     lists = []
#     for in_fn in all_files:
#         with open(in_fn, "rb") as f:
#             logging.debug(f"R {dist.get_rank()} Loading {in_fn}")
#             lists.append(pickle.load(f))
#     if log:
#         logging.debug(f"R {dist.get_rank()} Pre-Barrier #2 {out_fn}")
#     dist.barrier()
#     os.remove(out_fn)
#     if log:
#         logging.debug(f"R {dist.get_rank()} Deleted {out_fn}")
#     return lists

def share_list(l, rank, world_size, log=False):
    "Just use pickle"
    assert type(l) == list
    PGID = os.getpgid(os.getpid())
    out_fn = f"__shared_list_H{socket.gethostname()}_PGID{PGID}_R{rank}_W{world_size}.pkl"
    with open(out_fn, "wb") as f:
        pickle.dump(l, f)
    if log:
        logging.debug(f"R {dist.get_rank()} Data dumped. Approaching barrier #1 {out_fn}")
    dist.barrier()
    if log:
        logging.debug(f"R {dist.get_rank()} Reading {out_fn}")
    all_files = [f"__shared_list_H{socket.gethostname()}_PGID{PGID}_R{r}_W{world_size}.pkl" for r in range(world_size)]
    lists = []
    for in_fn in all_files:
        with open(in_fn, "rb") as f:
            lists.append(pickle.load(f))
    if log:
        logging.debug(f"R {dist.get_rank()} Pre-Barrier #2 {out_fn}")
    dist.barrier()
    os.remove(out_fn)
    if log:
        logging.debug(f"R {dist.get_rank()} Deleted {out_fn}")
    return lists
