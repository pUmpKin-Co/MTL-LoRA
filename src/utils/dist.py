import torch.distributed as dist


def get_global_rank():
    if dist.is_available():
        return dist.get_rank()
    else:
        return 0

def get_global_size():
    if dist.is_available():
        return dist.get_world_size()
    else:
        return 1