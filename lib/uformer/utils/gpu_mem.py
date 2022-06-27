import torch

def print_gpu_stats(gpu_stats,name):
    fmt_all = "[%s] Memory Allocated: %2.3f"
    fmt_res = "[%s] Memory Reserved: %2.3f"
    if gpu_stats:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.memory_allocated() / 1024**3
        print(fmt_all % (name,mem))
        mem = th.cuda.memory_reserved() / 1024**3
        print(fmt_res % (name,mem))

def print_peak_gpu_stats(gpu_stats,name,reset=True):
    fmt = "[%s] Peak Memory: %2.3f"
    if gpu_stats:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.max_memory_allocated(0)
        mem_gb = mem / (1024**3)
        print(fmt % (name,mem))
        print("Max Mem (GB): ",mem_gb)
        if reset: th.cuda.reset_peak_memory_stats()



