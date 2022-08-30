import torch as th

def print_gpu_stats(verbose,name):
    fmt_all = "[%s] Memory Allocated [GB]: %2.3f"
    fmt_res = "[%s] Memory Reserved [GB]: %2.3f"
    th.cuda.empty_cache()
    th.cuda.synchronize()
    mem_alloc = th.cuda.memory_allocated() / 1024**3
    mem_res = th.cuda.memory_reserved() / 1024**3
    if verbose:
        print(fmt_all % (name,mem_alloc))
        print(fmt_res % (name,mem_res))
    return mem_alloc,mem_res

def reset_peak_gpu_stats():
    th.cuda.reset_max_memory_allocated()

def print_peak_gpu_stats(verbose,name,reset=True):
    fmt = "[%s] Peak Memory Allocated [GB]: %2.3f"
    mem_alloc = th.cuda.max_memory_allocated(0) / (1024.**3)
    mem_res = th.cuda.max_memory_reserved(0) / (1024.**3)
    if verbose:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        print(fmt % (name,mem_alloc))
        if reset: th.cuda.reset_peak_memory_stats()
    return mem_alloc,mem_res
