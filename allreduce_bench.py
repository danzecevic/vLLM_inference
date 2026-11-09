# Measures all-reduce latency and bandwidth between GPUs (the operation TP runs twice per layer).
# Run: torchrun --nproc-per-node=N allreduce_bench.py
import os, time, torch, torch.distributed as dist

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()

for size in [64 * 1024, 1 << 20, 16 << 20, 256 << 20]:  # bytes
    x = torch.ones(size // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):  # warm-up
        dist.all_reduce(x)
    torch.cuda.synchronize()

    iters = 50 if size < (16 << 20) else 20
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    busbw = size * 2 * (world - 1) / world / dt / 1e9  # same definition as nccl-tests
    if rank == 0:
        print(f"{size / 1024:>9.0f} KB {dt * 1e6:10.1f} us   bus bw {busbw:6.2f} GB/s")

dist.destroy_process_group()
