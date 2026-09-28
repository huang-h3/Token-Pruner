"""CUDA-event and torch.profiler helpers for classification benchmarks."""

from contextlib import nullcontext

import torch


PROFILER_WAIT_STEPS = 1
PROFILER_WARMUP_STEPS = 1
PROFILER_ACTIVE_STEPS = 8


def profile_record(name, args):
    return (
        torch.profiler.record_function(name)
        if args.profiler
        else nullcontext()
    )


def profiler_ctx_func(args):
    return (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=PROFILER_WAIT_STEPS,
                warmup=PROFILER_WARMUP_STEPS,
                active=PROFILER_ACTIVE_STEPS,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.profiler_dir
            ),
            record_shapes=True,
            profile_memory=False,
            with_flops=True,
            with_stack=False,
            acc_events=True,
        )
        if args.profiler
        else nullcontext()
    )


def profiled_active_steps(total_steps):
    available = max(
        0,
        int(total_steps) - PROFILER_WAIT_STEPS - PROFILER_WARMUP_STEPS,
    )
    return min(PROFILER_ACTIVE_STEPS, available)


def warmup(args, model, num_frames, image_size, device):
    dummy_input = torch.randn(
        (args.batch_size, num_frames, 3, image_size, image_size),
        device=device,
    )
    with torch.inference_mode():
        print("Warming up GPU with dummy forward passes...", flush=True)
        for _ in range(5):
            model(dummy_input)
    torch.cuda.synchronize()
    del dummy_input
    print("GPU warmup complete.", flush=True)


def event_under_record(event, record_name):
    current = event
    while current is not None:
        if record_name in (current.name, current.key):
            return True
        current = current.cpu_parent
    return False


def sum_profiler_flops(profiler, record_name=None):
    return sum(
        int(event.flops or 0)
        for event in profiler.events()
        if record_name is None or event_under_record(event, record_name)
    )
