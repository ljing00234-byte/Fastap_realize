"""Compare parallel rollout sizes on a single team with an equal optimizer budget."""

import gc
from time import perf_counter

import torch


def _measure(learner, world, stage, warmup_episodes, measured_episodes):
    device = learner.device
    n = world.batch_size
    if warmup_episodes < learner.cfg.batch_size:
        raise ValueError("Warm-up must fill the replay before timed measurement")
    if learner.steps or learner.episodes or learner.updates:
        raise ValueError("Each benchmark size requires a freshly initialized learner")
    for _ in range(warmup_episodes // n):
        learner.train_batch(world, stage)
    warmup_steps, warmup_updates = learner.steps, learner.updates

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    rollout_seconds = update_seconds = 0.0
    start = perf_counter()
    for _ in range(measured_episodes // n):
        batch = learner.train_batch(world, stage)
        rollout_seconds += batch['rollout_seconds']
        update_seconds += batch['update_seconds']
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    steps = learner.steps - warmup_steps
    updates = learner.updates - warmup_updates
    if updates != measured_episodes or learner.episodes != warmup_episodes + measured_episodes:
        raise RuntimeError("QMIX benchmark collected an unexpected episode or optimizer budget")
    return dict(num_envs=n, device=str(device), measured_episodes=measured_episodes,
                measured_steps=steps, measured_updates=updates, wall_seconds=elapsed,
                steps_per_second=steps / elapsed, updates_per_second=updates / elapsed,
                rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                warmup_steps=warmup_steps, warmup_updates=warmup_updates,
                total_training_steps=learner.steps, total_optimizer_updates=learner.updates,
                measured_peak_torch_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2**20
                                                   if device.type == 'cuda' else None),
                measured_peak_torch_reserved_mib=(torch.cuda.max_memory_reserved(device) / 2**20
                                                  if device.type == 'cuda' else None))


def benchmark(learner_factory, world_factory, stage='adjacent', *, parallel_sizes=(1, 8, 32),
              warmup_episodes=32, measured_episodes=128):
    """Call fixed-seed learner_factory() and world_factory(num_envs) for each trial.

    Each run includes replay writes and optimization, excludes validation and checkpoints,
    and reports the measured PyTorch allocator peak separately from the outer GPU monitor.
    """
    sizes = tuple(parallel_sizes)
    if not sizes or min(warmup_episodes, measured_episodes) < 1:
        raise ValueError("Specify parallel sizes and positive episode counts")
    for n in sizes:
        if n < 1 or warmup_episodes % n or measured_episodes % n:
            raise ValueError("Each episode budget must be divisible by every parallel size")

    runs = []
    reference_seeds = None
    for n in sizes:
        learner = learner_factory()
        device = learner.device
        world = None
        try:
            world = world_factory(n)
            seeds = (learner.action_rng.initial_seed(), learner.replay_rng.initial_seed(),
                     world.generator.initial_seed())
            if reference_seeds is None:
                reference_seeds = seeds
            if seeds != reference_seeds or world.batch_size != n or world.device != device:
                raise ValueError("Benchmark factories must recreate the fixed seeds, size and device")
            runs.append(_measure(learner, world, stage, warmup_episodes, measured_episodes))
        finally:
            del learner, world
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return dict(stage=stage, warmup_episodes=warmup_episodes, measured_episodes=measured_episodes,
                parallel_sizes=list(sizes), seeds=dict(zip(('action', 'replay', 'world'), reference_seeds)),
                runs=runs)
