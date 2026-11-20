"""Print two example evaluation episodes; the controlled players stay in place."""

import argparse
import json

from .env import FastapLBFEnv


def initial_grid(world):
    # Food, the two controlled agents and their teammates are visible in this
    # diagnostic view. Individual policies receive only local observations.
    grid = [[f"F{level}" if level else "." for level in row] for row in world.field]
    for index, player in enumerate(world.players):
        row, col = player.position
        role = f"C{index + 1}" if index < 2 else f"T{index - 1}"
        grid[row][col] = f"{role}({player.level})"
    return [" ".join(row) for row in grid]


def preview(seed, interval):
    env = FastapLBFEnv(seed=seed, switch_interval=interval)
    try:
        env.set_test_mode(True)
        env.reset()
        grid = initial_grid(env.world)
        total = 0.0
        for step in range(1, env.episode_limit + 1):
            reward, done, _ = env.step([0, 0])
            total += reward
            if done:
                break
        return {"initial_grid": grid, "events": env.events, "steps": step, "return": total}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    seed = parser.parse_args().seed
    print(json.dumps({"seed": seed, "stationary": preview(seed, None),
                      "switching": preview(seed, (5, 8))}, indent=2))


if __name__ == "__main__":
    main()
