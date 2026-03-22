"""Benchmark: Cython engine vs pure-Python engine.

Three tiers:
  1. C-level simulation (simulate_random_rounds) — no Python in the loop
  2. Cython engine with Python agents — Python↔C boundary per move
  3. Pure-Python engine with Python agents — baseline

Usage:
    python benchmarks/bench_engine.py [N_ROUNDS]
"""
import os
import sys
import time
import random

N_ROUNDS_DEFAULT = 10_000


def bench_with_agents(label, play_round_fn, agent_cls, n_rounds):
    rng = random.Random(42)
    agents = (agent_cls(rng), agent_cls(rng))
    # Warm up
    for _ in range(min(100, n_rounds)):
        play_round_fn(agents, rng)

    start = time.perf_counter()
    for _ in range(n_rounds):
        play_round_fn(agents, rng)
    elapsed = time.perf_counter() - start
    rps = n_rounds / elapsed
    print(f"  {label:>25s}: {n_rounds:,} rounds in {elapsed:.3f}s  ({rps:,.0f} rounds/sec)")
    return rps


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else N_ROUNDS_DEFAULT
    print(f"Jaipur Engine Benchmark  (n={n_rounds:,})\n{'='*55}")

    results = {}

    # ── Tier 1: C-level simulation ───────────────────────────────
    try:
        from jaipur._engine import simulate_random_rounds
        # Warm up
        simulate_random_rounds(100)
        n_c = max(n_rounds, 100_000)
        start = time.perf_counter()
        w0, w1, draws = simulate_random_rounds(n_c)
        elapsed = time.perf_counter() - start
        rps = n_c / elapsed
        results["c_level"] = rps
        print(f"\n[Tier 1] C-level simulation (no Python in loop):")
        print(f"  {'simulate_random_rounds':>25s}: {n_c:,} rounds in {elapsed:.3f}s  ({rps:,.0f} rounds/sec)")
    except ImportError:
        print("\n[Tier 1] Cython engine not available — skipping")

    # ── Tier 2: Cython engine with Python agents ─────────────────
    try:
        from jaipur._engine import play_round as cy_play_round
        from jaipur.agents import RandomAgent
        print(f"\n[Tier 2] Cython engine + Python agents:")
        results["cython_agents"] = bench_with_agents(
            "Cython + RandomAgent", cy_play_round, RandomAgent, n_rounds
        )
    except ImportError:
        print("\n[Tier 2] Cython engine not available — skipping")

    # ── Tier 3: Pure-Python engine ───────────────────────────────
    print(f"\n[Tier 3] Pure-Python engine + Python agents:")
    src_dir = os.path.join(os.path.dirname(__file__), "..", "jaipur")
    so_files = [f for f in os.listdir(src_dir)
                if f.startswith("_engine") and f.endswith(".so")]
    moved = []
    for f in so_files:
        src = os.path.join(src_dir, f)
        dst = src + ".bak"
        os.rename(src, dst)
        moved.append((src, dst))

    # Clear cached modules
    for key in list(sys.modules.keys()):
        if "jaipur" in key:
            del sys.modules[key]

    try:
        from jaipur.game_fast import play_round as py_play_round
        from jaipur.agents import RandomAgent as PyRandomAgent
        results["pure_python"] = bench_with_agents(
            "Pure Python + RandomAgent", py_play_round, PyRandomAgent, n_rounds
        )
    finally:
        for src, dst in moved:
            os.rename(dst, src)
        for key in list(sys.modules.keys()):
            if "jaipur" in key:
                del sys.modules[key]

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Summary:")
    baseline = results.get("pure_python", 1)
    for label, key in [
        ("Pure Python", "pure_python"),
        ("Cython + agents", "cython_agents"),
        ("C-level (no Python)", "c_level"),
    ]:
        if key in results:
            speedup = results[key] / baseline
            print(f"  {label:>25s}: {results[key]:>10,.0f} rounds/sec  ({speedup:>6.1f}x)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
