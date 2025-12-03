#!/usr/bin/env python3
"""
Grid Evolution Benchmark Suite — Config-Driven Multi-Version Support
Benchmarks all implementations defined in benchmark_config.toml
"""

import csv
import hashlib
import importlib
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tomllib
from loguru import logger

# === Load config ===
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = SCRIPT_DIR / "benchmark_config.toml"

if not CONFIG_FILE.exists():
    print(f"Error: {CONFIG_FILE} not found")
    sys.exit(1)

with open(CONFIG_FILE, "rb") as f:
    cfg = tomllib.load(f)

b = cfg["benchmark"]
f = cfg.get("features", {})
p = cfg.get("paths", {})
o = cfg.get("output", {})

# Benchmark parameters
ROWS = int(b["rows"])
COLS = int(b["cols"])
GENS = int(b["generations"])
SEED = int(b.get("seed", 42))
WARMUP = int(b.get("warmup_generations", 5))

# Features
RUN_PYTHON = bool(f.get("run_python", True))
RUN_MOJO = bool(f.get("run_mojo", True))
PERSIST_MOJO = bool(f.get("persist_generated_mojo", False))

# Paths
INITIAL_GRID_CSV = SCRIPT_DIR / p.get("initial_grid_csv", "initial_grid.csv")
RESULTS_DIR = SCRIPT_DIR / p.get("results_dir", "benchmark_results")
LOG_FILE = SCRIPT_DIR / p.get("log_file", "benchmark_results/benchmark.log")

# Output options
SAVE_CSV = bool(o.get("save_results_to_csv", True))
CSV_TIMESTAMP_FMT = o.get("csv_timestamp_format", "%Y%m%d_%H%M%S")
LOG_LEVEL = o.get("log_level", "INFO")

# Mojo template - single file for all versions
MOJO_TEMPLATE = SCRIPT_DIR / "run_grid_bench.mojo"

# Implementations from config
IMPLEMENTATIONS = cfg.get("implementations", [])


# === Setup logging ===
RESULTS_DIR.mkdir(exist_ok=True)
logger.remove()  # Remove default handler
logger.add(sys.stderr, level=LOG_LEVEL)
logger.add(LOG_FILE, rotation="10 MB", retention="30 days", level="DEBUG")


# === Helpers ===
def trunc(s: str, n: int = 16) -> str:
    return s[:n] + "..." + s[-n:] if len(s) > n * 2 + 3 else s


def generate_initial_grid():
    random.seed(SEED)
    np.random.seed(SEED)
    data = [[random.randint(0, 1) for _ in range(COLS)] for _ in range(ROWS)]
    with open(INITIAL_GRID_CSV, "w", newline="") as f:
        csv.writer(f).writerows(data)
    logger.info(f"Generated initial grid: {ROWS}×{COLS}, seed={SEED}")
    return data


# === Benchmark Runner ===
class BenchmarkRunner:
    def __init__(self):
        self.results = {}  # name -> {time, description, type}
        self.fingerprints = {}
        self.run_timestamp = datetime.now().strftime(CSV_TIMESTAMP_FMT)

    def run_python_impl(self, impl_config):
        """Run a Python implementation."""
        name = impl_config["name"]
        module_name = impl_config["module"]
        class_name = impl_config["class"]
        description = impl_config.get("description", "")
        
        logger.info(f"Running {name}...")
        print(f"Running {name}...", end="", flush=True)
        
        try:
            module = importlib.import_module(module_name)
            GridCls = getattr(module, class_name)
            
            # Load data
            if "np" in module_name.lower() or "numpy" in name.lower():
                grid = GridCls(np.loadtxt(INITIAL_GRID_CSV, delimiter=",", dtype=np.uint8))
            else:
                random.seed(SEED)
                initial_data = [[random.randint(0, 1) for _ in range(COLS)] for _ in range(ROWS)]
                grid = GridCls(ROWS, COLS, initial_data)
            
            # Warmup
            for _ in range(WARMUP):
                grid = grid.evolve()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(GENS):
                grid = grid.evolve()
            duration = time.perf_counter() - start
            
            self.results[name] = {
                "time": duration,
                "type": "python",
                "description": description,
            }
            self.fingerprints[name] = grid.fingerprint()
            
            logger.info(f"{name}: {duration:.6f}s")
            print(f" → {duration:.6f} s")
            
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            print(f" FAILED: {e}")

    def run_mojo_impl(self, impl_config):
        """Run a Mojo implementation using template replacement."""
        name = impl_config["name"]
        grid_module = impl_config["grid_module"]
        description = impl_config.get("description", "")
        
        logger.info(f"Running {name}...")
        print(f"Running {name}...", end="", flush=True)
        
        if not MOJO_TEMPLATE.exists():
            logger.error(f"Template not found: {MOJO_TEMPLATE}")
            print(f" FAILED: Template not found")
            return
        
        # Read template
        code = MOJO_TEMPLATE.read_text()
        
        # Replace import statement
        code = code.replace("from gridv1 import Grid", f"from {grid_module} import Grid")
        
        # Replace alias values from config
        for key, value in b.items():
            if key in ["seed"]:  # Skip non-alias keys
                continue
            # Find and replace the entire alias line
            for line in code.splitlines():
                if line.strip().startswith(f"alias {key} ="):
                    new_line = f"alias {key} = {value}"
                    code = code.replace(line, new_line)
                    break
        
        # Generate version-specific file
        safe_name = name.lower().replace(" ", "_")
        generated_path = SCRIPT_DIR / f"run_grid_bench_{safe_name}_generated.mojo"
        generated_path.write_text(code)
        
        if PERSIST_MOJO:
            persistent = SCRIPT_DIR / f"{safe_name}_{int(time.time())}.mojo"
            persistent.write_text(code)
            logger.debug(f"Persisted: {persistent.name}")
        
        try:
            result = subprocess.run(
                ["mojo", "run", str(generated_path)],
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode != 0:
                logger.error(f"{name} failed: {result.stderr}")
                print(f" FAILED\n{result.stderr}")
                return
            
            output = result.stdout.strip()
            if "Mojo_time:" not in output or "Mojo_fingerprint_str:" not in output:
                logger.warning(f"{name}: incomplete output")
                print(" WARNING: Incomplete output")
                return
            
            duration = float(output.split("Mojo_time:")[1].split()[0])
            raw_str = output.split("Mojo_fingerprint_str:")[1].strip()
            mojo_hash = hashlib.sha256(raw_str.encode()).hexdigest()
            
            self.results[name] = {
                "time": duration,
                "type": "mojo",
                "description": description,
            }
            self.fingerprints[name] = mojo_hash
            
            logger.info(f"{name}: {duration:.6f}s")
            print(f" → {duration:.6f} s")
            
        except FileNotFoundError:
            logger.error("mojo not found in PATH")
            print(" ERROR: 'mojo' not found")
        except subprocess.TimeoutExpired:
            logger.error(f"{name} timeout")
            print(" ERROR: Timeout")
        finally:
            # Cleanup if not persisting
            if generated_path.exists() and not PERSIST_MOJO:
                generated_path.unlink()

    def run_all_implementations(self):
        """Run all enabled implementations from config."""
        for impl in IMPLEMENTATIONS:
            if not impl.get("enabled", True):
                logger.debug(f"Skipping disabled: {impl['name']}")
                continue
            
            impl_type = impl["type"]
            
            if impl_type == "python" and RUN_PYTHON:
                self.run_python_impl(impl)
            elif impl_type == "mojo" and RUN_MOJO:
                self.run_mojo_impl(impl)
            else:
                logger.debug(f"Skipping {impl['name']} (type={impl_type})")


    def verify(self):
        """Verify all implementations produce identical results."""
        logger.info("Starting correctness verification")
        print("\nCorrectness Verification")
        
        if not self.fingerprints:
            logger.warning("No fingerprints to verify")
            return
        
        # Use first result as reference
        ref_name = list(self.fingerprints.keys())[0]
        ref_fp = self.fingerprints[ref_name]
        
        print(f"   Reference: {ref_name}: {trunc(ref_fp)}")
        all_match = True
        
        for name, fp in self.fingerprints.items():
            if name == ref_name:
                continue
            if fp == ref_fp:
                print(f"   {name:<20}: ✓ match")
                logger.debug(f"{name}: fingerprint matches reference")
            else:
                print(f"   {name:<20}: ✗ MISMATCH! {trunc(fp)}")
                logger.error(f"{name}: fingerprint mismatch")
                all_match = False
        
        if all_match:
            print("   ✓ ALL IMPLEMENTATIONS PRODUCE IDENTICAL RESULTS")
            logger.info("Correctness verification passed")
        else:
            print("   ✗ CORRECTNESS FAILURE")
            logger.error("Correctness verification failed")
            sys.exit(1)
    
    def save_csv(self):
        """Save results to timestamped CSV."""
        if not SAVE_CSV or not self.results:
            return
        
        csv_filename = RESULTS_DIR / f"benchmark_results_{self.run_timestamp}.csv"
        logger.info(f"Saving results to {csv_filename}")
        
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "implementation",
                "type",
                "description",
                "grid_size",
                "generations",
                "time_seconds",
                "speedup_vs_baseline",
                "fingerprint"
            ])
            
            # Get baseline (first result)
            baseline_time = list(self.results.values())[0]["time"]
            
            for name, result in self.results.items():
                speedup = baseline_time / result["time"]
                writer.writerow([
                    self.run_timestamp,
                    name,
                    result["type"],
                    result["description"],
                    f"{ROWS}×{COLS}",
                    GENS,
                    f"{result['time']:.6f}",
                    f"{speedup:.2f}",
                    trunc(self.fingerprints.get(name, ""), 32)
                ])
        
        print(f"\nResults saved to: {csv_filename.name}")
    
    def run(self):
        """Main benchmark execution."""
        logger.info(f"Starting benchmark run: {ROWS}×{COLS} grid, {GENS} generations")
        
        # Generate initial grid
        generate_initial_grid()
        
        # Run all implementations
        self.run_all_implementations()
        
        # Verify correctness
        self.verify()
        
        # Print summary
        self.print_summary()
        
        # Save CSV
        self.save_csv()
        
        # Cleanup
        if INITIAL_GRID_CSV.exists():
            INITIAL_GRID_CSV.unlink()
            logger.debug("Cleaned up initial grid CSV")

    def print_summary(self):
        """Print formatted benchmark results."""
        if not self.results:
            logger.warning("No results to print")
            return
        
        # Get baseline (first implementation)
        baseline_name = list(self.results.keys())[0]
        baseline_time = self.results[baseline_name]["time"]
        
        # Find fastest
        fastest_name = min(self.results.items(), key=lambda x: x[1]["time"])[0]
        
        print("\n" + "═" * 80)
        print(f" GRID EVOLUTION BENCHMARK — {ROWS}×{COLS} grid, {GENS} generations")
        print(f" Seed: {SEED} | Warm-up: {WARMUP} generations | Time: {self.run_timestamp}")
        print("═" * 80)
        print(f" {'Implementation':<20} {'Time (s)':<12} {'Speedup':<10} {'Description'}")
        print("─" * 80)
        
        for name, result in self.results.items():
            t = result["time"]
            speedup = baseline_time / t
            marker = " ← FASTEST" if name == fastest_name else ""
            desc = result.get("description", "")
            print(f" {name:<20} {t:>10.6f} s   {speedup:>6.2f}×   {desc}{marker}")
        
        print("─" * 80)
        print("   ✓ ALL IMPLEMENTATIONS VERIFIED CORRECT")
        print("═" * 80)


if __name__ == "__main__":
    print("Grid Evolution Benchmark Suite — Config-Driven")
    print(f"Config: {CONFIG_FILE.name}")
    print(f"Results: {RESULTS_DIR}")
    print(f"Log: {LOG_FILE}\n")
    
    logger.info("=" * 60)
    logger.info("Starting new benchmark run")
    logger.info(f"Config: {CONFIG_FILE}")
    logger.info(f"Grid: {ROWS}×{COLS}, Generations: {GENS}, Seed: {SEED}")
    logger.info("=" * 60)
    
    BenchmarkRunner().run()
