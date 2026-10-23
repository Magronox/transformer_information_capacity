"""Export the per-seed results behind the paper from the Optuna study databases (read-only) to CSV."""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import closing
from pathlib import Path
import sqlite3

SEEDS = ("10", "20", "30")
TRIAL_BUDGET = 60
WIDTH_ENTITIES = (15, 30, 60, 120, 260, 520, 1040, 2080, 4160, 8320, 16640)
POWERLAW_ENTITIES = WIDTH_ENTITIES[:9]
OLMO3_ENTITIES = WIDTH_ENTITIES[:8]
SWEEP = ((9216, 1), (4608, 2), (3072, 3), (2304, 4), (1152, 8), (576, 16), (288, 32), (144, 64), (96, 96))
DEPTHS = (1, 2, 4, 6, 8, 12, 16)
DEPTHS_WITH_RUN_LOGS = (1, 2, 4, 6, 8)
MAX_SIZE_DIMS = (3, 4, 5, 7, 8, 9, 10, 13, 14, 16)
MAX_BATCHES = 40_000

UNIFORM_DB = ("optuna_uniform_ent_{e}_rel_{r}_layers_1_decay_{s}_seeds_10_20_30_v34_TPE.db",
              "min_dim_uniform_E{e}_R{r}_L1_{s}.db")
POWERLAW_DB = ("optuna_powerlaw_ent_{e}_rel_{r}_layers_1_decay_{s}_alpha_1.00_mintail_1_seeds_10_20_30_v34_TPE.db",
               "min_dim_powerlaw_E{e}_R{r}_L1_{s}_alpha1.00_mintail1.db")
OLMO3_DB = ("optuna_uniform_ent_{e}_rel_{r}_layers_1_decay_{s}_seeds_10_20_30_v34_TPE_olmo3.db",)
DEPTH_DB = ("maxds_uniform_dim_8_layers_{layers}_decay_{s}_seeds_10_20_30_v8r3.db",
            "max_size_uniform_d8_R3_L{layers}_{s}.db")
MAX_SIZE_DB = ("maxds_uniform_dim_{dim}_layers_1_decay_linear_seeds_10_20_30_v8r3f1.db",
               "max_size_uniform_d{dim}_R3_L1_linear.db")
DEPTH_RUN_LOGS = ("maxds_uniform_dim_8_layers_{layers}_{s}_v8r3", "max_size_uniform_d8_R3_L{layers}_{s}")


def connect(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def find(names: tuple[str, ...], dirs: list[Path], **fields) -> Path | None:
    paths = (d / name.format(**fields) for d in dirs for name in names)
    return next((p for p in paths if p.exists()), None)


def attribute(c: sqlite3.Connection, trial_id: int, key: str) -> dict:
    row = c.execute("SELECT value_json FROM trial_user_attributes WHERE trial_id = ? AND key = ?",
                    (trial_id, key)).fetchone()
    return json.loads(row[0])


def best_objective_trial(db: Path) -> tuple[float, int, list[int | None]]:
    with closing(connect(db)) as c:
        trial_id, number, value = c.execute(
            "SELECT t.trial_id, t.number, tv.value FROM trial_values tv JOIN trials t ON t.trial_id = tv.trial_id "
            "WHERE t.state = 'COMPLETE' ORDER BY tv.value ASC, t.number ASC LIMIT 1").fetchone()
        per_seed = attribute(c, trial_id, "per_seed_min_dim")
    return value, number, [per_seed.get(s) for s in SEEDS]


def min_dim_rows(db_dir: Path, patterns: tuple[str, ...], datasets: list[tuple[int, int]]) -> list[list]:
    rows = []
    for e, r in datasets:
        best = None
        for schedule in ("linear", "cosine"):
            db = find(patterns, [db_dir], e=e, r=r, s=schedule)
            if db:
                value, number, seeds = best_objective_trial(db)
                if best is None or value < best[0]:
                    best = (value, schedule, number, seeds)
        _, schedule, number, seeds = best
        rows.append([e, r, e * r, schedule, number, *seeds])
    return rows


def first_completed_trials(c: sqlite3.Connection) -> list[tuple[int, int]]:
    """(trial id, trial number) of the first TRIAL_BUDGET completed trials, by completion time."""
    return c.execute(
        "SELECT t.trial_id, t.number FROM trials t JOIN trial_user_attributes a ON a.trial_id = t.trial_id "
        "WHERE t.state = 'COMPLETE' AND a.key = 'per_seed_max_size' ORDER BY t.datetime_complete LIMIT ?",
        (TRIAL_BUDGET,)).fetchall()


def best_mean_trial(dbs: dict[str, Path]) -> tuple[str, int, list[int], list[int | None]]:
    """(schedule, trial number, per-seed max size, per-seed steps to 100%) of the largest seed mean."""
    best = None
    for schedule, db in dbs.items():
        with closing(connect(db)) as c:
            for trial_id, number in first_completed_trials(c):
                sizes = [attribute(c, trial_id, "per_seed_max_size").get(s) or 0 for s in SEEDS]
                if best is None or sum(sizes) > sum(best[2]):
                    steps = attribute(c, trial_id, "per_seed_batches100")
                    best = (schedule, number, sizes, [steps.get(s) for s in SEEDS])
    return best


def depth_rows(db_dirs: list[Path]) -> list[list]:
    rows = []
    for layers in DEPTHS:
        dbs = {s: find(DEPTH_DB, db_dirs, layers=layers, s=s) for s in ("cosine", "linear")}
        schedule, number, sizes, steps = best_mean_trial(dbs)
        rows.append([layers, schedule, number, *sizes, *steps])
    return rows


def max_size_rows(db_dirs: list[Path]) -> list[list]:
    rows = []
    for dim in MAX_SIZE_DIMS:
        schedule, number, sizes, _ = best_mean_trial({"linear": find(MAX_SIZE_DB, db_dirs, dim=dim)})
        rows.append([dim, schedule, number, *sizes])
    return rows


def hyperparameter_key(lr: float, weight_decay: float, min_lr: float, decay_end_ratio: float, schedule: str) -> tuple:
    return round(lr, 9), round(weight_decay, 12), round(min_lr, 12), round(decay_end_ratio, 6), schedule


def trial_keys(db: Path, schedule: str) -> set[tuple]:
    keys = set()
    with closing(connect(db)) as c:
        for trial_id, _ in first_completed_trials(c):
            p = dict(c.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id = ?", (trial_id,)))
            weight_decay = p["weight_decay"] if p.get("use_weight_decay", 0) == 1.0 else 0.0
            keys.add(hyperparameter_key(p["lr"], weight_decay, p["lr"] * p["min_lr_ratio"],
                                        min(1.0, p["decay_end_step"] / MAX_BATCHES), schedule))
    return keys


def depth_run_rows(db_dirs: list[Path], run_logs: Path) -> list[list]:
    """Per depth and schedule: training runs of the first 60 trials, and how many reached 100% but did not hold it."""
    rows = []
    for layers in DEPTHS_WITH_RUN_LOGS:
        for schedule in ("cosine", "linear"):
            keys = trial_keys(find(DEPTH_DB, db_dirs, layers=layers, s=schedule), schedule)
            runs = unstable = 0
            for path in find(DEPTH_RUN_LOGS, [run_logs], layers=layers, s=schedule).glob("*.json"):
                run = json.loads(path.read_text())
                k = run["key"]
                if hyperparameter_key(k["lr"], k["weight_decay"], k["min_lr"], k["decay_end_ratio"],
                                      k["lr_decay_type"]) in keys:
                    runs += 1
                    unstable += not run["result"]["converged"] and run["result"]["batches100"] is not None
            rows.append([layers, schedule, runs, unstable])
    return rows


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def seed_columns(name: str) -> list[str]:
    return [f"{name}_seed_{s}" for s in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-dim-dbs", type=Path, required=True, help="directory of the GPT-2 minimum-dimension studies")
    parser.add_argument("--olmo3-dbs", type=Path, required=True, help="directory of the OLMo-3 minimum-dimension studies")
    parser.add_argument("--max-size-dbs", type=Path, nargs="+", required=True,
                        help="directories of the maximum-size studies (searched in order)")
    parser.add_argument("--run-logs", type=Path, required=True, help="directory of the per-run logs of the depth studies")
    parser.add_argument("--results", type=Path, required=True, help="output directory for the CSVs")
    args = parser.parse_args()
    args.results.mkdir(exist_ok=True)

    min_dim_header = ["family", "entities", "relations", "facts", "schedule", "trial", *seed_columns("min_dim")]
    width = ([["uniform", *row] for row in min_dim_rows(args.min_dim_dbs, UNIFORM_DB, [(e, 3) for e in WIDTH_ENTITIES])]
             + [["powerlaw", *row] for row in min_dim_rows(args.min_dim_dbs, POWERLAW_DB, [(e, 3) for e in POWERLAW_ENTITIES])])
    write_csv(args.results / "width_min_dim.csv", min_dim_header, width)
    write_csv(args.results / "relations_sweep.csv", min_dim_header[1:], min_dim_rows(args.min_dim_dbs, UNIFORM_DB, list(SWEEP)))
    write_csv(args.results / "olmo3_min_dim.csv", min_dim_header[1:],
              min_dim_rows(args.olmo3_dbs, OLMO3_DB, [(e, 3) for e in OLMO3_ENTITIES]))
    write_csv(args.results / "depth_max_size.csv",
              ["layers", "schedule", "trial", *seed_columns("max_size"), *seed_columns("steps_to_100")],
              depth_rows(args.max_size_dbs))
    write_csv(args.results / "depth_training_runs.csv", ["layers", "schedule", "runs", "reached_but_not_held"],
              depth_run_rows(args.max_size_dbs, args.run_logs))
    write_csv(args.results / "width_max_size.csv", ["dim", "schedule", "trial", *seed_columns("max_size")],
              max_size_rows(args.max_size_dbs))


if __name__ == "__main__":
    main()
