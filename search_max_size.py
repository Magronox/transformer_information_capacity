"""Max-N search: the largest dataset a transformer of width d and depth L memorizes."""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import optuna
import torch
from optuna.samplers import TPESampler
from optuna.trial import TrialState

import train
from train import TrainConfig

SEEDS = (10, 20, 30)
FAMILIES = ("uniform", "symmetric", "two_hop", "powerlaw")
DECAY_TYPES = ("linear", "cosine")

MAX_BATCHES = 40_000
WARMUP_STEPS_PER_LAYER = 500
MICRO_BATCH_ABOVE = 10_000

LR_RANGE = (1e-7, 3.0)
WEIGHT_DECAY_RANGE = (1e-8, 1e-1)
MIN_LR_RATIO_RANGE = (1e-8, 1.0)
DECAY_STEP_MIN = 100

PATIENCE = 10
FAIL_ALLOWANCE = 10
STARTUP_TRIALS_PER_PARAM = 5
SAMPLER_SEED = 42


@dataclass(frozen=True)
class Setup:
    family: str
    dim: int
    layers: int
    num_relations: int
    coarse_step: int
    fine_step: int
    start_size: int
    max_size: int
    warmup_steps: int
    clip_norm: float
    alpha: float


@dataclass(frozen=True)
class Hparams:
    lr: float
    weight_decay: float
    lr_decay_type: str
    min_lr: float
    decay_end_ratio: float


def study_name(setup: Setup, decay_type: str) -> str:
    name = (f"max_size_{setup.family}_d{setup.dim}_R{setup.num_relations}"
            f"_L{setup.layers}_{decay_type}")
    if setup.family == "powerlaw":
        name += f"_alpha{setup.alpha:.2f}"
    return name


def round_to(size: int, step: int) -> int:
    return max(step, round(size / step) * step)


def initial_size(dim: int, layers: int, coarse_step: int, max_size: int) -> int:
    """80% of 2 bits per parameter at a vocabulary of 1000."""
    vocab = 1000
    n_params = 2 * vocab * dim + layers * 12 * dim ** 2
    facts = 2.0 * n_params / math.log2(vocab)
    return max(coarse_step, min(round_to(int(facts * 0.8), coarse_step), max_size))


@functools.lru_cache(maxsize=None)
def _gpu_memory_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def micro_batch_size(setup: Setup, num_entities: int, size: int) -> int:
    vocab = num_entities + setup.num_relations
    n_params = 2 * vocab * setup.dim + setup.layers * 12 * setup.dim ** 2
    free = int(_gpu_memory_gb() * 0.85 * 1024 ** 3) - 16 * n_params - 24 * size
    free = max(free, 1024 ** 3)
    return max(1000, min(free // (vocab * 12), 60000, size))


def train_config(setup: Setup, hp: Hparams, seed: int, size: int) -> TrainConfig:
    num_entities = size // setup.num_relations
    return TrainConfig(
        family=setup.family, num_entities=num_entities,
        num_relations=setup.num_relations, dim=setup.dim, layers=setup.layers,
        seed=seed, lr=hp.lr, weight_decay=hp.weight_decay,
        warmup_steps=setup.warmup_steps, lr_decay_type=hp.lr_decay_type,
        min_lr=hp.min_lr, decay_end_ratio=hp.decay_end_ratio,
        max_batches=MAX_BATCHES,
        clip_norm=setup.clip_norm, alpha=setup.alpha,
        micro_batch_size=(micro_batch_size(setup, num_entities, size)
                          if size > MICRO_BATCH_ABOVE else None))


def cache_key(cfg: TrainConfig) -> dict:
    key = asdict(cfg)
    del key["micro_batch_size"]
    return key


class ProbeCache:
    """One JSON file per finished training run, written atomically."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: dict) -> str:
        blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
        return os.path.join(self.directory, hashlib.sha256(blob.encode()).hexdigest()[:32] + ".json")

    def load(self, key: dict) -> dict | None:
        try:
            with open(self._path(key)) as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        return record.get("result") if record.get("key") == key else None

    def store(self, key: dict, result: dict) -> None:
        tmp = os.path.join(self.directory, f".tmp_{uuid.uuid4().hex}")
        try:
            with open(tmp, "w") as f:
                json.dump({"key": key, "result": result}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path(key))
        except OSError as e:
            print(f"probe cache write failed: {e}", flush=True)


def _is_oom(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


def _train_oom_fails_run(cfg: TrainConfig) -> tuple[bool, int | None]:
    """Train, retrying once after an out-of-memory error; a second one counts as not converged."""
    for attempt in range(2):
        try:
            result = train.train(cfg)
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            if attempt == 0:
                time.sleep(2)
            continue
        torch.cuda.empty_cache()
        return result.converged, result.steps_to_100
    return False, None


def probe(setup: Setup, cache: ProbeCache | None, hp: Hparams, seed: int,
          size: int) -> tuple[bool, int | None]:
    """Train once on `size` facts, or read the cached run. Returns (converged, steps_to_100)."""
    cfg = train_config(setup, hp, seed, size)
    key = cache_key(cfg)
    hit = cache.load(key) if cache else None
    if hit is not None:
        return hit["converged"], hit["batches100"]
    converged, steps = _train_oom_fails_run(cfg)
    if cache:
        cache.store(key, {"converged": converged, "batches100": steps})
    return converged, steps


def max_size_for_seed(setup: Setup, cache: ProbeCache | None, hp: Hparams,
                      seed: int) -> tuple[int | None, int | None]:
    """(max size, steps_to_100 there) on the fine grid, or (None, None)."""
    coarse, fine = setup.coarse_step, setup.fine_step
    tested: dict[int, tuple[bool, int | None]] = {}

    def passes(size: int) -> tuple[bool, int | None]:
        if size not in tested:
            tested[size] = probe(setup, cache, hp, seed, size)
        return tested[size]

    start = setup.start_size
    converged, steps = passes(start)
    if converged:
        ok, ok_steps = start, steps
        size = round_to(start * 2, coarse)
        while size <= setup.max_size:
            converged, steps = passes(size)
            if not converged:
                break
            ok, ok_steps = size, steps
            size = round_to(size * 2, coarse)
        else:
            return ok, ok_steps
        fail = size
    else:
        ok = ok_steps = None
        fail = start
        size = round_to(start // 2, coarse)
        while size >= fine:
            converged, steps = passes(size)
            if converged:
                ok, ok_steps = size, steps
                break
            fail = size
            half = size // 2
            if half < fine:
                break
            size = max(fine, round_to(half, coarse if half >= coarse else fine))
            if size >= fail:
                size = fail - fine
                if size < fine:
                    break
        if ok is None:
            return None, None

    lo = (ok // fine + 1) * fine
    hi = ((fail - 1) // fine) * fine
    while lo <= hi:
        mid = max(lo, min(hi, round_to((lo + hi) // 2, fine)))
        converged, steps = passes(mid)
        if converged:
            ok, ok_steps = mid, steps
            lo = mid + fine
        else:
            hi = mid - fine
    return ok, ok_steps


def max_size_all_seeds(setup: Setup, cache: ProbeCache | None, hp: Hparams) -> list[tuple]:
    """Run the size search for every seed in parallel, one thread per seed, spread over the GPUs."""
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def run(index_seed: tuple[int, int]) -> tuple:
        index, seed = index_seed
        if n_gpus > 0:
            torch.cuda.set_device(index % n_gpus)
        return max_size_for_seed(setup, cache, hp, seed)

    with ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        return list(pool.map(run, enumerate(SEEDS)))


def objective_value(per_seed_size: dict[int, int | None], coarse_step: int) -> float:
    """max_i N_i - 0.1 coarse_step * (failed seeds) - 0.05 * sum_i (max_j N_j - N_i); 0 if all fail."""
    sizes = [s for s in per_seed_size.values() if s is not None]
    if not sizes:
        return 0.0
    best = max(sizes)
    n_failed = len(per_seed_size) - len(sizes)
    spread = sum(best - s for s in sizes)
    return best - 0.1 * n_failed * coarse_step - 0.05 * spread


def n_startup_trials(decay_type: str) -> int:
    return STARTUP_TRIALS_PER_PARAM * 6


def sample_hparams(trial: optuna.Trial, decay_type: str) -> Hparams:
    lr = trial.suggest_float("lr", *LR_RANGE, log=True)
    min_lr = lr * trial.suggest_float("min_lr_ratio", *MIN_LR_RATIO_RANGE, log=True)
    end = trial.suggest_int("decay_end_step", DECAY_STEP_MIN, MAX_BATCHES)
    decay_end_ratio = min(1.0, end / MAX_BATCHES)
    if trial.suggest_categorical("use_weight_decay", [False, True]):
        weight_decay = trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True)
    else:
        weight_decay = 0.0
    return Hparams(lr, weight_decay, decay_type, min_lr, decay_end_ratio)


def first_trial_params(start_size: int) -> dict:
    return {"lr": 1.5 / math.sqrt(start_size), "use_weight_decay": True,
            "weight_decay": 4e-5, "min_lr_ratio": 0.01, "decay_end_step": 20000}


def _trials(study: optuna.Study, *states: TrialState) -> list[optuna.trial.FrozenTrial]:
    return study.get_trials(deepcopy=False, states=states)


def trials_since_best(trials: list[optuna.trial.FrozenTrial]) -> int:
    values = [t.value for t in trials]
    best = max(values)
    return len(values) - max(i for i, v in enumerate(values) if v == best) - 1


class EarlyStopping:
    """Stop after min_trials COMPLETE trials once the best value is patience trials old."""

    def __init__(self, done: list[optuna.trial.FrozenTrial], min_trials: int, patience: int):
        self.min_trials = min_trials
        self.patience = patience
        self.best = max(t.value for t in done) if done else None
        self.since_best = trials_since_best(done) if done else 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        if self.best is None or study.best_value > self.best:
            self.best = study.best_value
            self.since_best = 0
        else:
            self.since_best += 1
        if len(_trials(study, TrialState.COMPLETE)) >= self.min_trials and self.since_best >= self.patience:
            study.stop()


def requeue_orphans(study: optuna.Study, marker_dir: str, max_trials: int,
                    stale_hours: float) -> None:
    """Re-enqueue the params of RUNNING trials older than stale_hours, once across workers."""
    os.makedirs(marker_dir, exist_ok=True)
    now = datetime.datetime.now()
    budget_left = max_trials - len(_trials(study, TrialState.COMPLETE))
    for t in _trials(study, TrialState.RUNNING):
        if budget_left <= 0:
            break
        if t.datetime_start is None or not t.params:
            continue
        if (now - t.datetime_start).total_seconds() < stale_hours * 3600:
            continue
        signature = hashlib.sha256(json.dumps(
            [t.number, str(t.datetime_start), sorted(t.params.items())],
            default=str, sort_keys=True).encode()).hexdigest()[:16]
        try:
            os.close(os.open(os.path.join(marker_dir, f"t{t.number}_{signature}.enqueued"),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        study.enqueue_trial(dict(t.params))
        budget_left -= 1


def make_objective(setup: Setup, cache: ProbeCache | None, decay_type: str):
    def objective(trial: optuna.Trial) -> float:
        hp = sample_hparams(trial, decay_type)
        t0 = time.time()
        results = max_size_all_seeds(setup, cache, hp)
        per_seed_size = {s: r[0] for s, r in zip(SEEDS, results)}
        trial.set_user_attr("per_seed_max_size", per_seed_size)
        trial.set_user_attr("per_seed_batches100", {s: r[1] for s, r in zip(SEEDS, results)})
        trial.set_user_attr("found_any", any(s is not None for s in per_seed_size.values()))
        trial.set_user_attr("trial_wall_s", round(time.time() - t0, 1))
        return objective_value(per_seed_size, setup.coarse_step)
    return objective


def run_study(setup: Setup, decay_type: str, db_dir: str, cache_dir: str | None,
              min_trials: int, max_trials: int, orphan_stale_hours: float) -> optuna.Study:
    """Create or resume one study and run it until early stopping or max_trials COMPLETE trials."""
    name = study_name(setup, decay_type)
    os.makedirs(db_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(db_dir, name + '.db')}"
    cache = ProbeCache(os.path.join(cache_dir, name)) if cache_dir else None
    worker = int(os.environ.get("SLURM_JOB_ID", os.getpid()))
    # hash() of a str is salted per process unless PYTHONHASHSEED is set.
    sampler = TPESampler(
        seed=(SAMPLER_SEED + hash(decay_type) % 1000 + worker * 1_000_003) % 2 ** 31,
        n_startup_trials=n_startup_trials(decay_type), multivariate=True, group=True)
    objective = make_objective(setup, cache, decay_type)

    try:
        study = optuna.load_study(study_name=name, storage=storage, sampler=sampler)
        if cache:
            requeue_orphans(study, os.path.join(cache.directory, "orphans"),
                            max_trials, orphan_stale_hours)
        done = _trials(study, TrialState.COMPLETE)
    except KeyError:
        study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                    sampler=sampler, load_if_exists=True)
        if not study.trials:
            study.enqueue_trial(first_trial_params(setup.start_size))
        done = []
    early_stop = EarlyStopping(done, min_trials, PATIENCE)

    while True:
        n_complete = len(_trials(study, TrialState.COMPLETE))
        n_settled = sum(t.state != TrialState.RUNNING for t in study.trials)
        n_failed = n_settled - n_complete - len(_trials(study, TrialState.PRUNED))
        if early_stop.since_best >= PATIENCE and n_complete >= min_trials:
            break
        if n_failed >= FAIL_ALLOWANCE:
            raise RuntimeError(f"{n_failed} failed trials vs {n_complete} complete")
        n_left = min(max_trials - n_complete, max_trials + FAIL_ALLOWANCE - n_settled)
        if n_left <= 0:
            break
        study.optimize(objective, n_trials=n_left, callbacks=[early_stop], catch=(Exception,))
    return study


def report(study: optuna.Study) -> None:
    completed = _trials(study, TrialState.COMPLETE)
    if not completed:
        print(f"{study.study_name}: no completed trials", flush=True)
        return
    best = max(completed, key=lambda t: t.value or 0.0)
    sizes = best.user_attrs["per_seed_max_size"]
    n_max = max((s for s in sizes.values() if s is not None), default=None)
    print(f"{study.study_name}: best objective = {best.value:.0f} N_max = {n_max} "
          f"per seed = {sizes} params = {best.params}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, default="uniform")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--num-relations", type=int, default=3)
    parser.add_argument("--entity-step", type=int, default=334,
                        help="coarse step of the size search, in entities")
    parser.add_argument("--fine-entity-step", type=int, default=10,
                        help="binary-search step, in entities")
    parser.add_argument("--start-size", type=int, default=0,
                        help="first size tried, in facts; 0 = from the parameter count")
    parser.add_argument("--max-size", type=int, default=2_000_000)
    parser.add_argument("--lr-decay-types", nargs="+", choices=DECAY_TYPES, default=list(DECAY_TYPES))
    parser.add_argument("--min-trials", type=int, default=60)
    parser.add_argument("--max-trials", type=int, default=60)
    parser.add_argument("--warmup-cap", type=int, default=4000,
                        help="warmup = min(500 * layers, cap); 0 = no cap")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0, help="powerlaw exponent")
    parser.add_argument("--db-dir", default="studies")
    parser.add_argument("--cache-dir", default="probe_cache",
                        help="empty = no probe cache and no orphan requeue")
    parser.add_argument("--orphan-stale-hours", type=float, default=4.5,
                        help="RUNNING trials older than this are re-run")
    args = parser.parse_args(argv)
    if args.start_size % args.num_relations:
        parser.error("--start-size must be a multiple of --num-relations")

    coarse = args.entity_step * args.num_relations
    warmup = WARMUP_STEPS_PER_LAYER * args.layers
    if args.warmup_cap:
        warmup = min(warmup, args.warmup_cap)
    setup = Setup(
        family=args.family, dim=args.dim, layers=args.layers,
        num_relations=args.num_relations, coarse_step=coarse,
        fine_step=args.fine_entity_step * args.num_relations,
        start_size=args.start_size or initial_size(args.dim, args.layers, coarse, args.max_size),
        max_size=args.max_size, warmup_steps=warmup, clip_norm=args.clip_norm, alpha=args.alpha)
    for decay_type in args.lr_decay_types:
        report(run_study(setup, decay_type, args.db_dir, args.cache_dir or None,
                         args.min_trials, args.max_trials, args.orphan_stale_hours))


if __name__ == "__main__":
    main()
