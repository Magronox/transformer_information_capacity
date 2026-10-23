"""Every fit and lack-of-fit F-test in the paper, from the CSVs written by export_results.py."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

SEEDS = ("10", "20", "30")
MAX_BATCHES = 40_000


def read_csv(name: str, results: Path) -> list[dict[str, str]]:
    with (results / name).open() as f:
        return list(csv.DictReader(f))


def column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def seeds(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([[float(r[f"{name}_seed_{s}"]) for s in SEEDS] for r in rows])


@dataclass(frozen=True)
class Form:
    name: str
    f: Callable[..., np.ndarray]
    starts: tuple[tuple[float, ...], ...]
    bounds: tuple[list[float], list[float]] | None
    show: Callable[[np.ndarray], str]


@dataclass(frozen=True)
class Fit:
    form: Form
    params: np.ndarray
    r2: float
    p: float

    def __str__(self) -> str:
        return f"{self.form.name:<28} {self.form.show(self.params):<40} R2 = {self.r2:.3f}   p = {pval(self.p)}"


def pval(p: float) -> str:
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def lack_of_fit_p(sum_sq_means: float, pure_error: float, m: int, n: int, k: int) -> float:
    """p-value of the lack-of-fit F-test: m points, n seeds per point, k fitted parameters."""
    stat = (n * sum_sq_means / (m - k)) / (pure_error / (m * (n - 1)))
    # 1 - cdf, not sf: the reported p-values floor at 4.4e-16.
    return 1 - f_dist.cdf(stat, m - k, m * (n - 1))


def log_fit(x: np.ndarray, y: np.ndarray, form: Form) -> Fit:
    """Least-squares fit of log f(x) to the seed means of log y, with its lack-of-fit F-test."""
    logs = np.log(y)
    means = logs.mean(1)
    g = lambda xx, *t: np.log(np.clip(form.f(xx, *t), 1e-9, None))
    kwargs = {"bounds": form.bounds} if form.bounds else {}
    best = None
    for p0 in form.starts:
        try:
            params, _ = curve_fit(g, x, means, p0=p0, maxfev=800_000, **kwargs)
        except (RuntimeError, ValueError):
            continue
        ss = float(((means - g(x, *params)) ** 2).sum())
        if best is None or ss < best[1]:
            best = (params, ss)
    params, ss = best
    m, n = y.shape
    r2 = 1 - ss / float(((means - means.mean()) ** 2).sum())
    pure_error = float(((logs - means[:, None]) ** 2).sum())
    return Fit(form, params, r2, lack_of_fit_p(ss, pure_error, m, n, len(params)))


def log_offset_label(p: np.ndarray) -> str:
    a, c, d0 = p
    return f"d = {a:.3f} ln(N + {c:.1f}) {d0 - a * np.log(c):+.2f}"


WIDTH_FORMS = (
    Form("Logarithmic + offset", lambda x, a, c, d0: a * np.log1p(x / c) + d0,
         ((2.7, 80, 2), (1, 10, 1), (3, 300, 0)), ([1e-9, 1e-9, 0], [50, 1e6, 50]), log_offset_label),
    Form("Logarithmic", lambda x, a, b: a * np.log(x) + b, ((2.2, -5),), None,
         lambda p: f"d = {p[0]:.2f} ln N {p[1]:+.2f}"),
    Form("Power-law + offset", lambda x, a, b, c: a * x ** b + c, ((1.4, 0.26, 0), (3, 0.15, 1)),
         ([1e-9, 0, 0], [50, 2, 50]), lambda p: f"d = {p[0]:.2f} N^{p[1]:.3f} + {p[2]:.2f}"),
    Form("Power-law", lambda x, a, b: a * x ** b, ((1.4, 0.26),), ([1e-9, 0], [50, 2]),
         lambda p: f"d = {p[0]:.2f} N^{p[1]:.3f}"),
    Form("Linear", lambda x, a, b: a * x + b, ((5e-4, 6),), ([0, 0], [1, 50]),
         lambda p: f"d = {p[0]:.2e} N {p[1]:+.2f}"),
)


def width_fits(n: np.ndarray, d: np.ndarray) -> list[Fit]:
    return [log_fit(n, d, form) for form in WIDTH_FORMS]


def _inv_log_offset(d, a, c, d0): return c * (np.exp((d - d0) / a) - 1)
def _inv_log(d, a, b): return np.exp((d - b) / a)
def _inv_power_offset(d, a, b, c): return (np.clip(d - c, 1e-9, None) / a) ** (1 / b)
def _inv_power(d, a, b): return (d / a) ** (1 / b)
def _inv_linear(d, a, b): return np.clip(d - b, 1e-9, None) / a


MAX_SIZE_FORMS = (
    Form("Logarithmic + offset", _inv_log_offset, ((2.8, 80, 2), (3, 126, 1), (2.5, 40, 3)),
         ([1e-6, 1e-6, 0], [50, 1e6, 50]), log_offset_label),
    Form("Logarithmic", _inv_log, ((2, -5),), ([1e-6, -100], [50, 100]), lambda p: f"d = {p[0]:.3f} ln N {p[1]:+.2f}"),
    Form("Power-law + offset", _inv_power_offset, ((1.4, 0.26, 0), (1, 0.3, 1)), ([1e-6, 1e-3, 0], [50, 3, 2.99]),
         lambda p: f"d = {p[0]:.2f} N^{p[1]:.3f} + {p[2]:.2f}"),
    Form("Power-law", _inv_power, ((1.4, 0.26),), ([1e-6, 1e-3], [50, 3]), lambda p: f"d = {p[0]:.2f} N^{p[1]:.3f}"),
    Form("Linear", _inv_linear, ((1e-3, 2),), ([1e-9, -100], [1, 2.99]), lambda p: f"d = {p[0]:.2e} N {p[1]:+.2f}"),
)


DEPTH_FORMS = (
    Form("Linear without intercept", lambda l, a: a * l, ((600,),), None, lambda p: f"N = {p[0]:.0f} L"),
    Form("Linear with intercept", lambda l, a, b: a * l + b, ((400, 800),), None,
         lambda p: f"N = {p[0]:.0f} L + {p[1]:.0f}"),
    Form("Power-law, exponent >= 1", lambda l, a, b: a * l ** b, ((600, 1.0),), ([0, 1.0], [1e6, 10]),
         lambda p: f"N = {p[0]:.0f} L^{p[1]:.3f}"),
    Form("Exponential", lambda l, a, k: a * np.exp(k * l), ((1200, 0.1),), ([0, 0], [1e6, 5]),
         lambda p: f"N = {p[0]:.0f} exp({p[1]:.3f} L)"),
    Form("Logarithmic", lambda l, a, b: a * np.log(l) + b, ((1300, 1000),), None,
         lambda p: f"N = {p[0]:.0f} ln L + {p[1]:.0f}"),
    Form("Power-law", lambda l, a, b: a * l ** b, ((1000, 0.6),), None, lambda p: f"N = {p[0]:.0f} L^{p[1]:.3f}"),
)


BITS_FORMS = (
    Form("Logarithmic a + b ln N", lambda x, a, b: a + b * np.log(x), ((0.4, 0.2),), None,
         lambda p: f"{p[0]:.3f} + {p[1]:.3f} ln N"),
    Form("Power-law a N^b", lambda x, a, b: a * x ** b, ((0.9, 0.1),), None, lambda p: f"{p[0]:.3f} N^{p[1]:.3f}"),
    Form("Saturating a - b N^-c", lambda x, a, b, c: a - b * x ** (-c), ((2.3, 10, 0.5),), ([0, 0, 0], [10, 1000, 5]),
         lambda p: f"{p[0]:.2f} - {p[1]:.2f} N^-{p[2]:.3f}"),
    Form("Saturating a N/(N + k)", lambda x, a, k: a * x / (x + k), ((2.2, 80),), None,
         lambda p: f"{p[0]:.2f} N/(N + {p[1]:.1f})"),
)


def body_params(d: np.ndarray) -> np.ndarray:
    return 12 * d ** 2 + 15 * d


def embedding_params(entities: np.ndarray, relations: np.ndarray, d: np.ndarray) -> np.ndarray:
    return (entities + relations) * d


def dataset_bits(facts: np.ndarray, entities: np.ndarray) -> np.ndarray:
    return facts * np.log2(entities)


def bits_per_param(facts: np.ndarray, entities: np.ndarray, relations: np.ndarray,
                   d: np.ndarray) -> np.ndarray:
    return dataset_bits(facts, entities) / (body_params(d) + embedding_params(entities, relations, d))


@dataclass(frozen=True)
class JointFit:
    params: np.ndarray
    r2: float
    p: float


def joint_relations_fit(entities: np.ndarray, relations: np.ndarray, d: np.ndarray) -> JointFit:
    """d = a ln n + b ln m + c fitted to the seed means, with its lack-of-fit F-test."""
    means = d.mean(1)
    design = np.column_stack([np.log(entities), np.log(relations), np.ones_like(means)])
    params = np.linalg.lstsq(design, means, rcond=None)[0]
    ss = float(((means - design @ params) ** 2).sum())
    r2 = 1 - ss / float(((means - means.mean()) ** 2).sum())
    m, n = d.shape
    return JointFit(params, r2, lack_of_fit_p(ss, float(((d - means[:, None]) ** 2).sum()), m, n, 3))


def width_data(family: str, results: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = [r for r in read_csv("width_min_dim.csv", results) if r["family"] == family]
    return rows, column(rows, "facts"), seeds(rows, "min_dim")


def depth_data(results: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = read_csv("depth_max_size.csv", results)
    return rows, column(rows, "layers"), seeds(rows, "max_size")


def width_law(results: Path) -> Fit:
    _, n, d = width_data("uniform", results)
    return log_fit(n, d, WIDTH_FORMS[0])


def depth_law(results: Path) -> dict[str, Fit]:
    _, layers, sizes = depth_data(results)
    return {form.name: log_fit(layers, sizes, form) for form in DEPTH_FORMS}


def width_law_inverse(fit: Fit, d: np.ndarray) -> np.ndarray:
    a, c, d0 = fit.params
    a, c, b = round(a, 3), round(c, 1), round(d0 - a * np.log(c), 2)
    return np.exp((d - b) / a) - c


def section(title: str) -> None:
    print(f"\n## {title}\n")


def report_width(results: Path) -> None:
    rows, n, d = width_data("uniform", results)
    section(f"tab:width-fits (Uniform, {len(n)} sizes)")
    for fit in width_fits(n, d):
        print(fit)
    section("sec:exp-width: the same fits on truncated size ranges (lack-of-fit p)")
    for size in (6240, 12480):
        keep = n <= size
        print(f"N <= {size:>6}: " + "   ".join(f"{fit.form.name} {pval(fit.p)}" for fit in width_fits(n[keep], d[keep])))
    section("tab:bits-split (at the mean minimum dimension)")
    e, r, mean = column(rows, "entities"), column(rows, "relations"), d.mean(1)
    body, total = body_params(mean), body_params(mean) + embedding_params(e, r, mean)
    bits = dataset_bits(n, e)
    print(f"{'N':>7} {'d_min':>6} {'body %':>7} {'bits/P_body':>12} {'bits/P_tot':>11}")
    for row in zip(n, mean, 100 * body / total, bits / body, bits / total):
        print(f"{row[0]:>7.0f} {row[1]:>6.2f} {row[2]:>6.0f}% {row[3]:>12.1f} {row[4]:>11.2f}")
    p_max = body_params(d[-1].mean())
    print(f"\nnon-embedding parameters at N = {n[-1]:.0f}: {p_max:.0f};  sqrt(N) = {np.sqrt(n[-1]):.0f}")
    section("sec:analysis: bits per parameter vs N (per seed, all parameters)")
    per_seed = bits_per_param(n[:, None], e[:, None], r[:, None], d)
    for form in BITS_FORMS:
        print(log_fit(n, per_seed, form))


def report_depth(results: Path) -> None:
    rows, layers, sizes = depth_data(results)
    section(f"tab:depth-fits (d = 8, {len(layers)} depths)")
    print(f"{'L':>3} {'schedule':>8} {'trial':>5} {'mean':>6} {'smallest':>8} {'largest':>8}")
    for row, l, s in zip(rows, layers, sizes):
        print(f"{l:>3.0f} {row['schedule']:>8} {row['trial']:>5} {s.mean():>6.0f} {s.min():>8.0f} {s.max():>8.0f}")
    section("tab:depth-forms")
    for fit in depth_law(results).values():
        print(fit)
    section("sec:exp-depth: median convergence step of the reported runs, as a fraction of warmup + 40,000")
    steps = seeds(rows, "steps_to_100")
    budget = np.minimum(500 * layers, 4000) + MAX_BATCHES
    print("  ".join(f"L={l:.0f}: {np.median(s / b):.0%}" for l, s, b in zip(layers, steps, budget)))
    runs = read_csv("depth_training_runs.csv", results)
    section("sec:exp-depth: training runs that reach 100% but do not hold it (both schedules)")
    for l in dict.fromkeys(r["layers"] for r in runs):
        sub = [r for r in runs if r["layers"] == l]
        unstable, total = sum(int(r["reached_but_not_held"]) for r in sub), sum(int(r["runs"]) for r in sub)
        print(f"L={l}: {unstable}/{total} = {100 * unstable / total:.1f}%")


def report_max_size(results: Path) -> None:
    rows = read_csv("width_max_size.csv", results)
    dims, sizes = column(rows, "dim"), seeds(rows, "max_size")
    predicted = width_law_inverse(width_law(results), dims)
    section("tab:maxn-width (N_max at L = 1 vs the N predicted by the width law)")
    print(f"{'d':>3} {'measured':>9} {'width law':>9} {'difference':>10}")
    for d, s, p in zip(dims, sizes.mean(1), predicted):
        print(f"{d:>3.0f} {s:>9.0f} {p:>9.0f} {round(100 * (s / p - 1)):>+9d}%")
    section("tab:maxn-width-fits")
    for form in MAX_SIZE_FORMS:
        print(log_fit(dims, sizes, form))


def report_relations(results: Path) -> None:
    sweep = read_csv("relations_sweep.csv", results)
    width, _, _ = width_data("uniform", results)
    d = seeds(sweep, "min_dim")
    section("tab:er-sweep (N = 9,216)")
    print("entities  " + " ".join(f"{r['entities']:>6}" for r in sweep))
    print("relations " + " ".join(f"{r['relations']:>6}" for r in sweep))
    print("mean d    " + " ".join(f"{x:>6.2f}" for x in d.mean(1)))
    vocab = column(sweep, "entities") + column(sweep, "relations")
    print(f"vocabulary shrinks {vocab.max() / vocab.min():.0f}-fold")
    width_keys = {(w["entities"], w["relations"]) for w in width}
    rows = width + [r for r in sweep if (r["entities"], r["relations"]) not in width_keys]
    fit = joint_relations_fit(column(rows, "entities"), column(rows, "relations"), seeds(rows, "min_dim"))
    a, b, c = fit.params
    section(f"sec:exp-sweep: joint fit on {len(rows)} datasets (linear space)")
    print(f"d = {a:.2f} ln n + {b:.2f} ln m {c:+.2f}   R2 = {fit.r2:.3f}   p = {pval(fit.p)}")


def report_olmo3(results: Path) -> None:
    rows = read_csv("olmo3_min_dim.csv", results)
    _, _, gpt2 = width_data("uniform", results)
    n, d = column(rows, "facts"), seeds(rows, "min_dim")
    section("tab:olmo3-results-detailed and tab:olmo-width")
    print(f"{'n':>5} {'N':>6} {'bits':>7} {'seeds':>9} {'mean':>5} {'best':>4} {'GPT-2 mean':>10}")
    for row, x, s, g in zip(rows, n, d, gpt2):
        bits = dataset_bits(x, float(row["entities"]))
        print(f"{row['entities']:>5} {x:>6.0f} {bits:>7,.0f} {' / '.join(f'{v:.0f}' for v in s):>9} "
              f"{s.mean():>5.2f} {s.min():>4.0f} {g.mean():>10.2f}")
    section("sec:exp-olmo: OLMo-3 width fits")
    for fit in width_fits(n, d):
        print(fit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="directory with the CSVs written by export_results.py")
    args = parser.parse_args()
    for report in (report_width, report_depth, report_max_size, report_relations, report_olmo3):
        report(args.results)


if __name__ == "__main__":
    main()
