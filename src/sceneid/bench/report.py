"""Writing benchmark results: summary.json, per_query.csv, report.md and plots."""

import csv
import json
import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HEADLINE_ROWS = [
    ("dir", "Identified correctly (DIR)", "pct"),
    ("far", "False accepts on unknown clips (FAR)", "pct"),
    ("frr", "Known clips rejected (FRR)", "pct"),
    ("misid", "Accepted as the wrong film", "pct"),
    ("top1", "Best candidate right, before the gate (top-1)", "pct"),
    ("offset_median_s", "Timestamp error, median", "sec"),
    ("offset_within_1s", "Timestamp within 1 s", "pct"),
]


def git_revision() -> str | None:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _mb(x: float) -> str:
    return f"{x:.0f} MB" if x >= 10 else f"{x:.1f} MB"


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _cell(metrics: dict, key: str, kind: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    text = _pct(value) if kind == "pct" else f"{value:.2f} s"
    ci = metrics.get("ci95", {}).get(key)
    if ci and ci[0] is not None:
        text += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if kind == "pct" else ""
    return text


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _breakdown_table(runs: list[dict], key: str, label: str, fmt=str) -> str:
    values = list(runs[0][key])
    header = [label] + [f"{r['name']} DIR / FAR" for r in runs]
    rows = []
    for v in values:
        row = [fmt(v)]
        for r in runs:
            m = r[key].get(v, {})
            row.append(f"{_pct(m.get('dir'))} / {_pct(m.get('far'))}")
        rows.append(row)
    return _table(header, rows)


def write_results(
    out_dir: str | Path, summary: dict, runs: list[dict], per_query: list[dict]
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_runs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    full = {**summary, "runs": clean_runs}
    (out_dir / "summary.json").write_text(
        json.dumps(full, indent=2, default=_json_default), encoding="utf-8"
    )

    with open(out_dir / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)

    (out_dir / "report.md").write_text(render_markdown(full), encoding="utf-8")
    _plots(out_dir, clean_runs)
    log.info("bench.results_written", extra={"dir": str(out_dir)})
    return out_dir


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def render_markdown(s: dict) -> str:
    runs = s["runs"]
    lib = s["library"]
    q = s["queries"]
    test = runs[0]["test"]
    target = next(
        (r["selection"]["target_far"] for r in runs if r["selection"]["target_far"]), None
    )
    made_on = s["embeddings"].get("machine", {})
    device = made_on.get("gpu") or f"CPU ({made_on.get('cpu_count', '?')} threads)"
    lines = [
        f"# Benchmark: {s['manifest']['name']} · {s['embeddings']['embedder']}",
        "",
        f"{s['created_at'][:10]} · sceneid {s['sceneid']} · commit `{s['git'] or 'unknown'}` · "
        f"manifest sha256 `{s['manifest']['digest'][:12]}` · embeddings made on {device}",
        "",
        f"**Setup.** {lib['n_videos']} indexed films ({lib['hours']:.1f} h, {lib['n_vectors']:,} "
        f"vectors at {lib['index_fps']:g} fps, {_mb(lib['index_mb'])} index) and "
        f"{s['manifest']['n_heldout']} held-out films. {q['n_queries']:,} queries: "
        f"{q['n_base']} base clips × {q['n_distortions']} distortions; {q['n_known']:,} known "
        f"and {q['n_unknown']:,} unknown. {q['n_failed']} failed to render.",
        "",
        (
            f"Tuned thresholds were chosen on the val split for FAR ≤ {_pct(target)}. "
            if target
            else ""
        )
        + f"Every number below is the **test split** ({test['n_known']:,} known, "
        f"{test['n_unknown']:,} unknown clips). Brackets are 95% bootstrap intervals over base "
        "clips.",
        "",
        "## Headline",
        "",
    ]
    rows = [
        [label] + [_cell(r["test"], key, kind) for r in runs] for key, label, kind in HEADLINE_ROWS
    ]
    rows.append(
        ["Thresholds"] + [", ".join(f"{k}={v:g}" for k, v in r["thresholds"].items()) for r in runs]
    )
    lines += [_table([""] + [r["name"] for r in runs], rows), ""]
    if any(r["selection"]["val_target_met"] is False for r in runs):
        lines += [
            "> Some tuned runs could not reach the FAR target on val; they use the lowest-FAR "
            "thresholds instead.",
            "",
        ]
    lines += [
        "## By distortion",
        "",
        "DIR on known clips / FAR on unknown clips, per distortion.",
        "",
        _breakdown_table(runs, "by_distortion", "Distortion"),
        "",
        "## By clip length",
        "",
        _breakdown_table(runs, "by_length", "Length", fmt=lambda v: f"{float(v):g} s"),
        "",
        "## Unknown clips: false accepts by look-alike group",
        "",
        _table(
            ["Group"] + [f"{r['name']} FAR" for r in runs],
            [
                [g] + [_pct(r["unknown_by_group"][g].get("far")) for r in runs]
                for g in runs[0]["unknown_by_group"]
            ],
        ),
        "",
        "## Known clips by film",
        "",
        _table(
            ["Film"] + [f"{r['name']} DIR" for r in runs],
            [
                [f] + [_pct(r["by_film"][f].get("dir")) for r in runs]
                for f in runs[0]["by_film"]
                if runs[0]["by_film"][f]["n_known"]
            ],
        ),
        "",
        "## Speed and size",
        "",
        _table(
            ["Stage (per query)", "p50", "p95", "Measured on"],
            [
                [
                    "Decode frames",
                    f"{s['latency']['decode_ms'][0]:.0f} ms",
                    f"{s['latency']['decode_ms'][1]:.0f} ms",
                    device,
                ],
                [
                    "Embed frames (batched across clips)",
                    f"{s['latency']['embed_ms'][0]:.0f} ms",
                    f"{s['latency']['embed_ms'][1]:.0f} ms",
                    device,
                ],
                [
                    "Search (one query, exact)",
                    f"{s['latency']['search']['p50_ms']:.1f} ms",
                    f"{s['latency']['search']['p95_ms']:.1f} ms",
                    s["evaluated_on"].get("platform", "?"),
                ],
            ]
            + [
                [
                    f"Decide ({r['name']})",
                    f"{r['decide_ms_p50']:.2f} ms",
                    f"{r['decide_ms_p95']:.2f} ms",
                    s["evaluated_on"].get("platform", "?"),
                ]
                for r in runs
            ],
        ),
        "",
        f"Index: {lib['n_vectors']:,} vectors × {lib['dim']} dims = {_mb(lib['index_mb'])} "
        f"(exact inner-product search).",
        "",
    ]
    return "\n".join(lines)


def _plots(out_dir: Path, runs: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("bench.plots_skipped", extra={"reason": "matplotlib not installed"})
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for r in runs:
        pts = sorted(
            (p["far"], p["dir"])
            for p in r["roc_test"]
            if p["far"] is not None and p["dir"] is not None
        )
        if pts:
            far, dir_ = zip(*pts, strict=True)
            (line,) = ax.plot(np.maximum(far, 1e-4), dir_, label=r["name"], lw=1.8)
            t = r["test"]
            if t.get("far") is not None and t.get("dir") is not None:
                ax.scatter(
                    [max(t["far"], 1e-4)], [t["dir"]], color=line.get_color(), zorder=3, s=30
                )
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False accept rate on unknown clips (log scale)")
    ax.set_ylabel("Identified correctly (DIR)")
    ax.set_title("Open-set ROC, test split (dots: chosen thresholds)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)
