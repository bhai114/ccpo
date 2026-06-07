"""绘制 GSPO 训练中 negative_approx_kl 的分布图，判断是否重尾 / 是否含离群噪声。

输入数据由 verl/trainer/ppo/core_algos.py 中 _save_negative_approx_kl_data 写入：
  save_gspo_kl_logs/
      stats.txt                          每 step 一行汇总
      raw_token_step_{N}.txt             (B, L)  negative_approx_kl * response_mask
      raw_mask_step_{N}.txt              (B, L)  1=valid token, 0=padding
      raw_seq_step_{N}.txt               (B,)    序列级均值

用法（默认参数已对齐脚本输出目录）：
    python3 plot_negative_approx_kl_dist.py \
        --log_dir ./save_gspo_kl_logs \
        --out_dir ./figs_negative_approx_kl \
        --steps 10 200 600 1200

不传 --steps 时脚本自动挑早 / 中 / 末三个 step。
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats


def _load_stats(stats_path: str) -> pd.DataFrame:
    df = pd.read_csv(stats_path, sep="\t")
    return df


def _list_raw_steps(log_dir: str) -> list[int]:
    pat = re.compile(r"raw_token_step_(\d+)\.txt$")
    out = []
    for p in glob(os.path.join(log_dir, "raw_token_step_*.txt")):
        m = pat.search(os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _load_valid_tokens(log_dir: str, step: int) -> np.ndarray:
    """读 raw_token + raw_mask，返回展开成 1D 的有效 token 值。"""
    token_path = os.path.join(log_dir, f"raw_token_step_{step}.txt")
    mask_path = os.path.join(log_dir, f"raw_mask_step_{step}.txt")
    if not (os.path.exists(token_path) and os.path.exists(mask_path)):
        raise FileNotFoundError(f"missing raw files for step {step}")

    tok = np.loadtxt(token_path)
    msk = np.loadtxt(mask_path)
    if tok.ndim == 1:
        tok = tok[None, :]
        msk = msk[None, :]
    valid = tok[msk > 0.5]
    return valid


def _tail_stats(x: np.ndarray) -> dict:
    """汇总尾部相关统计。"""
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 0:
        return {
            "n": int(x.size), "mean": mu, "std": sd,
            "min": float(x.min()), "max": float(x.max()),
            "skew": 0.0, "excess_kurt": 0.0,
            "frac_3sigma": 0.0, "frac_5sigma": 0.0, "frac_10sigma": 0.0,
        }
    z = (x - mu) / sd
    return {
        "n": int(x.size),
        "mean": mu,
        "std": sd,
        "min": float(x.min()),
        "max": float(x.max()),
        "skew": float(sstats.skew(x)),
        "excess_kurt": float(sstats.kurtosis(x, fisher=True)),  # Normal=0
        "frac_3sigma": float(np.mean(np.abs(z) > 3)),
        "frac_5sigma": float(np.mean(np.abs(z) > 5)),
        "frac_10sigma": float(np.mean(np.abs(z) > 10)),
    }


def plot_timeseries(stats_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(stats_df["step"], stats_df["mean"], label="mean", color="C0", lw=0.8)
    ax.fill_between(
        stats_df["step"],
        stats_df["mean"] - stats_df["std"],
        stats_df["mean"] + stats_df["std"],
        alpha=0.2, color="C0", label="±1 std",
    )
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("token-level\nmean ± std")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("negative_approx_kl statistics over training (from stats.txt)")

    ax = axes[1]
    ax.plot(stats_df["step"], stats_df["max"], color="C3", lw=0.6, label="max")
    ax.plot(stats_df["step"], stats_df["min"], color="C2", lw=0.6, label="min")
    ax.plot(stats_df["step"], stats_df["q75"], color="C1", lw=0.6, alpha=0.8, label="q75")
    ax.plot(stats_df["step"], stats_df["q25"], color="C4", lw=0.6, alpha=0.8, label="q25")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("min / q25 / q75 / max")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax = axes[2]
    sd = stats_df["std"].replace(0, np.nan)
    rng = (stats_df["max"] - stats_df["min"]) / sd
    ax.plot(stats_df["step"], rng, color="C5", lw=0.7, label="(max-min)/std")
    ax.axhline(6, color="k", lw=0.5, ls=":", label="Normal expected ≈ 6")
    ax.set_xlabel("step")
    ax.set_ylabel("range / std")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_per_step(values: np.ndarray, step: int, out_path: str) -> dict:
    """对单个 step 的 valid token 画 4 联图：linear hist / log-y hist / QQ / boxplot。"""
    s = _tail_stats(values)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title = (
        f"step {step}: n={s['n']}, mean={s['mean']:.3e}, std={s['std']:.3e}\n"
        f"min={s['min']:.3f}, max={s['max']:.3f}, "
        f"skew={s['skew']:.2f}, excess_kurt={s['excess_kurt']:.2f}\n"
        f"|z|>3σ: {s['frac_3sigma']*100:.3f}%   "
        f"|z|>5σ: {s['frac_5sigma']*100:.3f}%   "
        f"|z|>10σ: {s['frac_10sigma']*100:.3f}%"
    )
    fig.suptitle(title, fontsize=10)

    bins = 200

    ax = axes[0, 0]
    ax.hist(values, bins=bins, color="C0", alpha=0.85)
    ax.axvline(s["mean"], color="C3", lw=1, label=f"mean={s['mean']:.2e}")
    ax.axvline(np.median(values), color="C2", lw=1, ls="--", label=f"median={np.median(values):.2e}")
    ax.set_xlabel("negative_approx_kl (valid tokens)")
    ax.set_ylabel("count")
    ax.set_title("Histogram (linear y)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(values, bins=bins, color="C0", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(s["mean"], color="C3", lw=1)
    ax.axvline(np.median(values), color="C2", lw=1, ls="--")
    if s["std"] > 0:
        for k in (3, 5, 10):
            ax.axvline(s["mean"] + k * s["std"], color="k", lw=0.5, ls=":")
            ax.axvline(s["mean"] - k * s["std"], color="k", lw=0.5, ls=":")
    ax.set_xlabel("negative_approx_kl")
    ax.set_ylabel("count (log)")
    ax.set_title("Histogram (log y, ±3/5/10 σ marked)")

    ax = axes[1, 0]
    sample = values
    if sample.size > 200_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(sample.size, size=200_000, replace=False)
        sample = sample[idx]
    sstats.probplot(sample, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q plot")
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_alpha(0.5)

    ax = axes[1, 1]
    ax.boxplot(values, vert=True, showfliers=True,
               flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.set_yscale("symlog", linthresh=max(s["std"], 1e-6))
    ax.set_ylabel("negative_approx_kl (symlog)")
    ax.set_title("Boxplot (symlog)")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] step {step}: saved {out_path}")
    return s


def plot_overlay_log_pdfs(per_step_values: dict[int, np.ndarray], out_path: str) -> None:
    """在同一张 log-y 图上叠加多个 step 的归一化分布，便于观察重尾随训练的演化。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(
        min(v.min() for v in per_step_values.values()),
        max(v.max() for v in per_step_values.values()),
        300,
    )
    for i, (step, v) in enumerate(sorted(per_step_values.items())):
        hist, edges = np.histogram(v, bins=bins, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centers, hist + 1e-12, lw=1.0, alpha=0.85,
                label=f"step {step} (n={v.size})", color=f"C{i}")
    ax.set_yscale("log")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("negative_approx_kl")
    ax.set_ylabel("density (log)")
    ax.set_title("valid-token distribution across steps (log y, normalized density)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def _pick_default_steps(available: list[int], k: int = 4) -> list[int]:
    if not available:
        return []
    if len(available) <= k:
        return available
    idx = np.linspace(0, len(available) - 1, k).round().astype(int)
    return [available[i] for i in idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./save_gspo_kl_logs")
    parser.add_argument("--out_dir", default="./figs_negative_approx_kl")
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="要绘制的 step 列表，例如 --steps 10 200 600 1200")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    stats_path = os.path.join(args.log_dir, "stats.txt")
    if os.path.exists(stats_path):
        df = _load_stats(stats_path)
        plot_timeseries(df, os.path.join(args.out_dir, "timeseries.png"))

    available = _list_raw_steps(args.log_dir)
    if not available:
        print(f"[warn] no raw_token_step_*.txt under {args.log_dir}, skip per-step plots")
        return

    steps = args.steps or _pick_default_steps(available, k=4)
    steps = [s for s in steps if s in available]
    if not steps:
        print("[warn] none of the requested steps exist; falling back to default selection")
        steps = _pick_default_steps(available, k=4)

    summary_rows = []
    per_step_values: dict[int, np.ndarray] = {}
    for s in steps:
        v = _load_valid_tokens(args.log_dir, s)
        per_step_values[s] = v
        out_path = os.path.join(args.out_dir, f"step_{s:05d}.png")
        info = plot_per_step(v, s, out_path)
        info["step"] = s
        summary_rows.append(info)

    if len(per_step_values) >= 2:
        plot_overlay_log_pdfs(per_step_values, os.path.join(args.out_dir, "overlay_log_pdf.png"))

    summary_df = pd.DataFrame(summary_rows)[
        ["step", "n", "mean", "std", "min", "max", "skew", "excess_kurt",
         "frac_3sigma", "frac_5sigma", "frac_10sigma"]
    ]
    summary_path = os.path.join(args.out_dir, "tail_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[plot] tail summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
