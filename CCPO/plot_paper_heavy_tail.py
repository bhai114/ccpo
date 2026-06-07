"""绘制单张论文级复合图，证明 negative_approx_kl 是重尾分布。

论文图 = 2x2 panel：
  (a) log-y histogram + 匹配 mu/sigma 的 Normal PDF 包络
  (b) Normal Q-Q plot（重尾会显著偏离 y=x）
  (c) Survival function P(|Z|>t) on log-log（核心证据：实测曲线远高于正态参考）
  (d) Excess kurtosis & frac(|z|>5sigma) 随训练 step 的演化

输入数据来自 verl/trainer/ppo/core_algos.py:_save_negative_approx_kl_data 写入的
save_gspo_kl_logs/{stats.txt, raw_token_step_*.txt, raw_mask_step_*.txt}.

用法：
    python3 plot_paper_heavy_tail.py \
        --log_dir ./save_gspo_kl_logs \
        --out_path ./figs_negative_approx_kl/heavy_tail_paper.png \
        --steps 10 1340 2670 4000
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


# ----- 数据读取 -----

def _list_raw_steps(log_dir: str) -> list[int]:
    pat = re.compile(r"raw_token_step_(\d+)\.txt$")
    out = []
    for p in glob(os.path.join(log_dir, "raw_token_step_*.txt")):
        m = pat.search(os.path.basename(p))
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _load_valid_tokens(log_dir: str, step: int) -> np.ndarray:
    tok = np.loadtxt(os.path.join(log_dir, f"raw_token_step_{step}.txt"))
    msk = np.loadtxt(os.path.join(log_dir, f"raw_mask_step_{step}.txt"))
    if tok.ndim == 1:
        tok = tok[None, :]
        msk = msk[None, :]
    return tok[msk > 0.5]


# ----- 4 个 panel -----

def _standardize(v: np.ndarray) -> np.ndarray:
    """经典 (x - mean) / std 标准化。

    与 N(0,1) 比对时，重尾分布会同时呈现两个特征：
      - z=0 附近的密度峰远高于 N(0,1)（因为 sigma 被 outlier 撑大、bulk 显得集中）
      - |z| > 3 的尾部密度远高于 N(0,1)
    这正是 leptokurtic / heavy-tailed 的"双重特征"。
    """
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd <= 0:
        sd = 1.0
    return (v - mu) / sd


def panel_loghist_with_normal(ax, samples_by_step: dict[int, np.ndarray], colors):
    """(a) 标准化后的密度 vs N(0,1) PDF（log y）。

    leptokurtic 的双重特征：z=0 附近的尖峰 + 远端的厚尾，都会在这一张图上同时体现。
    """
    bins = np.linspace(-30.0, 30.0, 1200)
    centers = 0.5 * (bins[1:] + bins[:-1])

    for (step, v), c in zip(sorted(samples_by_step.items()), colors):
        z = _standardize(v)
        hist, _ = np.histogram(z, bins=bins, density=True)
        mask = hist > 0
        ax.plot(centers[mask], hist[mask], lw=1.2, color=c,
                alpha=0.92, label=f"step {step}")

    ax.plot(centers, sstats.norm.pdf(centers), color="k", lw=1.6, ls="--",
            label=r"$\mathcal{N}(0,1)$ reference")

    # 对比标尺
    for k in (3, 5, 10):
        ax.axvline(k, color="grey", lw=0.5, ls=":", alpha=0.6)
        ax.axvline(-k, color="grey", lw=0.5, ls=":", alpha=0.6)

    ax.set_yscale("log")
    ax.set_xlim(-25, 25)
    ax.set_ylim(1e-7, 5e2)
    ax.set_xlabel(r"standardized $z = (\delta - \mu)/\sigma$")
    ax.set_ylabel("density (log scale)")
    ax.set_title(r"(a) Empirical density vs $\mathcal{N}(0,1)$")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.35)


def panel_qq(ax, samples_by_step: dict[int, np.ndarray], colors, max_pts: int = 8000):
    """(b) Normal Q-Q plot：bulk 偏离 y=x 表示中心过尖，尾部远离 y=x 表示重尾。"""
    rng = np.random.default_rng(0)
    abs_max = 5.0
    for (step, v), c in zip(sorted(samples_by_step.items()), colors):
        z = _standardize(v)
        if z.size > max_pts:
            z = rng.choice(z, size=max_pts, replace=False)
        z = np.sort(z)
        n = z.size
        theo = sstats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.scatter(theo, z, s=4, color=c, alpha=0.65,
                   label=f"step {step}", rasterized=True)
        abs_max = max(abs_max, float(np.abs(z).max()) * 1.05)

    xs = np.linspace(-4.5, 4.5, 200)
    ax.plot(xs, xs, color="k", lw=1.2, ls="--", label="Normal reference y=x")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-abs_max, abs_max)
    ax.set_xlabel(r"theoretical quantile of $\mathcal{N}(0,1)$")
    ax.set_ylabel(r"empirical standardized quantile")
    ax.set_title("(b) Normal Q-Q plot")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)


def panel_survival(ax, samples_by_step: dict[int, np.ndarray], colors):
    """(c) Survival function P(|Z|>t) on log-log，叠加正态参考线。

    若实测曲线全程位于正态参考之上，就是重尾的"硬证据"。
    """
    t_grid = np.logspace(np.log10(0.3), np.log10(40.0), 220)

    for (step, v), c in zip(sorted(samples_by_step.items()), colors):
        z = np.abs(_standardize(v))
        z_sorted = np.sort(z)
        n = z_sorted.size
        survival = 1.0 - np.searchsorted(z_sorted, t_grid, side="right") / n
        ax.plot(t_grid, survival + 1e-12, lw=1.7, color=c, alpha=0.95,
                label=f"step {step}")

    normal_surv = 2 * (1 - sstats.norm.cdf(t_grid))
    ax.plot(t_grid, normal_surv, color="k", lw=1.6, ls="--",
            label=r"Normal $\mathcal{N}(0,1)$ reference")

    for k, label in [(3, r"$3\sigma$"), (5, r"$5\sigma$"), (10, r"$10\sigma$")]:
        ax.axvline(k, color="grey", lw=0.6, ls=":")
        ax.text(k, 1.6, label, color="grey", fontsize=8,
                ha="center", va="bottom", alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.3, 40)
    ax.set_ylim(1e-7, 2.5)
    ax.set_xlabel(r"$t$ (in units of $\sigma$)")
    ax.set_ylabel(r"$P(|Z|>t)$")
    ax.set_title("(c) Tail survival function vs Normal (log-log)")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.grid(True, which="both", ls=":", alpha=0.35)


def panel_metrics_over_steps(ax, stats_df: pd.DataFrame, log_dir: str,
                              raw_steps: list[int], steps_used: list[int]):
    """(d) 重尾度量随 step 演化：excess kurtosis（左轴） + frac(|z|>kσ)（右轴）。

    标准化用 MAD（与 panel a/b/c 一致），更接近 bulk 真实尺度，对比 Normal 更公平。
    """
    kurts, fracs5, fracs3, fracs10, used = [], [], [], [], []

    for step in raw_steps:
        try:
            v = _load_valid_tokens(log_dir, step)
        except Exception:
            continue
        if v.size == 0:
            continue
        z = np.abs(_standardize(v))
        used.append(step)
        kurts.append(float(sstats.kurtosis(v, fisher=True)))
        fracs3.append(float(np.mean(z > 3)))
        fracs5.append(float(np.mean(z > 5)))
        fracs10.append(float(np.mean(z > 10)))

    used = np.array(used)
    kurts = np.array(kurts)
    fracs3 = np.array(fracs3)
    fracs5 = np.array(fracs5)
    fracs10 = np.array(fracs10)

    l_kurt, = ax.plot(used, kurts, color="C3", lw=1.2, marker="o", ms=2.5,
                      label="excess kurtosis (left axis)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("excess kurtosis (log)", color="C3")
    ax.tick_params(axis="y", colors="C3")
    ax.grid(True, which="both", ls=":", alpha=0.35)

    ax2 = ax.twinx()
    l3, = ax2.plot(used, np.clip(fracs3, 1e-9, None), color="C0", lw=1.0, alpha=0.9,
                   label=r"$P(|Z|>3\sigma)$")
    l5, = ax2.plot(used, np.clip(fracs5, 1e-9, None), color="C1", lw=1.2, alpha=0.95,
                   label=r"$P(|Z|>5\sigma)$")
    l10, = ax2.plot(used, np.clip(fracs10, 1e-9, None), color="C5", lw=1.0, alpha=0.85,
                    label=r"$P(|Z|>10\sigma)$")

    # 正态参考线 + 标注（放在中段位置，避开 legend）
    x_text = used.min() + (used.max() - used.min()) * 0.35
    for k, lvl, lab in [
        (3, 2 * (1 - sstats.norm.cdf(3)),
         r"$\mathcal{N}$: $P(|Z|>3\sigma)\!\approx\!2.7\!\times\!10^{-3}$"),
        (5, 2 * (1 - sstats.norm.cdf(5)),
         r"$\mathcal{N}$: $P(|Z|>5\sigma)\!\approx\!5.7\!\times\!10^{-7}$"),
    ]:
        ax2.axhline(lvl, color="grey", lw=0.7, ls="--", alpha=0.75)
        ax2.text(x_text, lvl * 1.4, lab, fontsize=7,
                 color="dimgrey", va="bottom", ha="left",
                 bbox=dict(facecolor="white", edgecolor="none",
                           alpha=0.7, pad=1))

    ax2.set_yscale("log")
    ax2.set_ylim(1e-8, 1.0)
    ax2.set_ylabel(r"tail fraction $P(|Z|>k\sigma)$ (log)")

    for s in steps_used:
        ax.axvline(s, color="grey", lw=0.5, ls=":", alpha=0.45)

    ax.legend([l_kurt, l3, l5, l10],
              [l.get_label() for l in [l_kurt, l3, l5, l10]],
              fontsize=7, loc="upper left", framealpha=0.9, ncol=2)
    ax.set_title("(d) Heavy-tail metrics over training")


def make_single_panel_survival(samples_by_step, colors, out_path):
    """额外的论文正文单 panel 主图：Tail survival function 单独放大。

    论文里"重尾"的最经典证据图：实测 P(|Z|>t) 全程位于 Normal 参考线之上，
    在 t=5 处差距即可达数个数量级。
    """
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    panel_survival(ax, samples_by_step, colors)
    ax.set_title(
        r"Token-level $\delta=\log\pi_\theta-\log\pi_{\theta_\mathrm{old}}$"
        r" has heavier tails than $\mathcal{N}(0,1)$"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    print(f"[plot] saved {pdf_path}")


def make_qq_survival_figure(samples_by_step, colors, out_path):
    """论文正文用的双子图：左 Tail survival function，右 Q-Q plot。

    两张图在重尾文献里互补：
      - Tail survival 在 log-log 下定量说明尾部比 Normal 厚多少个数量级
      - Q-Q 图直观展示 bulk 与 tail 同时偏离 Normal

    字体方案：
      - pdf/ps fonttype = 42 (TrueType 嵌入，等价于 Type 1 字体合规检查)
      - 衬线字体 DejaVu Serif，数学符号用 STIX (Times-like)
      - 字号按论文版面（单/双栏 ~10-11pt）放大
    """
    paper_rc = {
        # 字体子集嵌入，避开 Type 3，会议系统不会拒
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # 衬线字体 + STIX 数学符号，最接近 Times Roman 风格
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        # 论文字号
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "savefig.bbox": "tight",
    }
    with plt.rc_context(paper_rc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        panel_survival(axes[0], samples_by_step, colors)
        panel_qq(axes[1], samples_by_step, colors)

        # 子图标题改成 (a)/(b) 简短形式
        axes[0].set_title("(a) Tail survival function vs Normal")
        axes[1].set_title("(b) Normal Q-Q plot")

        # panel_qq / panel_survival 内部用了 hardcoded 的小字号，统一覆盖一次
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontsize(paper_rc["legend.fontsize"])
            for t in ax.texts:
                t.set_fontsize(paper_rc["xtick.labelsize"])
            # 散点尺寸放大一点，论文里更显眼
            for col in ax.collections:
                try:
                    sizes = col.get_sizes()
                    if sizes.size > 0:
                        col.set_sizes(sizes * 2.5)
                except Exception:
                    pass

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, dpi=300)
        plt.close(fig)
    print(f"[plot] saved {out_path}")
    print(f"[plot] saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./save_gspo_kl_logs")
    parser.add_argument("--out_path", default="./figs_negative_approx_kl/heavy_tail_paper.png")
    parser.add_argument("--steps", type=int, nargs="*",
                        default=[10, 1340, 2670, 4000])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    available = _list_raw_steps(args.log_dir)
    steps = [s for s in args.steps if s in available]
    if not steps:
        raise SystemExit(f"none of {args.steps} found in {args.log_dir}")
    print(f"[info] panels (a)(b)(c) use steps: {steps}")

    samples_by_step: dict[int, np.ndarray] = {
        s: _load_valid_tokens(args.log_dir, s) for s in steps
    }

    # 颜色：从 viridis 取离散颜色，避免和黑色参考线撞色
    cmap = plt.get_cmap("viridis", max(len(steps), 4))
    colors = [cmap(i / max(len(steps) - 1, 1) * 0.85) for i in range(len(steps))]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_loghist_with_normal(axes[0, 0], samples_by_step, colors)
    panel_qq(axes[0, 1], samples_by_step, colors)
    panel_survival(axes[1, 0], samples_by_step, colors)

    stats_path = os.path.join(args.log_dir, "stats.txt")
    stats_df = pd.read_csv(stats_path, sep="\t") if os.path.exists(stats_path) else pd.DataFrame()
    panel_metrics_over_steps(axes[1, 1], stats_df, args.log_dir, available, steps)

    fig.suptitle(
        r"Heavy-tail behaviour of token-level $\delta=\log\pi_\theta-\log\pi_{\theta_\mathrm{old}}$ in GSPO training",
        fontsize=13, y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig.savefig(args.out_path, dpi=200)
    pdf_path = os.path.splitext(args.out_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    print(f"[plot] saved {args.out_path}")
    print(f"[plot] saved {pdf_path}  (vector, recommended for paper)")

    # 额外：独立的单 panel 主图（survival function）
    base, ext = os.path.splitext(args.out_path)
    single_path = base + "_survival_only" + ext
    make_single_panel_survival(samples_by_step, colors, single_path)

    # 额外：Q-Q + Survival 的双子图主图
    qq_surv_path = base + "_qq_survival" + ext
    make_qq_survival_figure(samples_by_step, colors, qq_surv_path)


if __name__ == "__main__":
    main()
