"""Figures for the blog post. Static PNGs, one function per figure."""

import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette (dataviz skill), in fixed order: one slot per model, never cycled.
SERIES = {"Qwen3-8B": "#2a78d6", "Qwen3.5-9B": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "text.color": INK,
})


def think_time_cdf(df, caps=(60, 300)):
    m = df[(df.stream == "main") & (df.think_time > 0)]
    x = np.sort(m.think_time.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(x, y, color=SERIES["Qwen3-8B"])
    ax.set_xscale("log")
    for cap in caps:
        share = (x <= cap).mean()
        ax.axvline(cap, color=INK_2, linewidth=1, linestyle=":")
        ax.annotate(f"{share:.0%} of gaps\nunder {cap} s", (cap, share), xytext=(-6, 4),
                    textcoords="offset points", ha="right", fontsize=9, color=INK_2)
    ax.set_xlabel("gap between a response and the next request (s, log scale)")
    ax.set_ylabel("share of gaps")
    ax.set_title("How long a session stays parked", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def sessions_vs_context(pools, required=32, marked_context=None):
    """pools: {Model: KV pool bytes}."""
    ctx = np.linspace(4_096, 131_072, 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, pool in pools.items():
        n = [m.resident_sessions(c, pool) for c in ctx]
        ax.plot(ctx / 1024, n, color=SERIES[m.name])
        ax.annotate(m.name, (ctx[-1] / 1024, n[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axhline(required, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"requirement: {required}", (ctx[0] / 1024, required), xytext=(0, 4),
                textcoords="offset points", fontsize=9, color=INK_2)
    if marked_context:
        ax.axvline(marked_context / 1024, color=INK_2, linewidth=1, linestyle=":")
        ax.annotate("our traces", (marked_context / 1024, 1), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=INK_2)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_ylim(1, 300)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlim(0, 150)
    ax.set_xlabel("context per session (K tokens)")
    ax.set_ylabel("sessions that fit in HBM (log)")
    ax.set_title("Sessions resident in BF16, napkin", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def prefill_vs_napkin(runs, C):
    """runs: {Model: m3m4 result dict}. Measured TTFT (median of reps) as dots, FLOPs/C as a line."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, r in runs.items():
        T = np.array([p["tokens"] for p in r["prefill"]])
        measured = np.array([sorted(p["ttft_s"])[len(p["ttft_s"]) // 2] for p in r["prefill"]])
        grid = np.linspace(T.min(), T.max(), 200)
        ax.plot(grid / 1024, [m.prefill_flops(t) / C for t in grid], color=SERIES[m.name], linewidth=1.5, alpha=0.6)
        ax.plot(T / 1024, measured, "o", color=SERIES[m.name], markersize=6, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(m.name, (T[-1] / 1024, measured[-1]), xytext=(-8, 8), textcoords="offset points",
                    ha="right", fontsize=9, color=INK)
    ax.set_xlabel("prompt length (K tokens)")
    ax.set_ylabel("time to first token (s)")
    ax.set_title("Prefill: measured (dots) against FLOPs / C (lines)", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def decode_vs_napkin(runs, C, W):
    """Step time against batch, one line per context: measured as dots, the bandwidth napkin dashed."""
    contexts = {4096: "#2a78d6", 16384: "#eb6834", 65536: "#1baf7a"}  # categorical slots 1-3, fixed order
    fig, axes = plt.subplots(1, len(runs), figsize=(max(6.4, 4.8 * len(runs)), 3.6), sharey=True)
    for ax, (m, r) in zip(np.atleast_1d(axes), runs.items()):
        for S, color in contexts.items():
            pts = sorted((d["batch"], d["step_s"] * 1e3) for d in r["decode"] if d["context"] == S)
            if not pts:
                continue
            B = np.array([b for b, _ in pts])
            grid = np.geomspace(1, B.max(), 50)
            ax.plot(grid, [m.decode_step_seconds(b, S, C, W) * 1e3 for b in grid], color=color, linewidth=1.5,
                    linestyle="--", alpha=0.7)
            ax.plot(B, [t for _, t in pts], "o", color=color, markersize=6, markeredgecolor="white",
                    markeredgewidth=1.5)
            ax.annotate(f"{S // 1024}K", (B[-1], pts[-1][1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 32, 64])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlim(0.8, 90)
        ax.set_title(m.name, loc="left", fontsize=10)
        ax.set_xlabel("batch size (log)")
    np.atleast_1d(axes)[0].set_ylabel("decode step time (ms)")
    fig.suptitle("Decode: measured (dots) against the bandwidth napkin (dashed), per context length",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


CACHED, COMPUTED, IDLE = "#2a78d6", "#eb6834", "#f1f0ec"


def cache_reuse_grid(n=6):
    """Which positions' keys and values get computed at each generation step, without and with a cache."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, use_cache in zip(axes, (False, True)):
        work = 0
        for step in range(n):
            for pos in range(n):
                if pos > step:
                    color = IDLE
                elif use_cache and pos < step:
                    color = CACHED
                else:
                    color, work = COMPUTED, work + 1
                ax.add_patch(plt.Rectangle((pos + 0.06, step + 0.06), 0.88, 0.88, color=color, linewidth=0))
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks(np.arange(n) + 0.5, [f"t{i}" for i in range(n)])
        ax.set_yticks(np.arange(n) + 0.5, [f"step {i + 1}" for i in range(n)])
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("token position")
        ax.set_title(f"{'With' if use_cache else 'Without'} a KV cache: {work} K,V computations", loc="left", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COMPUTED, CACHED)]
    fig.legend(handles, ["computed this step", "read from the cache"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def roofline(C, W, points):
    """points: [(label, arithmetic intensity in FLOP/byte, label offset in points)] placed on the roof min(C, W * I)."""
    I = np.geomspace(0.5, 1e5, 400)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(I, np.minimum(C, W * I) / 1e12, color=INK_2, linewidth=2)
    ridge = C / W
    ax.axvline(ridge, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(f"ridge: {ridge:.0f} FLOP/byte", (ridge, C / 1e12 / 30), xytext=(5, 0), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.annotate("bandwidth-bound", (3, W * 3 / 1e12), xytext=(0, 10), textcoords="offset points", fontsize=9,
                color=INK_2, rotation=33)
    ax.annotate("compute-bound", (1e3, C / 1e12), xytext=(0, 6), textcoords="offset points", fontsize=9, color=INK_2)
    for label, intensity, offset in points:
        y = min(C, W * intensity) / 1e12
        ax.plot(intensity, y, "o", color=COMPUTED, markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(label, (intensity, y), xytext=offset, textcoords="offset points", fontsize=8.5, color=INK,
                    ha="center" if offset[0] == 0 else "left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.5, C / 1e12 * 3)
    ax.set_xlabel("arithmetic intensity (FLOPs per byte read from memory)")
    ax.set_ylabel("best possible TFLOP/s")
    ax.set_title("Roofline of this GPU, from measured peaks", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def memory_budget(rows, total_gb):
    """rows: {model name: [(segment label, GB)]}, drawn as one stacked horizontal bar per model."""
    colors = {"weights": "#2a78d6", "KV cache pool": "#eb6834", "recurrent-state pool": "#1baf7a",
              "other runtime buffers": "#eda100", "left outside the 0.85 fraction": "#d9d8d2"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for y, (name, segments) in enumerate(rows.items()):
        left = 0
        for label, size in segments:
            ax.barh(y, size, left=left, color=colors[label], height=0.55, edgecolor="white", linewidth=2,
                    label=label if y == 0 or label == "recurrent-state pool" else None)
            if size > 6:
                ax.text(left + size / 2, y, f"{size:.0f}", ha="center", va="center", fontsize=9, color=INK)
            left += size
    ax.set_yticks(range(len(rows)), list(rows))
    ax.invert_yaxis()
    ax.set_xlim(0, total_gb)
    ax.set_xlabel("GB of GPU memory")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in colors if l in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("Where GPU memory goes, as the engine allocated it", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def layer_diagram():
    """One decoder layer of Qwen3-8B for the newest token, and where its K and V go."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.7)
    ax.axis("off")
    box_fill, edge = "#f1f0ec", INK_2

    def box(x, y, w, h, text, fill=box_fill, color=INK, size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=edge, linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color)

    def arrow(x0, y0, x1, y1, color=INK_2, rad=0.0):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=0,
                                                             shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    # the newest token's path through the layer, bottom to top
    box(0.2, 0.2, 3.6, 0.65, "token $t$'s vector $x_t$ (4,096 numbers)")
    arrow(2.0, 0.85, 2.0, 1.25)
    box(0.2, 1.25, 3.6, 0.5, "RMSNorm", size=8.5)
    for x, label in ((0.2, "$W_Q$\n4096→32×128"), (1.4, "$W_K$\n4096→8×128"), (2.6, "$W_V$\n4096→8×128")):
        arrow(2.0, 1.75, x + 0.6, 2.15)
        box(x, 2.15, 1.2, 0.75, label, size=7.5)
    box(0.2, 3.3, 1.2, 0.55, "$q_t$ (+RoPE)", fill="#ffffff", size=7.5)
    box(1.4, 3.3, 1.2, 0.55, "$k_t$ (+RoPE)", fill=COMPUTED, color="white", size=7.5)
    box(2.6, 3.3, 1.2, 0.55, "$v_t$", fill=COMPUTED, color="white", size=7.5)
    for x in (0.8, 2.0, 3.2):
        arrow(x, 2.9, x, 3.3)
    box(0.2, 4.35, 3.6, 0.65, "attention: $q_t$ against the cache", size=8.5)
    arrow(0.8, 3.85, 0.8, 4.35)
    box(0.2, 5.45, 3.6, 0.5, "$W_O$, add back to $x_t$", size=8.5)
    arrow(2.0, 5.0, 2.0, 5.45)
    box(0.2, 6.4, 3.6, 0.8, "MLP 4096 → 12288 → 4096\n(this token only), add back", size=8.5)
    arrow(2.0, 5.95, 2.0, 6.4)
    box(0.2, 7.65, 3.6, 0.55, "next layer (36 in total)", fill="#ffffff", size=8.5)
    arrow(2.0, 7.2, 2.0, 7.65)

    # this layer's cache: one row per past token, plus the new one at the top
    cx, cy, rows, rh = 6.2, 1.3, 7, 0.42
    for j in range(rows):
        new = j == rows - 1
        label = f"token {j}" if j < rows - 2 else ("…" if j == rows - 2 else "token $t$")
        for col, x in enumerate((cx, cx + 1.7)):
            ax.add_patch(plt.Rectangle((x, cy + j * rh), 1.5, rh - 0.06, color=COMPUTED if new else CACHED, linewidth=0))
            ax.text(x + 0.75, cy + j * rh + (rh - 0.06) / 2, ("K " if col == 0 else "V ") + label, ha="center",
                    va="center", fontsize=7, color="white")
    top = cy + rows * rh
    ax.text(cx + 1.6, top + 0.15, "KV cache of this layer", ha="center", fontsize=9.5, color=INK)
    ax.text(cx + 1.6, cy - 0.4, "written once, read by every later token", ha="center", fontsize=8, color=INK_2)
    arrow(3.8, 3.58, cx, cy + (rows - 1) * rh + 0.18, color=COMPUTED)
    ax.text(4.9, 3.75, "append", fontsize=8, color=COMPUTED, rotation=12)
    arrow(cx, cy + 1.2, 3.8, 4.6, color=CACHED, rad=-0.15)
    ax.text(4.6, 4.75, "read all of it", fontsize=8, color=CACHED, rotation=-15)

    ax.text(6.0, 7.95, "Only $k_t$ and $v_t$ outlive this step.", fontsize=10, color=INK)
    ax.text(6.0, 7.45, "$q_t$ and the MLP's numbers are used once and thrown away.", fontsize=9, color=INK_2)
    ax.text(6.0, 7.0, "All 36 layers keep a cache like this one.", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def kv_vs_weights(lines, weights_gb, pool_gb, max_ctx=131_072):
    """lines: [(label, bytes per token, color)]; one conversation's cache against context length."""
    ctx = np.linspace(0, max_ctx, 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for label, per_token, color in lines:
        ax.plot(ctx / 1024, ctx * per_token / 1e9, color=color)
        x_lab = min(max_ctx * 0.97, 50e9 / per_token)
        ax.annotate(label, (x_lab / 1024, x_lab * per_token / 1e9), xytext=(-6, 6), textcoords="offset points",
                    ha="right", fontsize=9, color=INK)
        cross = weights_gb * 1e9 / per_token
        if cross < max_ctx:
            ax.plot(cross / 1024, weights_gb, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{cross / 1000:.0f}K tokens", (cross / 1024, weights_gb), xytext=(6, -14),
                        textcoords="offset points", fontsize=8.5, color=INK_2)
    for y, label in ((weights_gb, "the model's own weights"), (pool_gb, "everything the GPU has left for caches")):
        ax.axhline(y, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(label, (2, y), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_ylim(0, 80)
    ax.set_xlim(0, max_ctx / 1024)
    ax.set_xlabel("conversation length (K tokens)")
    ax.set_ylabel("KV cache of one conversation (GB)")
    ax.set_title("One conversation's cache against the model, BF16", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def decode_breakdown(model, run, W, points):
    """Napkin split of one decode step into weight reads and cache reads, with the measured step on top."""
    measured = {(d["batch"], d["context"]): d["step_s"] * 1e3 for d in run["decode"]}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w_ms = 2 * model.matmul_params / W * 1e3
    for i, (B, S) in enumerate(points):
        kv_ms = B * S * model.kv_bytes_per_token() / W * 1e3
        ax.bar(i, w_ms, color="#d9d8d2", width=0.6, edgecolor="white", linewidth=2, label="read the weights" if i == 0 else None)
        ax.bar(i, kv_ms, bottom=w_ms, color=CACHED, width=0.6, edgecolor="white", linewidth=2,
               label="read the KV cache" if i == 0 else None)
        if (B, S) in measured:
            ax.plot(i, measured[(B, S)], "o", color=COMPUTED, markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    label="measured step" if i == 0 else None)
        ax.text(i, w_ms + kv_ms + 1.2, f"{kv_ms / (w_ms + kv_ms):.0%} cache", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(points)), [f"{B} seq × {S // 1024}K" for B, S in points])
    ax.set_ylabel("time per decode step (ms)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(f"{model.name}: where one decode step's time goes", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def ttft_by_turn(series, r2=1.0):
    """series: [(label, records, color)]. Median time to first token per turn index, with the p10-p90 band."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, recs, color in series:
        turns = sorted({r["turn"] for r in recs})
        vals = [[r["ttft"] for r in recs if r["turn"] == t] for t in turns]
        med = [np.median(v) for v in vals]
        ax.fill_between(turns, [np.percentile(v, 10) for v in vals], [np.percentile(v, 90) for v in vals],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(turns, med, color=color, marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(label, (turns[-1], med[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.axhline(r2, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("R2: 1 s", (0, r2), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel("turn of the session")
    ax.set_ylabel("time to first token (s, log)")
    ax.set_xlim(-0.5, max(r["turn"] for _, recs, _ in series for r in recs) + 4)
    ax.set_title("Time to first token by turn (median, p10-p90 band)", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def ttft_vs_napkin(series, C, r2=1.0):
    """series: [(label, model, records, color)]. Each request's measured TTFT against FLOPs(new, cached) / C."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lo, hi = 1e-2, 1e-2
    for label, model, recs, color in series:
        x = [model.prefill_flops(r["prompt_tokens"] - (r["cached_tokens"] or 0), r["cached_tokens"] or 0) / C for r in recs]
        y = [r["ttft"] for r in recs]
        ax.plot(x, y, "o", color=color, markersize=4.5, alpha=0.7, markeredgewidth=0, label=label)
        hi = max(hi, max(x), max(y))
    ax.plot([lo, hi * 1.3], [lo, hi * 1.3], color=INK_2, linewidth=1, linestyle=":")
    ax.axhline(r2, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("R2: 1 s", (lo * 1.2, r2), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi * 1.3)
    ax.set_ylim(lo, hi * 1.3)
    ax.set_xlabel("napkin: FLOPs for the new tokens over the cached prefix / C (s)")
    ax.set_ylabel("measured time to first token (s)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Every request with the cache on: on the dotted line, the napkin is exact", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def cliff(points, napkin, r2=1.0):
    """points: {(model label, workload): [(sessions, hit rate or None, p90 TTFT)]}, already averaged over repeats.
    napkin: {(model label, workload): predicted sessions}. Two panels, one x axis: hit rate, then p90 TTFT."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
    style = {"synthetic": dict(linestyle="-", marker="o", markerfacecolor=None),
             "AgentX": dict(linestyle="--", marker="s", markerfacecolor="white")}
    for (model, work), rows in points.items():
        color, st = SERIES[model], style[work]
        face = st["markerfacecolor"] or color
        xs = [c for c, h, _ in rows if h is not None]
        top.plot(xs, [h for c, h, _ in rows if h is not None], color=color, linestyle=st["linestyle"], marker=st["marker"],
                 markersize=5.5, markerfacecolor=face, markeredgecolor=color, label=f"{model}, {work}")
        bot.plot([c for c, _, t in rows], [t for _, _, t in rows], color=color, linestyle=st["linestyle"],
                 marker=st["marker"], markersize=5.5, markerfacecolor=face, markeredgecolor=color)
        x = napkin[(model, work)]
        top.plot([x], [1.04], marker="v", markersize=7, color=color, markerfacecolor=face, markeredgecolor=color,
                 clip_on=False, linestyle="none")
    top.annotate("napkin cliff", (1, 1.04), xytext=(0, 0), textcoords="offset points", fontsize=8, color=INK_2,
                 va="center")
    top.set_ylim(-0.03, 1.0)
    top.set_ylabel("cache hit rate")
    top.legend(frameon=False, fontsize=8.5, loc="lower left")
    bot.axhline(r2, color=INK_2, linewidth=1, linestyle=":")
    bot.annotate("R2: 1 s", (1, r2), xytext=(0, 4), textcoords="offset points", fontsize=9, color=INK_2)
    bot.set_yscale("log")
    bot.set_ylabel("p90 time to first token (s, log)")
    bot.set_xscale("log", base=2)
    ticks = sorted({c for rows in points.values() for c, _, _ in rows})
    bot.set_xticks(ticks, [str(t) for t in ticks])
    bot.minorticks_off()
    bot.set_xlabel("agent sessions at once")
    top.set_title("The cliff: sessions at once against hit rate and wait", loc="left", fontsize=11, pad=14)
    fig.tight_layout()
    return fig


def _bar(ax, y, segments, h=0.62, size=7.5):
    """segments: [(label, width, color)] laid end to end on row y. White labels inside wide-enough segments."""
    x = 0.0
    for label, w, color in segments:
        ax.add_patch(plt.Rectangle((x + 0.03, y), w - 0.06, h, color=color, linewidth=0))
        if label:
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size,
                    color="white" if color != IDLE else INK_2)
        x += w
    return x


def session_prefixes():
    """Left: four turns of one agent, each the previous prompt plus new tokens. Right: two agents in a radix tree."""
    from matplotlib.patches import FancyBboxPatch
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
    sizes = [3.0, 1.2, 0.5, 1.6]  # system + task, then what each turn adds
    labels = ["system prompt + task", "tool result", "grep", "big test log"]
    for turn in range(4):
        segs = [(None, sum(sizes[:turn]), CACHED)] if turn else []
        segs.append((labels[turn], sizes[turn], COMPUTED))
        _bar(left, 3 - turn, [(s[0] if s[0] else ("cached" if s[2] == CACHED and s[1] > 1 else ""), s[1], s[2]) for s in segs])
        left.text(-0.15, 3 - turn + 0.31, f"turn {turn + 1}", ha="right", va="center", fontsize=8.5, color=INK_2)
    left.set_xlim(-1.3, sum(sizes) + 0.2)
    left.set_ylim(-0.3, 4.2)
    left.axis("off")
    left.set_title("One agent: each prompt is the last one plus a bit", loc="left", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CACHED, COMPUTED)]
    left.legend(handles, ["found in the cache", "prefilled this turn"], loc="lower right", frameon=False, fontsize=8,
                ncol=2, bbox_to_anchor=(1.0, -0.12))

    right.set_xlim(0, 10)
    right.set_ylim(0, 6)
    right.axis("off")
    right.set_title("Two agents in the radix tree", loc="left", fontsize=10)

    def node(x, y, w, text, fill):
        right.add_patch(FancyBboxPatch((x, y), w, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                       facecolor=fill, edgecolor=INK_2, linewidth=0.8))
        right.text(x + w / 2, y + 0.35, text, ha="center", va="center", fontsize=7.5,
                   color="white" if fill != "#ffffff" else INK)

    def edge(x0, y0, x1, y1):
        right.plot([x0, x1], [y0, y1], color=INK_2, linewidth=1)

    node(3.2, 5.0, 3.6, "root", "#ffffff")
    node(2.4, 3.8, 5.2, "shared system prompt (stored once)", CACHED)
    edge(5.0, 5.0, 5.0, 4.5)
    for x, who in ((0.3, "A"), (5.4, "B")):
        node(x, 2.5, 4.3, f"agent {who}: task, turn 1", CACHED)
        node(x, 1.3, 4.3, f"agent {who}: turns 2, 3, …", CACHED)
        edge(5.0, 3.8, x + 2.15, 3.2)
        edge(x + 2.15, 2.5, x + 2.15, 2.0)
    right.text(5.0, 0.55, "a new request walks down from the root;\nwhere it falls off, the prefill starts",
               ha="center", va="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    return fig


def prefix_breaks():
    """Turn N's prompt under four harness habits: what is still reusable from turn N-1's cache, and what is not."""
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    rows = [
        ("append-only", [("system", 2.2, CACHED), ("tools", 2.2, CACHED), ("history", 3.4, CACHED),
                         ("new", 1.0, COMPUTED)]),
        ("clock at the top", [("", 0.35, COMPUTED), ("system", 1.85, COMPUTED), ("tools", 2.2, COMPUTED),
                              ("history", 3.4, COMPUTED), ("new", 1.0, COMPUTED)]),
        ("tools reordered", [("system", 2.2, CACHED), ("tools, new order", 2.2, COMPUTED), ("history", 3.4, COMPUTED),
                             ("new", 1.0, COMPUTED)]),
        ("old reply dropped", [("system", 2.2, CACHED), ("tools", 2.2, CACHED), ("history", 2.2, CACHED),
                               ("prev. tool result", 1.2, COMPUTED), ("new", 1.0, COMPUTED)]),
    ]
    for i, (name, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        _bar(ax, y, segs)
        ax.text(-0.15, y + 0.31, name, ha="right", va="center", fontsize=9, color=INK)
    ax.annotate("the clock", (0.17, 2.62), xytext=(0.4, 3.0 - 0.02), fontsize=7.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.set_xlim(-2.6, 9.0)
    ax.set_ylim(-0.6, 4.0)
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CACHED, COMPUTED)]
    ax.legend(handles, ["reused from the last turn", "prefilled again"], loc="lower center", ncol=2, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.6, -0.12))
    ax.set_title("Turn N's prompt: everything after the first changed token is a new cache", loc="left", fontsize=10.5)
    fig.tight_layout()
    return fig


def _simulate_loop(n, pool, policy, steps):
    """n sessions return in a fixed cycle; the pool holds `pool` of them. Returns [(step, session, hit, resident)]."""
    resident, last_use, out = [], {}, []
    for t in range(steps):
        s = t % n
        hit = s in resident
        if not hit:
            if len(resident) == pool:
                if policy == "lru":
                    victim = min(resident, key=lambda r: last_use[r])
                else:  # Belady: evict the one whose next use is furthest away; in a cycle that is the most recent
                    victim = max(resident, key=lambda r: (r - s) % n)
                resident.remove(victim)
            resident.append(s)
        last_use[s] = t
        out.append((t, s, hit, sorted(resident)))
    return out


def loop_grid(policies=("lru",), n=5, pool=4, steps=12):
    """Rows are time steps, columns sessions. Shows which caches are in the pool as sessions come back in a loop."""
    names = {"lru": "least recently used (LRU)", "belady": "furthest next use (Belady)"}
    fig, axes = plt.subplots(1, len(policies), figsize=(max(5.0, 3.6 * len(policies)), 4.2), squeeze=False)
    HIT, MISS = "#1baf7a", COMPUTED
    for ax, policy in zip(axes[0], policies):
        trace = _simulate_loop(n, pool, policy, steps)
        hits = 0
        for t, s, hit, resident in trace:
            for col in range(n):
                if col == s:
                    color = HIT if hit else MISS
                elif col in resident:
                    color = CACHED
                else:
                    color = IDLE
                ax.add_patch(plt.Rectangle((col + 0.06, t + 0.06), 0.88, 0.88, color=color, linewidth=0))
            hits += hit
        ax.set_xlim(0, n)
        ax.set_ylim(steps, 0)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks(np.arange(n) + 0.5, [chr(65 + i) for i in range(n)])
        ax.set_yticks(np.arange(steps) + 0.5, [str(i + 1) for i in range(steps)], fontsize=7.5)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("session")
        ax.set_ylabel("turn (in order of arrival)")
        ax.set_title(f"{names[policy]}\n{hits} hits in {steps - n} returns", loc="left", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (HIT, MISS, CACHED)]
    fig.legend(handles, ["returns, cache still there", "returns, cache gone: full prefill", "parked, cache in the pool"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def loop_hit_rate(pool=8, max_n=32):
    """Steady-state hit rate of a cycle of n sessions through a pool of `pool`, LRU against Belady."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ns = list(range(1, max_n + 1))
    for policy, label, color in (("lru", "LRU", COMPUTED), ("belady", "Belady (best possible)", CACHED)):
        rates = []
        for n in ns:
            tr = _simulate_loop(n, pool, policy, 60 * n)[10 * n:]
            rates.append(np.mean([h for _, _, h, _ in tr]))
        ax.plot(ns, rates, color=color, marker="o", markersize=3.5, label=label)
    ax.axvline(pool, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate(f"pool holds {pool} sessions", (pool, 0.08), xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                color=INK_2)
    ax.set_xlabel("sessions in the loop")
    ax.set_ylabel("hit rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("A loop of sessions through a fixed pool", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def pool_split(rows, pool):
    """rows: [(sessions, held by running requests, parked and evictable, free)] in tokens. Stacked bars."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(rows))
    run = np.array([r[1] for r in rows]) / 1e3
    park = np.array([r[2] for r in rows]) / 1e3
    free = np.array([r[3] for r in rows]) / 1e3
    ax.bar(xs, run, color=COMPUTED, width=0.7, label="held by running requests (cannot be evicted)")
    ax.bar(xs, park, bottom=run, color=CACHED, width=0.7, label="parked sessions' caches (evictable)")
    ax.bar(xs, free, bottom=run + park, color=IDLE, width=0.7, label="free")
    ax.axhline(pool / 1e3, color=INK_2, linewidth=1, linestyle=":")
    ax.set_xticks(xs, [str(r[0]) for r in rows])
    ax.set_xlabel("agent sessions at once")
    ax.set_ylabel("tokens in the pool (thousands)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=1)
    ax.set_title("What fills the pool", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# Ordinal blue steps (reference palette, light surface: no lighter than step 250).
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def salted_tree():
    """Left: one radix tree shared by two customers. Right: the same agents with a per-customer salt."""
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, salted in zip(axes, (False, True)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.2)
        ax.axis("off")

        def node(x, y, w, text, fill):
            ax.add_patch(FancyBboxPatch((x, y), w, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=fill, edgecolor=INK_2, linewidth=0.8))
            ax.text(x + w / 2, y + 0.35, text, ha="center", va="center", fontsize=7.5,
                    color="white" if fill != "#ffffff" else INK)

        def edge(x0, y0, x1, y1):
            ax.plot([x0, x1], [y0, y1], color=INK_2, linewidth=1)

        node(3.7, 4.3, 2.6, "root", "#ffffff")
        if not salted:
            node(2.2, 3.1, 5.6, "system prompt (one copy)", CACHED)
            edge(5.0, 4.3, 5.0, 3.8)
            for i, (x, who) in enumerate(((0.1, "A1"), (2.6, "A2"), (5.1, "B1"), (7.6, "B2"))):
                node(x, 1.7, 2.3, f"agent {who}", COMPUTED if who[0] == "A" else CATEGORICAL[2])
                edge(5.0, 3.1, x + 1.15, 2.4)
            ax.set_title("Shared: customers A and B under one prefix", loc="left", fontsize=10)
            ax.text(5.0, 0.7, "B's first request finds A's prompt: faster, and visible", ha="center", fontsize=8, color=INK_2)
        else:
            for x, who, c in ((0.3, "A", COMPUTED), (5.3, "B", CATEGORICAL[2])):
                node(x, 3.1, 4.4, f"salt {who} + system prompt (copy {who})", CACHED)
                edge(5.0, 4.3, x + 2.2, 3.8)
                for j in range(2):
                    node(x + j * 2.3, 1.7, 2.1, f"agent {who}{j + 1}", c)
                    edge(x + 2.2, 3.1, x + j * 2.3 + 1.05, 2.4)
            ax.set_title("Salted: one branch per customer", loc="left", fontsize=10)
            ax.text(5.0, 0.7, "agents of one customer still share; customers do not", ha="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    return fig


def cliff_by_format(points, napkin):
    """points: {format label: [(agents, hit rate, median wait)]}, napkin: {format label: agents}. One model."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    for (name, rows), color in zip(points.items(), (ORDINAL[0], ORDINAL[2], ORDINAL[4])):
        xs = [c for c, _, _ in rows]
        top.plot(xs, [h for _, h, _ in rows], color=color, marker="o", markersize=5, label=name)
        bot.plot(xs, [w for _, _, w in rows], color=color, marker="o", markersize=5)
        top.plot([napkin[name]], [1.04], marker="v", markersize=7, color=color, clip_on=False, linestyle="none")
    top.annotate("napkin cliff", (6.2, 1.04), fontsize=8, color=INK_2, va="center")
    top.set_ylim(-0.03, 1.0)
    top.set_ylabel("cache hit rate")
    top.legend(frameon=False, fontsize=8.5, loc="lower left")
    bot.set_yscale("log")
    bot.set_ylabel("median wait (s, log)")
    bot.set_xscale("log", base=2)
    ticks = sorted({c for rows in points.values() for c, _, _ in rows})
    bot.set_xticks(ticks, [str(t) for t in ticks])
    bot.minorticks_off()
    bot.set_xlabel("agents at once")
    top.set_title("Qwen3-8B: the cliff per cache format", loc="left", fontsize=11, pad=14)
    fig.tight_layout()
    return fig


def kv_outliers(amax_k, amax_v):
    """Largest |key| and |value| per layer, from the calibration windows. Log scale."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    layers = np.arange(len(amax_k))
    ax.plot(layers, amax_k, color=CATEGORICAL[0], marker="o", markersize=3.5, label="keys")
    ax.plot(layers, amax_v, color=CATEGORICAL[1], marker="o", markersize=3.5, label="values")
    ax.axhline(448, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate("largest FP8 (E4M3) number: 448", (len(layers) - 1, 448), xytext=(0, -11), textcoords="offset points",
                fontsize=8, color=INK_2, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("largest |number| (log)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Qwen3-8B: the largest key and value in each layer", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def read_less(evict, oracle, threshold, ctx):
    """KL against the share of the cache a step uses: eviction keeps a share, the oracle reads a share."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (name, rows), color in zip(evict.items(), CATEGORICAL[1:3]):
        ax.plot([b for b, _ in rows], [k for _, k in rows], color=color, marker="s", markersize=5, label=f"{name}: keep this share")
    ax.plot([t / ctx for t, _ in oracle], [k for _, k in oracle], color=CATEGORICAL[0], marker="o", markersize=5,
            label="oracle: keep all, read this share")
    ax.axhline(threshold, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate("R4 threshold", (0.55, threshold), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2,
                ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    xt = [0.01, 0.03, 0.1, 0.25, 0.5]
    ax.set_xticks(xt, [f"{x:.0%}" for x in xt])
    ax.minorticks_off()
    ax.set_xlabel("share of the cache used per step (log)")
    ax.set_ylabel("KL vs full cache (log)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    ax.set_title("Qwen3-8B, 32K contexts: dropping tokens vs reading fewer", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def speculation(rows, r3=30):
    """rows: {draft tokens: [(requests at once, tokens/s per user)]}."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (d, pts), color in zip(rows.items(), ORDINAL):
        ax.plot([c for c, _ in pts], [t for _, t in pts], color=color, marker="o", markersize=5,
                label="no speculation" if d == 0 else f"{d} draft token{'s' if d > 1 else ''}")
    ax.axhline(r3, color=INK_2, linewidth=1, linestyle=":")
    ax.annotate("R3: 30 tokens/s", (32, r3), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK_2,
                ha="right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = sorted({c for pts in rows.values() for c, _ in pts})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("requests at once")
    ax.set_ylabel("tokens/s per user (log)")
    yt = [10, 20, 30, 50, 100, 200]
    ax.set_yticks(yt, [str(y) for y in yt])
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Qwen3.5, 16K-token code prompts: speculation against load", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def tier_ladder(rows):
    """rows: [(label, seconds)] to bring one agent's private context back into the GPU. Horizontal bars, log x."""
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ys = np.arange(len(rows))[::-1]
    for y, (label, s), color in zip(ys, rows, (ORDINAL[0], ORDINAL[2], ORDINAL[4], COMPUTED)):
        ax.barh(y, s, color=color, height=0.6)
        ax.text(s * 1.08, y, f"{s:.2f} s" if s < 1 else f"{s:.1f} s", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(min(s for _, s in rows) / 2, max(s for _, s in rows) * 4)
    ax.set_xlabel("seconds (log)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Napkin: bringing back one agent's 33K private tokens, Qwen3-8B", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


def host_tier(points):
    """points: {model label: {tier label: [(agents, median wait)]}}. One panel per model, median wait, log y."""
    fig, axes = plt.subplots(1, len(points), figsize=(9.6, 3.4), sharey=True)
    for ax, (model, tiers) in zip(axes, points.items()):
        for (name, pts), color in zip(tiers.items(), (ORDINAL[0], ORDINAL[2], ORDINAL[4])):
            pts = [(c, w) for c, w in pts if w is not None]
            ax.plot([c for c, _ in pts], [w for _, w in pts], color=color, marker="o", markersize=5, label=name)
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ticks = sorted({c for pts in tiers.values() for c, _ in pts})
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("agents at once")
        ax.set_title(model, loc="left", fontsize=10)
    axes[0].set_ylabel("median wait (s, log)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", title="CPU pool", title_fontsize=8.5)
    fig.suptitle("A CPU tier behind the GPU pool", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def context_growth(rows):
    """rows: [(label, tokens already cached, tokens prefilled now, tokens generated)], one per request."""
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.2))
    ys = np.arange(len(rows))[::-1]
    for y, (label, cached, new, out) in zip(ys, rows):
        ax.barh(y, cached, color=CACHED, height=0.62)
        ax.barh(y, new, left=cached, color=COMPUTED, height=0.62)
        ax.barh(y, out, left=cached + new, color=CATEGORICAL[2], height=0.62)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("tokens in the context")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CACHED, COMPUTED, CATEGORICAL[2])]
    ax.legend(handles, ["already in the cache", "prefilled by this request", "generated by this request"],
              frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3)
    ax.set_title("One agent session, request by request", loc="left", fontsize=11)
    fig.tight_layout()
    return fig
