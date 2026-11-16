"""Reproduce the article's analytical examples and figures, not GPU timings.

Python 3; pip install numpy matplotlib
Run this file from any directory. Output is written beside the script.
Source/config revisions are recorded in the article and source_manifest.json.
Hardware values: NVIDIA Blackwell Ultra datasheet, Oct 2025, p. 5.
Logical bytes, packed allocation and ideal HBM traffic are distinct quantities.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
D, H, LAYERS, EXPERTS, TOPK = 5120, 2304, 40, 384, 6
FP4_BYTES = 0.5 + 1 / 32
PE = 3 * D * H
ACTIVE = PE * LAYERS * TOPK
STEP_MS = 2.71
BW_B300 = 7.7e12
BW_GB300 = 8e12
DENSE_FP8_B300 = 4.5e15


def packed_expert(tp):
    inter = H // tp
    pad128 = ((inter + 127) // 128) * 128
    pad64 = ((inter + 63) // 64) * 64
    # Exact sgl_expert_bytes / sgl_w2_row_bytes formulas from packing.rs.
    w13 = 2 * pad128 * D // 2 + (2 * pad128 // 128) * (D // 128) * 512
    w2 = D * pad64 // 2 + (D // 128) * (pad128 // 128) * 512
    return {'inter': inter, 'w13_pad': pad128, 'w2_payload_channels': pad64,
            'per_rank_bytes': w13 + w2,
            'all_rank_bytes': tp * (w13 + w2),
            'storage_overhead_fraction': tp * (w13 + w2) / (PE * FP4_BYTES) - 1}


def calculations():
    union = EXPERTS * (1 - (1 - TOPK / EXPERTS) ** 7)
    return {
        'kind': 'analytical calculations, not GPU measurements',
        'reported_rate_implied_commit_length': {'TP4': 1740 * .00330, 'TP8': 2060 * .00271},
        'headline_equivalent_ms_per_token': 1000 / 2060,
        'reported_rate_ratio_TP8_TP4': 2060 / 1740,
        'expert_weights': PE,
        'all_routed_weights': PE * LAYERS * EXPERTS,
        'active_routed_weights_per_target_row': ACTIVE,
        'logical_fp4_bytes_per_weight': FP4_BYTES,
        'active_logical_routed_weight_bytes': ACTIVE * FP4_BYTES,
        'logical_expert_bytes': PE * FP4_BYTES,
        'packing': {str(tp): packed_expert(tp) for tp in [4, 8]},
        'seven_row_toy_expert_union': union,
        'weight_only_intensity': {'one_or_disjoint': 2 / FP4_BYTES,
                                  'seven_identical': 14 / FP4_BYTES,
                                  'ratio_of_expected_work_to_expected_bytes_uniform': 2 * 7 * TOPK / (FP4_BYTES * union)},
        'ideal_component_HBM_ms': {
            'single_row_TP4_GB300': ACTIVE * FP4_BYTES / (4 * BW_GB300) * 1000,
            'single_row_TP8_B300': ACTIVE * FP4_BYTES / (8 * BW_B300) * 1000,
            'seven_rows_uniform_TP8_B300': ACTIVE * FP4_BYTES * union / TOPK / (8 * BW_B300) * 1000,
        },
        'fused_wqkv': {'m': 7, 'k': D, 'n': 1792,
                       'weight_elements': D * 1792,
                       'useful_FLOPs': 2 * 7 * D * 1792,
                       'ideal_FP8_payload_HBM_us': D * 1792 / BW_B300 * 1e6,
                       'ideal_dense_FP8_math_us': 2 * 7 * D * 1792 / DENSE_FP8_B300 * 1e6},
        'eliminated_BF16_write_read_bytes_7x5120': 2 * 7 * D * 2,
        'FP8_B300_hardware_balance_FLOPs_per_byte': DENSE_FP8_B300 / BW_B300,
        'prefill_layer_rows_N_1M': {'conventional': 40_000_000, 'bounded': 20_002_560,
                                   'ratio': 20_002_560 / 40_000_000},
    }


plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none',
                     'figure.facecolor': 'white', 'axes.labelcolor': '#263444',
                     'text.color': '#263444', 'axes.edgecolor': '#9aa7b2',
                     'savefig.facecolor': 'white'})
TEAL, ORANGE, BLUE = '#087f83', '#bd5730', '#456a9b'


def finish(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    # PNG is for local visual inspection; SVG is used in the article.
    fig.savefig(OUT / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figures():
    intensity = np.logspace(-.3, 3.4, 500)
    ceiling = np.minimum(DENSE_FP8_B300, intensity * BW_B300) / 1e12
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    ax.loglog(intensity, ceiling, color=TEAL, lw=2.6)
    ax.axvline(DENSE_FP8_B300 / BW_B300, ls=':', color='#96a3ad')
    for val, label, offset in [(2, '1 row: ~2 FLOPs/byte', (15, 25)),
                               (14, '7 shared-weight rows: ~14', (15, -25))]:
        y = val * BW_B300 / 1e12
        ax.scatter(val, y, color=ORANGE, s=45, zorder=3)
        ax.annotate(label, (val, y), xytext=offset, textcoords='offset points',
                    fontsize=10, arrowprops={'arrowstyle': '-', 'color': '#8796a0'})
    ax.text(660, 800, 'Crossover\n~584 FLOPs/byte', fontsize=10)
    ax.set(xlabel='Arithmetic intensity (FLOPs / HBM byte)',
           ylabel='Ideal useful throughput (TFLOPs/s)',
           title='A tiny FP8 projection remains below the compute roof')
    ax.grid(which='major', alpha=.15)
    fig.supxlabel('Analytical envelope: 7.7 TB/s HBM, 4.5 PFLOPS dense FP8; no kernel timings.',
                   fontsize=9, color='#637282')
    finish(fig, 'roofline')

    k = np.linspace(1, 7, 100)
    fig, ax = plt.subplots(figsize=(9, 4.2), layout='constrained')
    ax.plot(k, k / (STEP_MS / 1000), color=TEAL, lw=2.6)
    implied = 2060 * STEP_MS / 1000
    ax.scatter(implied, 2060, color=ORANGE, s=50, zorder=3)
    ax.annotate('Headline arithmetic\n5.58 tokens / 2.71 ms ≈ 2,060 TPS',
                (implied, 2060), xytext=(2.3, 2250), fontsize=10,
                arrowprops={'arrowstyle': '->', 'color': ORANGE})
    ax.set(xlabel='Average committed tokens per step', ylabel='Emitted tokens / second',
           title='Useful output per step changes the headline rate',
           xticks=np.arange(1, 8), ylim=(0, 2800))
    ax.grid(alpha=.15)
    fig.supxlabel('Sensitivity only: step time held fixed at 2.71 ms. Real draft-length changes also change time.',
                   fontsize=9, color='#637282')
    finish(fig, 'speculation')

    rows = np.arange(1, 8)
    union = EXPERTS * (1 - (1 - TOPK / EXPERTS) ** rows)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    ax.plot(rows, rows, '--', color=ORANGE, lw=2, label='Disjoint selections')
    ax.plot(rows, union / TOPK, 'o-', color=BLUE, lw=2, label='Independent uniform routing (toy model)')
    ax.plot(rows, np.ones_like(rows), color=TEAL, lw=2.5, label='Identical expert selections')
    ax.set(xlabel='Token rows in the verification block',
           ylabel='Logical expert-weight bytes / one-row bytes',
           title='More speculative rows need not reuse the same experts', xticks=rows)
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    ax.grid(alpha=.15)
    fig.supxlabel('384 experts, top-6 per row; each used expert fetched once. Excludes padding and other traffic.',
                   fontsize=9, color='#637282')
    finish(fig, 'moe_reuse')


if __name__ == '__main__':
    values = calculations()
    (OUT / 'calculations.json').write_text(json.dumps(values, indent=2) + '\n')
    figures()
    print(json.dumps(values, indent=2))
