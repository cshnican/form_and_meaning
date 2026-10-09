"""
Run the tradeoff model across a spectrum of societies and produce figures.

Narrative order (written to ./figures):
  1. cost_vs_society.png         -- sanity check: mixed encoding beats naive strategies
  2. shared_knowledge.png        -- how far conventions can be known, by society type
  3. encoding_and_lexicon.png    -- which items to conventionalize, and the resulting lexicon
"""

from __future__ import annotations  # postponed evaluation of type hints

import os  # mkdir / path join / remove stale figures
from pathlib import Path  # locate this study folder

import matplotlib  # plotting library

matplotlib.use("Agg")  # write PNGs without opening a window
import matplotlib.pyplot as plt  # axes-level plotting API
import numpy as np  # grids of E and frequency rank
from matplotlib.colors import ListedColormap  # two-color encoding heatmap
from matplotlib.patches import Patch  # discrete legend swatches

from society_model import (  # cost-model primitives used by every figure
    ModelParams,  # default knobs
    evaluate_society,  # per-society optimum
    knowledge_prob,  # q_i(E, f)
    sweep_exposure,  # evaluate a range of E
    zipf_frequencies,  # Zipf frequency vector
)

ROOT = Path(__file__).resolve().parent  # this study's directory
FIG_DIR = str(ROOT / "figures")  # where PNGs are written
SOCIETIES = [("open", 20.0), ("loose", 80.0), ("mid", 300.0), ("close-knit", 5000.0)]  # named E values


def _mark_archetypes(ax) -> None:
    """Vertical guides; names sit *above* the axes so they cannot hit curves."""
    specs = [(20.0, "open", "left"), (5000.0, "close-knit", "right")]  # two labeled society poles
    for E, name, ha in specs:  # one guide per pole
        ax.axvline(E, color="0.6", lw=1, ls=(0, (2, 3)), zorder=1)  # dashed vertical line at that E
        ax.text(  # label sitting above the axes
            E, 1.03, f"{name}\nE={E:g}", color="0.35", fontsize=9,  # name and numeric E
            va="bottom", ha=ha, transform=ax.get_xaxis_transform(),  # x in data units, y in axes units
            clip_on=False,  # allow the label to sit outside the plot box
        )


def _sweep(params: ModelParams):
    E_grid = np.geomspace(10, 8000, 80)  # log-spaced shared-exposure values
    results = sweep_exposure(E_grid, params)  # evaluate the model at each E
    return E_grid, results  # grid and matching result list


def fig_cost_vs_society(params: ModelParams) -> None:
    E_grid, results = _sweep(params)  # mixed-encoding cost across societies
    total = [r.total_cost for r in results]  # optimal mixed cost at each E
    all_comp = results[0].total_cost_all_comp  # all-transparent cost (flat in E)
    all_conv = [r.total_cost_all_conv for r in results]  # all-conventional cost (U-shaped)

    fig, ax = plt.subplots(figsize=(8, 5.8))  # single-panel figure
    ax.plot(E_grid, total, lw=2.5, color="C3", label="OPTIMAL: best choice per meaning")  # mixed strategy
    ax.axhline(all_comp, ls="--", lw=2, color="C0",  # horizontal baseline
               label="naive: everything transparent (flat — needs no sharing)")  # no dependence on E
    ax.plot(E_grid, all_conv, ls=":", lw=2.5, color="C2",  # naive conventionalize-everything
            label="naive: everything conventional")  # expensive at both low and high E
    ax.set_xscale("log")  # E spans orders of magnitude
    ax.set_ylim(2500, 6400)  # keep the three curves comparable
    ax.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")  # x label
    ax.set_ylabel("communication and learning cost   (lower = better)")  # y label
    ax.set_title("Mixed encoding beats both naive strategies")  # claim this figure is checking
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18),  # legend below the plot
              fontsize=9, ncol=1)  # one column
    ax.grid(alpha=0.3)  # light grid

    fig.tight_layout()  # pack legend and axes
    out = os.path.join(FIG_DIR, "cost_vs_society.png")  # output path
    fig.savefig(out, dpi=140)  # write PNG
    plt.close(fig)  # free the figure
    print("wrote", out)  # console confirmation


def fig_shared_knowledge(params: ModelParams) -> None:
    f = zipf_frequencies(params.V, params.zipf_s)  # frequency of each meaning
    ranks = np.arange(1, params.V + 1)  # rank axis (1 = most frequent)

    fig, ax = plt.subplots(figsize=(8.2, 5))  # single-panel figure
    for label, E in SOCIETIES:  # one knowledge curve per named society
        ax.plot(ranks, knowledge_prob(f, E, params.lam), label=f"{label} (E={E:g})")  # q vs rank
    ax.set_xscale("log")  # rank is heavy-tailed
    ax.set_xlabel("meaning, by frequency rank (1 = most frequent)")  # x label
    ax.set_ylabel("P(random hearer knows the convention),  $q_i$")  # y label
    ax.set_title("Shared exposure determines how far conventions can be known")  # claim
    ax.legend(frameon=True, fancybox=False, facecolor="white", edgecolor="none",  # legend to the right
              loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)  # outside the axes
    ax.grid(alpha=0.3)  # light grid

    fig.tight_layout()  # make room for the external legend
    out = os.path.join(FIG_DIR, "shared_knowledge.png")  # output path
    fig.savefig(out, dpi=140)  # write PNG
    plt.close(fig)  # free the figure
    print("wrote", out)  # console confirmation


def fig_encoding_and_lexicon(params: ModelParams) -> None:
    f = zipf_frequencies(params.V, params.zipf_s)  # frequencies (used only to set V-length ranks)
    ranks = np.arange(1, params.V + 1)  # rank axis for the heatmap
    E_grid_hm = np.geomspace(10, 8000, 60)  # coarser E grid for the heatmap rows
    frontier = np.array([evaluate_society(E, params).conventionalize for E in E_grid_hm])  # bool matrix

    E_grid, results = _sweep(params)  # finer sweep for the line panel
    frac_types = [r.frac_conv_types for r in results]  # % of meanings conventionalized
    frac_tokens = [r.frac_conv_tokens for r in results]  # % of usage that is conventional
    shared = [r.shared_vocab_frac for r in results]  # % of items with q >= 0.5

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.2))  # heatmap | lexicon composition

    cmap = ListedColormap(["#1f77b4", "#d62728"])  # transparent, conventional
    ax0.pcolormesh(  # binary encoding heatmap
        ranks, E_grid_hm, frontier.astype(float),  # x=rank, y=E, color=choice
        shading="auto", cmap=cmap, vmin=0, vmax=1,  # two discrete colors
    )
    ax0.set_xscale("log")  # rank is heavy-tailed
    ax0.set_yscale("log")  # E spans orders of magnitude
    ax0.set_xlabel("meaning, by frequency rank")  # x label
    ax0.set_ylabel("society: shared exposure  $E$  (open → close-knit)")  # y label
    ax0.set_title("(a) Which meanings to conventionalize")  # panel claim
    ax0.legend(  # discrete legend instead of a colorbar
        handles=[
            Patch(facecolor="#d62728", edgecolor="none", label="conventional"),  # red
            Patch(facecolor="#1f77b4", edgecolor="none", label="transparent"),  # blue
        ],
        frameon=True, fancybox=False, facecolor="white", edgecolor="none",  # white box
        loc="lower right", fontsize=9,  # sit on the rare/open corner
    )

    ax1.plot(E_grid, shared, lw=2, label="% lexicon shareable ($q \\geq 0.5$)")  # how far q reaches
    ax1.plot(E_grid, frac_tokens, lw=2, label="% of usage that is conventional")  # token share
    ax1.plot(E_grid, frac_types, lw=2, label="% of meanings conventionalized")  # type share
    ax1.set_xscale("log")  # E on a log axis
    ax1.set_ylim(0, 1.0)  # fractions
    ax1.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")  # x label
    ax1.set_ylabel("fraction")  # y label
    ax1.set_title("(b) What the resulting lexicon looks like")  # panel claim
    ax1.legend(  # three-curve legend
        frameon=True, fancybox=False, facecolor="white", edgecolor="none",  # white box
        loc="upper left", fontsize=8, bbox_to_anchor=(0.18, 0.98),  # inset
    )
    ax1.grid(alpha=0.3)  # light grid
    _mark_archetypes(ax1)  # open / close-knit guides

    fig.tight_layout(rect=(0, 0, 1, 0.92))  # leave a little top margin for the E labels
    out = os.path.join(FIG_DIR, "encoding_and_lexicon.png")  # output path
    fig.savefig(out, dpi=140)  # write PNG
    plt.close(fig)  # free the figure
    print("wrote", out)  # console confirmation


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)  # create figures/ if needed
    params = ModelParams()  # default knobs
    fig_cost_vs_society(params)  # figure 1
    fig_shared_knowledge(params)  # figure 2
    fig_encoding_and_lexicon(params)  # figure 3
    for stale in ("society_profiles.png", "lexicon_vs_society.png"):  # old filenames from earlier drafts
        path = os.path.join(FIG_DIR, stale)  # path of a leftover figure
        if os.path.exists(path):  # only delete if it is still there
            os.remove(path)  # drop the stale PNG
            print("removed", path)  # say so
    print("\nAll figures written to ./figures/")  # done


if __name__ == "__main__":
    main()  # run all three figures when invoked as a script
