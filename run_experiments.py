"""
Run the tradeoff model across a spectrum of societies and produce figures.

Outputs (written to ./figures):
  1. society_profiles.png       -- who-knows-what and the conventionalization frontier
  2. lexicon_vs_society.png     -- lexicon composition vs society type
  3. cost_vs_society.png        -- optimal vs naive strategy cost vs society type
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from society_model import (
    ModelParams,
    evaluate_society,
    knowledge_prob,
    sweep_exposure,
    zipf_frequencies,
)

FIG_DIR = "figures"
SOCIETIES = [("open", 20.0), ("loose", 80.0), ("mid", 300.0), ("close-knit", 5000.0)]


def fig_society_profiles(params: ModelParams) -> None:
    f = zipf_frequencies(params.V, params.zipf_s)
    ranks = np.arange(1, params.V + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) knowledge curves: how deep into the frequency tail conventions are shared
    for label, E in SOCIETIES:
        ax0.plot(ranks, knowledge_prob(f, E, params.lam), label=f"{label} (E={E:g})")
    ax0.set_xscale("log")
    ax0.set_xlabel("meaning, by frequency rank (1 = most frequent)")
    ax0.set_ylabel("P(random hearer knows the convention),  $q_i$")
    ax0.set_title("(a) How far shared knowledge reaches into the tail")
    ax0.legend(frameon=False)
    ax0.grid(alpha=0.3)

    # (b) conventionalization frontier: optimal encoding per item, per society
    E_grid = np.geomspace(10, 8000, 60)
    frontier = np.array([evaluate_society(E, params).conventionalize for E in E_grid])
    mesh = ax1.pcolormesh(
        ranks, E_grid, frontier.astype(float),
        shading="auto", cmap="RdYlBu_r", vmin=0, vmax=1,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("meaning, by frequency rank")
    ax1.set_ylabel("society: shared exposure  $E$  (open → close-knit)")
    ax1.set_title("(b) Optimal encoding\n(red = conventional, blue = compositional)")
    cbar = fig.colorbar(mesh, ax=ax1, ticks=[0, 1])
    cbar.ax.set_yticklabels(["compositional", "conventional"])

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "society_profiles.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def _mark_archetypes(ax, y: float, va: str = "top") -> None:
    """Drop labelled guide-lines for the open and close-knit archetypes."""
    for E, name in [(20.0, "open"), (5000.0, "close-knit")]:
        ax.axvline(E, color="0.6", lw=1, ls=(0, (2, 3)))
        ax.text(E, y, f" {name}\n E={E:g}", color="0.35", fontsize=9,
                va=va, ha="left")


def _sweep(params: ModelParams):
    E_grid = np.geomspace(10, 8000, 80)
    results = sweep_exposure(E_grid, params)
    return E_grid, results


def fig_lexicon_vs_society(params: ModelParams) -> None:
    E_grid, results = _sweep(params)
    frac_types = [r.frac_conv_types for r in results]
    frac_tokens = [r.frac_conv_tokens for r in results]
    shared = [r.shared_vocab_frac for r in results]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(E_grid, shared, lw=2, label="% lexicon shareable ($q \\geq 0.5$)")
    ax.plot(E_grid, frac_tokens, lw=2, label="% of usage that is conventional")
    ax.plot(E_grid, frac_types, lw=2, label="% of meanings conventionalized")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")
    ax.set_ylabel("fraction")
    ax.set_title("Lexicon composition vs society type")
    ax.legend(frameon=False, loc="upper left", fontsize=9,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.grid(alpha=0.3)
    _mark_archetypes(ax, 1.06)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "lexicon_vs_society.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def fig_cost_vs_society(params: ModelParams) -> None:
    E_grid, results = _sweep(params)
    total = [r.total_cost for r in results]
    all_comp = results[0].total_cost_all_comp
    all_conv = [r.total_cost_all_conv for r in results]

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(E_grid, total, lw=2.5, color="C3", label="OPTIMAL: best choice per meaning")
    ax.axhline(all_comp, ls="--", lw=2, color="C0",
               label="naive: everything compositional (flat — needs no sharing)")
    ax.plot(E_grid, all_conv, ls=":", lw=2.5, color="C2",
            label="naive: everything conventional")
    ax.set_xscale("log")
    ax.set_ylim(top=6700)
    ax.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")
    ax.set_ylabel("total communicative cost per agent   (lower = better)")
    ax.set_title("Cost of the optimal strategy vs. naive ones\n"
                 "(gap to red = value of choosing per-meaning)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    ax.annotate("all-conventional fails here\n(hearer often doesn't know it → repair)",
                xy=(E_grid[2], all_conv[2]), xytext=(40, 4650),
                fontsize=8, color="C2",
                arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    ax.annotate("and wastes memory here\n(stores the rare tail nobody needs short)",
                xy=(E_grid[-1], all_conv[-1]), xytext=(700, 5100),
                fontsize=8, color="C2", ha="center",
                arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    _mark_archetypes(ax, 6600)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cost_vs_society.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    params = ModelParams()
    fig_society_profiles(params)
    fig_lexicon_vs_society(params)
    fig_cost_vs_society(params)
    print("\nAll figures written to ./figures/")


if __name__ == "__main__":
    main()
