"""
Run the tradeoff model across a spectrum of societies and produce figures.

Outputs (written to ./figures):
  1. society_profiles.png  -- who-knows-what and the conventionalization frontier
  2. exposure_sweep.png    -- lexicon composition and total cost vs society type
  3. cost_vs_shared.png    -- results plotted against % shared vocabulary
  4. validation.png        -- bottom-up agent simulation vs analytic mean-field
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agent_simulation import simulate_society
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


def _mark_archetypes(ax, ymax: float) -> None:
    """Drop labelled guide-lines for the open and close-knit archetypes."""
    for E, name in [(20.0, "open"), (5000.0, "close-knit")]:
        ax.axvline(E, color="0.6", lw=1, ls=(0, (2, 3)))
        ax.text(E, ymax, f" {name}\n E={E:g}", color="0.35", fontsize=9,
                va="top", ha="left")


def fig_exposure_sweep(params: ModelParams) -> None:
    E_grid = np.geomspace(10, 8000, 80)
    results = sweep_exposure(E_grid, params)

    frac_types = [r.frac_conv_types for r in results]
    frac_tokens = [r.frac_conv_tokens for r in results]
    shared = [r.shared_vocab_frac for r in results]
    total = [r.total_cost for r in results]
    all_comp = results[0].total_cost_all_comp  # constant baseline
    all_conv = [r.total_cost_all_conv for r in results]

    frac_types = np.asarray(frac_types)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.6))

    # --- (a) optimal composition of the lexicon: conventional vs compositional ---
    ax0.fill_between(E_grid, 0, frac_types, color="C3", alpha=0.85,
                     label="conventionalized (short, memorized)")
    ax0.fill_between(E_grid, frac_types, 1.0, color="C0", alpha=0.85,
                     label="compositional (long, transparent)")
    ax0.plot(E_grid, frac_types, color="k", lw=1.2)
    ax0.set_xscale("log")
    ax0.set_ylim(0, 1.0)
    ax0.set_xlim(E_grid[0], E_grid[-1])
    ax0.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")
    ax0.set_ylabel("fraction of the lexicon (words / types)")
    ax0.set_title("(a) Optimal composition of the lexicon\n"
                  "(each x = a different society, its lexicon re-optimized)")
    ax0.legend(frameon=False, loc="center left", fontsize=9)
    _mark_archetypes(ax0, 0.98)

    # --- (b) cost of optimal vs the two pure strategies ---
    ax1.plot(E_grid, total, lw=2.5, color="C3", label="OPTIMAL: best choice per meaning")
    ax1.axhline(all_comp, ls="--", lw=2, color="C0",
                label="naive: everything compositional (flat — needs no sharing)")
    ax1.plot(E_grid, all_conv, ls=":", lw=2.5, color="C2",
             label="naive: everything conventional")
    ax1.set_xscale("log")
    ax1.set_xlabel("society type:  shared exposure $E$   (open  ←→  close-knit)")
    ax1.set_ylabel("total communicative cost per agent   (lower = better)")
    ax1.set_title("(b) Cost of the optimal strategy vs. naive ones\n"
                  "(gap to red = value of choosing per-meaning)")
    ax1.legend(frameon=False, loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # annotate why 'all conventional' is U-shaped
    imin = int(np.argmin(all_conv))
    ax1.annotate("all-conventional fails here\n(hearer often doesn't know it → repair)",
                 xy=(E_grid[3], all_conv[3]), xytext=(E_grid[3] * 1.1, all_conv[3] - 900),
                 fontsize=8, color="C2",
                 arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    ax1.annotate("and wastes memory here\n(stores the rare tail nobody needs short)",
                 xy=(E_grid[-1], all_conv[-1]), xytext=(E_grid[-1] * 0.16, all_conv[-1] + 250),
                 fontsize=8, color="C2",
                 arrowprops=dict(arrowstyle="->", color="C2", lw=1))
    _mark_archetypes(ax1, max(all_conv) * 1.0)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "exposure_sweep.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def fig_cost_vs_shared(params: ModelParams) -> None:
    """Everything plotted against the parameter the user cares about: % shared vocab."""
    E_grid = np.geomspace(10, 8000, 80)
    results = sweep_exposure(E_grid, params)
    shared = np.array([r.shared_vocab_frac for r in results])
    total = np.array([r.total_cost for r in results])
    frac_tokens = np.array([r.frac_conv_tokens for r in results])

    order = np.argsort(shared)
    shared, total, frac_tokens = shared[order], total[order], frac_tokens[order]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(shared * 100, total, color="C3", lw=2, label="total cost (per agent)")
    ax.set_xlabel("% of vocabulary that is shareable across the population")
    ax.set_ylabel("total communicative cost (per agent)", color="C3")
    ax.tick_params(axis="y", labelcolor="C3")

    ax2 = ax.twinx()
    ax2.plot(shared * 100, frac_tokens * 100, color="C0", lw=2, ls="--",
             label="% usage conventional")
    ax2.set_ylabel("% of usage carried by conventions", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    ax.set_title("As more of the lexicon can be shared,\ncost falls and conventions carry more of the load")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "cost_vs_shared.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def fig_validation(params: ModelParams) -> None:
    f = zipf_frequencies(params.V, params.zipf_s)
    ranks = np.arange(1, params.V + 1)
    rng = np.random.default_rng(7)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    labels, sims = [], []
    for label, E in SOCIETIES:
        sim = simulate_society(E, params, rng=rng)
        sims.append(sim)
        labels.append(label)
        ax0.plot(ranks, knowledge_prob(f, E, params.lam), color="k", alpha=0.4, lw=1)
        ax0.plot(ranks, sim.emp_q, ".", ms=2, label=f"{label} (E={E:g})")
    ax0.set_xscale("log")
    ax0.set_xlabel("meaning, by frequency rank")
    ax0.set_ylabel("P(knows) — dots = simulated, lines = analytic")
    ax0.set_title("(a) Emergent knowledge matches the mean-field")
    ax0.legend(frameon=False)
    ax0.grid(alpha=0.3)

    x = np.arange(len(labels))
    w = 0.27
    ax1.bar(x - w, [s.cost_per_agent_all_comp for s in sims], w, label="all compositional", color="C0")
    ax1.bar(x, [s.cost_per_agent_optimal for s in sims], w, label="optimal (analytic strategy)", color="C3")
    ax1.bar(x + w, [s.cost_per_agent_all_conv for s in sims], w, label="all conventional", color="C2")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("realized cost per agent (simulated)")
    ax1.set_title("(b) Optimal strategy wins in every society")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "validation.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    params = ModelParams()
    fig_society_profiles(params)
    fig_exposure_sweep(params)
    fig_cost_vs_shared(params)
    fig_validation(params)
    print("\nAll figures written to ./figures/")


if __name__ == "__main__":
    main()
