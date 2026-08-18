"""
Bottom-up (agent-based) version of the tradeoff model.

The analytic model in `society_model.py` is a mean-field summary. This file
re-derives the same behaviour from the ground up with individual agents, to
check that the aggregate story actually falls out of concrete interactions:

  1. EXPOSURE: each agent independently accumulates exposures to meanings,
     sampled by frequency and scaled by the society's shared-exposure level E.
     An agent "knows" a convention once it has been exposed at least once.
     -> the population-level knowledge curve q_i emerges from this, and should
        match 1 - exp(-lambda * E * f_i).

  2. COMMUNICATION: many random speaker/hearer dialogues. For each, a meaning is
     drawn by frequency and encoded per the society's strategy (which items are
     conventional). A conventional attempt succeeds only if BOTH parties know
     it; otherwise the speaker repairs compositionally.

  3. COST: realized production + repair cost (per agent) plus memory cost for the
     conventions each agent actually stores.

We then check that (a) empirical q matches the analytic curve and (b) the
analytic optimal strategy beats the all-compositional and all-conventional
baselines in realized cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from society_model import ModelParams, evaluate_society, knowledge_prob, zipf_frequencies


@dataclass
class SimResult:
    exposure: float
    emp_q: np.ndarray            # empirical P(knows) per item
    cost_per_agent_optimal: float
    cost_per_agent_all_comp: float
    cost_per_agent_all_conv: float


def simulate_knowledge(freqs: np.ndarray, exposure: float, n_agents: int,
                       lam: float, rng: np.random.Generator) -> np.ndarray:
    """
    Return an (n_agents x V) boolean matrix of who knows which convention.

    Each agent gets Poisson(lam * E * f_i) exposures to item i and knows it if
    exposed at least once -> P(knows) = 1 - exp(-lam * E * f_i), bottom-up.
    """
    rates = lam * exposure * freqs                      # expected exposures per agent
    counts = rng.poisson(lam=rates, size=(n_agents, freqs.size))
    return counts > 0


def realized_cost(knows: np.ndarray, freqs: np.ndarray, strategy: np.ndarray,
                  params: ModelParams, n_events: int, rng: np.random.Generator) -> float:
    """
    Mean total cost PER AGENT for a given per-item strategy (True = conventional).

    Production/repair cost is measured over `n_events` random dialogues and
    normalized to a per-agent horizon; memory cost counts the conventions each
    agent actually stores.
    """
    n_agents, V = knows.shape

    # --- production + repair cost over random dialogues ---
    meanings = rng.choice(V, size=n_events, p=freqs)
    speakers = rng.integers(0, n_agents, size=n_events)
    hearers = rng.integers(0, n_agents, size=n_events)

    use_conv = strategy[meanings]                       # society wants a convention here
    spk_knows = knows[speakers, meanings]
    hnr_knows = knows[hearers, meanings]

    # A convention is actually used only if the society designates it AND the
    # speaker knows it; otherwise the speaker describes it compositionally.
    conv_used = use_conv & spk_knows
    success = conv_used & hnr_knows                     # short + understood
    failure = conv_used & ~hnr_knows                    # short attempt, then repair

    event_cost = np.full(n_events, params.cost_comp, dtype=float)   # default: compositional
    event_cost[success] = params.cost_conv
    event_cost[failure] = params.cost_conv + params.cost_repair

    # scale from n_events sampled dialogues to a per-agent horizon
    prod_per_agent = event_cost.mean() * params.horizon

    # --- memory cost: each agent pays for conventions it stores ---
    stored = knows & strategy[np.newaxis, :]            # knows it AND it's conventional
    mem_per_agent = params.mem_cost * stored.sum(axis=1).mean()

    return float(prod_per_agent + mem_per_agent)


def simulate_society(exposure: float, params: ModelParams, n_agents: int = 400,
                     n_events: int = 200_000, rng: np.random.Generator | None = None) -> SimResult:
    rng = rng or np.random.default_rng(0)
    f = zipf_frequencies(params.V, params.zipf_s)
    knows = simulate_knowledge(f, exposure, n_agents, params.lam, rng)
    emp_q = knows.mean(axis=0)

    analytic = evaluate_society(exposure, params)
    all_comp = np.zeros(params.V, dtype=bool)
    all_conv = np.ones(params.V, dtype=bool)

    return SimResult(
        exposure=exposure,
        emp_q=emp_q,
        cost_per_agent_optimal=realized_cost(knows, f, analytic.conventionalize, params, n_events, rng),
        cost_per_agent_all_comp=realized_cost(knows, f, all_comp, params, n_events, rng),
        cost_per_agent_all_conv=realized_cost(knows, f, all_conv, params, n_events, rng),
    )


if __name__ == "__main__":
    p = ModelParams()
    rng = np.random.default_rng(42)
    f = zipf_frequencies(p.V, p.zipf_s)
    print("Bottom-up simulation vs analytic mean-field\n")
    for label, E in [("open", 20.0), ("mid", 200.0), ("close-knit", 5000.0)]:
        sim = simulate_society(E, p, rng=rng)
        ana_q = knowledge_prob(f, E, p.lam)
        q_err = float(np.abs(sim.emp_q - ana_q).max())
        print(
            f"{label:11s} E={E:7.0f} | max|q_emp-q_analytic|={q_err:.3f} | "
            f"cost/agent: optimal={sim.cost_per_agent_optimal:8.1f}  "
            f"all-comp={sim.cost_per_agent_all_comp:8.1f}  "
            f"all-conv={sim.cost_per_agent_all_conv:8.1f}"
        )
