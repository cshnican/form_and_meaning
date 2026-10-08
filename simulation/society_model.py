"""
Core cost model for the compositionality <-> conventionalization tradeoff.

Idea
----
Each meaning can be encoded in one of two ways:

  * TRANSPARENT: build the utterance from parts. It is LONG (high production
    cost) but any hearer can decode it with no prior memory.

  * CONVENTIONAL: use a short, memorized, opaque form. It is SHORT (low
    production cost) but must be LEARNED, and it only works if the *hearer* also
    shares the convention. If they don't, the message fails and the speaker has
    to fall back on a transparent description.

A "language" (or a population) chooses, per meaning, whichever encoding gives
lower expected communication and learning cost. Two costs trade off:

  * production cost   -- paid EVERY time the meaning is used  (favours short)
  * learning/memory   -- paid ONCE per person who stores it   (favours few)

The key amortization: memory cost is spread over (how often the item is used)
x (how many people share it). So conventionalizing an item pays off only when
it is BOTH frequent AND widely shared.

Society type enters through *shared exposure* E: how many community-wide
exposures a typical learner accumulates. The probability that a random person
knows the convention for item i is modelled as a saturating function of
(exposure x frequency):

        q_i = 1 - exp(-lambda * E * f_i)

  * Close-knit society  -> large E -> q_i ~ 1 even for rare items
                           (the whole lexicon can be shared/opaque).
  * Open/complex society -> small E -> q_i ~ 1 only for the frequent core
                           (the long tail cannot be commonly shared).

So "% of shared vocabulary" is an *output* of the society's exposure level,
and it is the quantity that caps how far down the frequency distribution
conventionalization can profitably reach.
"""

from __future__ import annotations  # postponed evaluation of type hints

from dataclasses import dataclass, field  # structured parameter and result objects

import numpy as np  # vectorized frequencies, knowledge, and costs


@dataclass
class ModelParams:
    """All free parameters of the analytic cost model."""

    V: int = 1000  # vocabulary size (number of distinct meanings)
    zipf_s: float = 1.0  # Zipf exponent for the frequency distribution
    horizon: float = 1000.0  # T: communication events a typical agent takes part in

    # production costs (in arbitrary "effort" units, e.g. syllables)
    cost_comp: float = 6.0  # L : cost of a transparent utterance
    cost_conv: float = 1.0  # l : cost of a conventional (short) utterance
    # on a failed conventional attempt the speaker repairs by describing it;
    # repair cost defaults to a full transparent utterance.
    cost_repair: float = 6.0  # extra cost paid when the hearer does not know the convention

    # learning / memory
    mem_cost: float = 4.0  # m : cost of storing one conventional form (per knower)

    # knowledge model
    lam: float = 1.0  # lambda : exposures -> knowledge conversion rate

    def __post_init__(self) -> None:
        if self.cost_conv >= self.cost_comp:  # a short form must actually be cheaper to produce
            raise ValueError("conventional forms should be cheaper to produce than transparent ones")


def zipf_frequencies(V: int, s: float = 1.0) -> np.ndarray:
    """Return normalized Zipf frequencies for V meanings (rank 1 = most frequent)."""
    ranks = np.arange(1, V + 1, dtype=float)  # rank 1..V
    w = ranks ** (-s)  # unnormalized Zipf weights
    return w / w.sum()  # convert to probabilities that sum to 1


def knowledge_prob(freqs: np.ndarray, exposure: float, lam: float = 1.0) -> np.ndarray:
    """
    Probability a random community member knows the convention for each item.

    `exposure` (E) is the society knob: high = close-knit, low = open.
    Saturating in exposure x frequency, so frequent items are learned by more
    people, and closer-knit societies push knowledge further into the tail.
    """
    return 1.0 - np.exp(-lam * exposure * freqs)  # Poisson "at least one exposure" knowledge curve


@dataclass
class ModelResult:
    """Per-item and aggregate outputs for one society."""

    exposure: float  # E used for this evaluation
    freqs: np.ndarray  # Zipf frequency of each meaning
    q: np.ndarray  # P(random hearer knows convention), per item
    conventionalize: np.ndarray  # bool: optimal choice per item
    cost_comp_item: np.ndarray  # per-item cost if transparent
    cost_conv_item: np.ndarray  # per-item cost if conventional
    cost_item: np.ndarray  # per-item cost under the optimal choice
    total_cost: float  # sum over items (optimal)
    total_cost_all_comp: float  # baseline: everything transparent
    total_cost_all_conv: float  # baseline: everything conventional
    # summaries of "shared vocabulary"
    frac_conv_types: float  # fraction of meanings conventionalized (count)
    frac_conv_tokens: float  # fraction of *usage* that is conventional
    shared_vocab_frac: float  # fraction of items with q >= 0.5
    mean_q_tokens: float  # usage-weighted mean knowledge (sum f_i q_i)

    extras: dict = field(default_factory=dict)  # unused hook for extra diagnostics


def evaluate_society(exposure: float, params: ModelParams | None = None) -> ModelResult:
    """
    Compute per-item costs, the optimal per-item strategy, and aggregates for a
    society defined by its shared-exposure level `exposure`.
    """
    p = params or ModelParams()  # default knobs if the caller passed none
    f = zipf_frequencies(p.V, p.zipf_s)  # frequency of each meaning
    q = knowledge_prob(f, exposure, p.lam)  # share of hearers who know each convention
    T = p.horizon  # number of communication events in the accounting window

    # Expected cost per item over the horizon, for each encoding choice.
    #   transparent   : used f*T times, each costs L, no memory.
    cost_comp_item = f * T * p.cost_comp  # production only; meaning is inferable

    #   conventional  : used f*T times; each success costs l, each failure costs
    #                   l + repair. Failure prob = (1 - q) (hearer doesn't know).
    #                   plus memory cost m paid by the expected q-fraction of the
    #                   population who store it (one knower's cost, amortized by
    #                   folding population size into `mem_cost`).
    per_event_conv = p.cost_conv + (1.0 - q) * p.cost_repair  # expected production+repair per use
    cost_conv_item = f * T * per_event_conv + p.mem_cost * q  # add amortized learning cost

    conventionalize = cost_conv_item < cost_comp_item  # conventionalize iff it is cheaper
    cost_item = np.where(conventionalize, cost_conv_item, cost_comp_item)  # pick the cheaper encoding

    total_cost = float(cost_item.sum())  # mixed (optimal) lexicon cost
    total_all_comp = float(cost_comp_item.sum())  # naive all-transparent baseline
    total_all_conv = float(cost_conv_item.sum())  # naive all-conventional baseline

    frac_conv_types = float(conventionalize.mean())  # share of types that are conventional
    frac_conv_tokens = float(f[conventionalize].sum())  # share of usage that is conventional
    shared_vocab_frac = float((q >= 0.5).mean())  # share of items a typical hearer likely knows
    mean_q_tokens = float((f * q).sum())  # usage-weighted knowledge

    return ModelResult(  # pack per-item arrays and scalar summaries
        exposure=exposure,  # society knob used
        freqs=f,  # Zipf frequencies
        q=q,  # knowledge probabilities
        conventionalize=conventionalize,  # optimal encoding mask
        cost_comp_item=cost_comp_item,  # transparent costs
        cost_conv_item=cost_conv_item,  # conventional costs
        cost_item=cost_item,  # chosen costs
        total_cost=total_cost,  # mixed total
        total_cost_all_comp=total_all_comp,  # all-transparent total
        total_cost_all_conv=total_all_conv,  # all-conventional total
        frac_conv_types=frac_conv_types,  # type share
        frac_conv_tokens=frac_conv_tokens,  # token share
        shared_vocab_frac=shared_vocab_frac,  # q>=0.5 share
        mean_q_tokens=mean_q_tokens,  # usage-weighted q
    )


def sweep_exposure(exposures: np.ndarray, params: ModelParams | None = None) -> list[ModelResult]:
    """Evaluate a range of societies (from open -> close-knit)."""
    return [evaluate_society(float(e), params) for e in exposures]  # one result per E


if __name__ == "__main__":
    # Quick smoke test across three archetypal societies.
    p = ModelParams()  # default parameter set
    for label, E in [("open", 20.0), ("mid", 200.0), ("close-knit", 5000.0)]:  # three society types
        r = evaluate_society(E, p)  # run the cost comparison at this E
        print(  # one-line summary for human inspection
            f"{label:11s} E={E:7.0f} | "
            f"conv types={r.frac_conv_types:5.1%} | "
            f"conv tokens={r.frac_conv_tokens:5.1%} | "
            f"shared vocab={r.shared_vocab_frac:5.1%} | "
            f"communication and learning cost={r.total_cost:9.1f} "
            f"(all-comp={r.total_cost_all_comp:.0f}, all-conv={r.total_cost_all_conv:.0f})"
        )
