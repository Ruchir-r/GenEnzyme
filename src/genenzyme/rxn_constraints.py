#!/usr/bin/env python3
"""Grammar + conservation constraints for metabolite-level reaction decoding.

The em_ext vocabulary is METABOLITE-level, not character-level: ids 0-4 are
<START>/<END>/<PAD>/<MASK>/<UNK>, id 5 is '>>', id 6 is '.', and ids 7+ are
whole canonical metabolite SMILES. That makes per-token atom composition a
lookup rather than a parse, so composition can be tracked during decoding.

Two constraint tiers, deliberately separated because their evidential status
differs:

  grammar     -- provably satisfied by EVERY training reaction, so it forbids
                 nothing the data contains:
                   * '>>' appears at most once (a reaction has one arrow)
                   * no empty components ('.' cannot follow <START>, '>>', '.')
                   * a metabolite repeats on one side at most --mult-cap times
                     (measured max in em_ext train = 4: LHS 22 reactions at 4x,
                     RHS 2 at 4x; 2.30 pct of test reactions repeat at all, so a
                     hard no-repeat rule would make them ungenerable)
                   * <END> only after at least one product metabolite
  composition -- additionally gates product tokens on heavy-atom reachability
                 against the reactant side.

Provenance note on the composition tier: a web search in the originating
session returned the title+URL of "MARLIN: De Novo Molecular Structure
Elucidation from Tandem Mass Spectra without a Ground-Truth Formula"
(arXiv 2607.04774), which is a real result and appears to be the nearest
published analogue -- a mass-budget-gated decoder. The search result carried
NO abstract or body text, so the specific mechanism attributed to it in the
originating discussion (forbid stopping while the target mass is
unreachable, permit it once reachable) is UNVERIFIED RECALL, not something
read from the paper. Read it before citing it, and do not treat it as
validating the design below. The design stands on this project's own
measurement instead: heavy-atom conservation is satisfied by 100.00 pct of
true reactions on both test and val, and conservation-based RE-RANKING of an
existing pool lifted strict top-1 from 0.89 pct to 10.39 pct, which is why
the same signal is worth applying during decoding.

IMPORTANT asymmetry: the reactant side here is GENERATED, not teacher-forced.
So composition gating enforces internal consistency of the model's own
reaction, NOT agreement with the true reactants. It is therefore NOT provably
lossless: if the generated reactant side is wrong, forcing the product side to
balance against it can exclude the true product. Grammar tier is safe; keep
composition behind its own flag and measure it.

Heavy-atom accounting excludes BOTH implicit and explicit hydrogen -- em_ext
carries free protons as their own '[H+]' species, so counting explicit H makes
the test proton-sensitive again (measured: truth passes heavy-atom
conservation 100.00 pct on test and val; including H it is only 56.19 pct).
"""
import json
import os

import torch

N_SPECIAL = 7          # ids 0..6 are special / structural, never metabolites


def build_comp_table(vocab, cache_path, device):
    """(V, E) int16 tensor of heavy-atom counts per vocabulary token.

    Cached to JSON: parsing 16k SMILES with RDKit takes ~30 s, and every
    decode job would otherwise repeat it.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            blob = json.load(fh)
        elements = blob["elements"]
        table = torch.tensor(blob["table"], dtype=torch.int16)
        return table.to(device), elements

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    V = len(vocab)
    inv = {i: t for t, i in vocab.items()}
    per_tok = {}
    elements = set()
    n_bad = 0
    for tid in range(N_SPECIAL, V):
        smi = inv.get(tid)
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            n_bad += 1
            continue
        c = {}
        for a in m.GetAtoms():
            sym = a.GetSymbol()
            if sym == "H":       # explicit H / [H+] species: not a heavy atom
                continue
            c[sym] = c.get(sym, 0) + 1
        per_tok[tid] = c
        elements.update(c)

    elements = sorted(elements)
    eidx = {e: k for k, e in enumerate(elements)}
    table = torch.zeros(V, len(elements), dtype=torch.int16)
    for tid, c in per_tok.items():
        for e, k in c.items():
            table[tid, eidx[e]] = k

    print("[constraints] comp table: V=%d elements=%d unparseable=%d"
          % (V, len(elements), n_bad), flush=True)
    if cache_path:
        with open(cache_path, "w") as fh:
            json.dump({"elements": elements, "table": table.tolist()}, fh)
    return table.to(device), elements


class ReactionConstraints:
    """Callable logits processor: (logits (H,V), seqs (H,L)) -> logits.

    State is recomputed from the full token prefix every step rather than
    carried incrementally. That is deliberate: beam search REORDERS
    hypotheses each step (gather_beam_idx), so any incrementally-maintained
    per-hypothesis state would silently attach to the wrong beam. Sequences
    here are short (<= ~10 tokens: max 7 components + '>>' + <END>), so
    recomputation is cheap and cannot desynchronise.
    """

    def __init__(self, vocab, device, mult_cap=4, composition=False,
                 comp_table=None):
        self.RXN = vocab[">>"]
        self.DOT = vocab["."]
        self.START = vocab["<START>"]
        self.END = vocab["<END>"]
        self.PAD = vocab["<PAD>"]
        self.MASKT = vocab.get("<MASK>", 3)
        self.UNK = vocab.get("<UNK>", 4)
        self.mult_cap = int(mult_cap)
        self.composition = bool(composition)
        self.comp = comp_table
        if self.composition:
            assert self.comp is not None, "composition=True needs comp_table"
        self.n_calls = 0

    def __call__(self, logits, seqs):
        H, V = logits.shape
        L = seqs.size(1)
        dev = logits.device
        self.n_calls += 1

        pos = torch.arange(L, device=dev)
        is_rxn = seqs == self.RXN
        after = is_rxn.any(1)                                   # (H,) past the arrow
        last_rxn = torch.where(is_rxn, pos.unsqueeze(0),
                               torch.full_like(seqs, -1)).max(1).values
        last = seqs[:, -1]

        allow = torch.ones(H, V, dtype=torch.bool, device=dev)

        # control tokens are never legal continuations
        for t in (self.START, self.PAD, self.MASKT, self.UNK):
            allow[:, t] = False

        # '>>' at most once, and not straight after <START> or '.'
        allow[:, self.RXN] &= ~after
        allow[:, self.RXN] &= ~((last == self.START) | (last == self.DOT))

        # no empty components
        allow[:, self.DOT] &= ~((last == self.START) | (last == self.RXN)
                                | (last == self.DOT))

        # <END> requires being past '>>' with at least one product metabolite
        allow[:, self.END] &= after & (last >= N_SPECIAL)

        # per-side multiplicity cap (positions since the arrow, excluding BOS)
        valid = pos.unsqueeze(0) > torch.clamp(last_rxn, min=0).unsqueeze(1)
        src = torch.where(valid, seqs, torch.full_like(seqs, self.PAD))
        counts = torch.zeros(H, V, dtype=torch.int16, device=dev)
        counts.scatter_add_(1, src, torch.ones_like(src, dtype=torch.int16))
        over = counts >= self.mult_cap
        over[:, :N_SPECIAL] = False
        allow &= ~over

        if self.composition:
            comp = self.comp                                     # (V,E) int16
            cs = comp[seqs.clamp(min=0)].to(torch.int32)          # (H,L,E)
            react = ((pos.unsqueeze(0) > 0)
                     & (pos.unsqueeze(0) < last_rxn.unsqueeze(1)))
            prod = pos.unsqueeze(0) > last_rxn.unsqueeze(1)
            target = (cs * react.unsqueeze(-1)).sum(1)            # (H,E)
            running = (cs * prod.unsqueeze(-1)).sum(1)            # (H,E)
            remaining = target - running

            # a token is illegal if ANY element would overshoot what is left.
            # Chunked over elements: the (H,V,E) broadcast would be ~77M
            # entries at H=400, which is not worth allocating.
            bad = torch.zeros(H, V, dtype=torch.bool, device=dev)
            compi = comp.to(torch.int32)
            for e in range(comp.size(1)):
                bad |= compi[:, e].unsqueeze(0) > remaining[:, e].unsqueeze(1)
            bad &= after.unsqueeze(1)
            bad[:, :N_SPECIAL] = False
            allow &= ~bad
            # and stopping is only legal once the budget is exactly met
            allow[:, self.END] &= (remaining == 0).all(1)

        # Never hand back an all-masked row: softmax over all -inf is NaN, and
        # a NaN here poisons the whole batch rather than one hypothesis.
        dead = ~allow.any(1)
        if dead.any():
            allow[dead, self.END] = True

        return logits.masked_fill(~allow, torch.finfo(logits.dtype).min)
