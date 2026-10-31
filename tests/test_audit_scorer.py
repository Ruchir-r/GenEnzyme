"""Tests for the stereo-respecting multiset comparison used by scripts/full_rxn_audit.py.

Run:  python tests/test_audit_scorer.py        (no pytest needed)   or   pytest tests/
Requires RDKit. The scorer's `side()` returns (iso_key, flat_key): sorted canonical tokens with and
without stereochemistry. A candidate is a STRICT hit only if iso keys are equal.
"""
import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load(os.path.join(ROOT, "scripts", "full_rxn_audit.py"), "full_rxn_audit")
side = audit.side
from genenzyme.stats import wilson, mcnemar_exact, cluster_bootstrap_ci  # noqa: E402

L_ALA, D_ALA = "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"


def test_enantiomers_differ_strictly_but_match_constitutionally():
    a, b = side(L_ALA), side(D_ALA)
    assert a[0] != b[0], "strict key must separate L- and D-alanine"
    assert a[1] == b[1], "constitutional key must ignore stereochemistry"


def test_component_order_is_irrelevant():
    assert side("O.CC(=O)O") == side("CC(=O)O.O")


def test_smiles_spelling_is_canonicalised():
    assert side("OCC") == side("CCO")


def test_multiset_counts_stoichiometry():
    assert side("O.O") != side("O")
    assert len(side("O.O")[0]) == 2


def test_unparseable_or_unk_token_rejects_whole_side():
    assert side("CC.C[") is None
    assert side("CC.<UNK>") is None
    assert side("") is None


def test_wilson_matches_intervals_recorded_in_run_plan():
    # (hits, n, expected lo, expected hi) in percent, as printed in the project's re-ranking tables
    for h, n, lo, hi in [(132, 2349, 4.76, 6.63), (291, 2349, 11.12, 13.78), (2, 2349, 0.02, 0.31), (53, 2349, 1.73, 2.94)]:
        got = wilson(h, n)
        assert abs(got[0] - lo) < 0.006 and abs(got[1] - hi) < 0.006, (h, n, got)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-12
    assert abs(mcnemar_exact(0, 10) - 2 / 1024) < 1e-12
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)


def test_cluster_bootstrap_is_deterministic_and_brackets_the_estimate():
    hits = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 20
    clusters = [i // 4 for i in range(len(hits))]
    a = cluster_bootstrap_ci(hits, clusters, n_boot=500, seed=1)
    b = cluster_bootstrap_ci(hits, clusters, n_boot=500, seed=1)
    assert a == b
    p = 100 * sum(hits) / len(hits)
    assert a[0] <= p <= a[1]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    print("RESULT", "all passed" if not fails else "%d failed" % fails)
    sys.exit(1 if fails else 0)
