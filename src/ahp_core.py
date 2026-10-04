# -*- coding: utf-8 -*-
"""
Model 2 -- Analytic Hierarchy Process engine (Saaty 1980, 1986, 1994), as used by Scala (2008).

  * pairwise comparison matrix A (reciprocal: a_ji = 1 / a_ij, a_ii = 1) on Saaty's 1-9 scale
  * priority vector w = principal right eigenvector of A, normalised to sum 1
  * lambda_max, CI = (lambda_max - n) / (n - 1), CR = CI / RI(n); Saaty accepts CR <= 0.10
  * synthesis: global priority of alternative a = sum_c w_c * p(a | c)
      - "distributive" mode (Saaty's original, the one Scala uses implicitly): p(.|c) sums to 1
      - "ideal" mode: p(.|c) divided by its max -- avoids rank reversal when alternatives are added

The geometric-mean (row) method is included for comparison; it equals the eigenvector for
consistent matrices and is the maximum-likelihood estimate under multiplicative log-normal errors.
"""
import numpy as np

# Saaty's random consistency index (average CI of random reciprocal matrices), n = 1..15.
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
                10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}

SAATY_LABELS = {1: "equal", 3: "moderate", 5: "strong", 7: "very strong", 9: "extreme"}


def from_upper(n, judgments):
    """Build a reciprocal matrix from {(i, j): a_ij} for i < j."""
    a = np.ones((n, n))
    for (i, j), v in judgments.items():
        if i == j:
            continue
        a[i, j] = v
        a[j, i] = 1.0 / v
    return a


def check_reciprocal(a, tol=1e-9):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("pairwise matrix must be square")
    if np.any(a <= 0):
        raise ValueError("pairwise matrix entries must be positive")
    if not np.allclose(np.diag(a), 1.0, atol=tol):
        raise ValueError("diagonal must be 1")
    if not np.allclose(a * a.T, 1.0, atol=1e-6):
        raise ValueError("matrix is not reciprocal (a_ij * a_ji != 1)")
    return a


def eigen_priority(a):
    """Principal eigenvector priorities and lambda_max."""
    a = check_reciprocal(a)
    n = a.shape[0]
    if n == 1:
        return np.array([1.0]), 1.0
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    return w, float(vals[k].real)


def geometric_mean_priority(a):
    a = check_reciprocal(a)
    g = np.exp(np.log(a).mean(axis=1))
    return g / g.sum()


def consistency(a):
    """Return dict(w, lambda_max, CI, RI, CR). CR is 0 for n <= 2 (always consistent)."""
    a = check_reciprocal(a)
    n = a.shape[0]
    w, lam = eigen_priority(a)
    if n <= 2:
        return {"w": w, "lambda_max": lam, "CI": 0.0, "RI": 0.0, "CR": 0.0}
    ci = (lam - n) / (n - 1)
    ri = RANDOM_INDEX.get(n, 1.59)
    return {"w": w, "lambda_max": lam, "CI": ci, "RI": ri, "CR": ci / ri}


def synthesize(criteria_weights, local_priorities, mode="distributive"):
    """criteria_weights: (C,) ; local_priorities: (C, A) each row = priorities of alternatives
    under that criterion (any positive scale). Returns global priorities (A,) summing to 1."""
    w = np.asarray(criteria_weights, dtype=float)
    p = np.asarray(local_priorities, dtype=float)
    if mode == "distributive":
        p = p / p.sum(axis=1, keepdims=True)
    elif mode == "ideal":
        p = p / p.max(axis=1, keepdims=True)
    else:
        raise ValueError(mode)
    g = w @ p
    return g / g.sum()


class AHP:
    """A two-level hierarchy (goal -> criteria -> alternatives), which is exactly the structure
    Scala's model reduces to once sub-criteria are aggregated by the Pugh chart."""

    def __init__(self, criteria, alternatives):
        self.criteria = list(criteria)
        self.alternatives = list(alternatives)
        self.criteria_matrix = None
        self.alt_matrices = {}

    def set_criteria_matrix(self, a):
        a = check_reciprocal(a)
        if a.shape[0] != len(self.criteria):
            raise ValueError("criteria matrix size mismatch")
        self.criteria_matrix = a

    def set_alternative_matrix(self, criterion, a):
        a = check_reciprocal(a)
        if a.shape[0] != len(self.alternatives):
            raise ValueError("alternative matrix size mismatch")
        self.alt_matrices[criterion] = a

    def solve(self, mode="distributive"):
        crit = consistency(self.criteria_matrix)
        local, crs = [], {}
        for c in self.criteria:
            r = consistency(self.alt_matrices[c])
            local.append(r["w"])
            crs[c] = r["CR"]
        local = np.vstack(local)
        g = synthesize(crit["w"], local, mode=mode)
        order = np.argsort(-g)
        return {
            "criteria_weights": dict(zip(self.criteria, crit["w"])),
            "criteria_CR": crit["CR"],
            "local": {c: dict(zip(self.alternatives, local[i])) for i, c in enumerate(self.criteria)},
            "alternative_CR": crs,
            "global": dict(zip(self.alternatives, g)),
            "ranking": [self.alternatives[i] for i in order],
            "best": self.alternatives[order[0]],
            "n_comparisons": self.n_comparisons(),
        }

    def n_comparisons(self):
        c, a = len(self.criteria), len(self.alternatives)
        return c * (c - 1) // 2 + c * a * (a - 1) // 2
