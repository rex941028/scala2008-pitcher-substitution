# -*- coding: utf-8 -*-
"""
Model 4 -- Pugh chart rating of the alternatives (Scala 2008, p. 5).

  "Each alternative was ranked for each of the relevant characteristics defined on the influence
   diagram. A +1 is assigned if the pitcher is above average, -1 if below average, and 0 if average
   or unable to be determined. The totals for each alternative are then used in the rankings.
   The absolute value of difference between alternatives with respect to an attribute determines
   the ranking for the AHP. A difference of 1 is 'moderate', 2 is 'strong', 3 is 'very strong',
   4 is 'very strong', and 5 is 'extremely'."

So for every criterion (cluster) the alternatives get a Pugh total, and each pair (a, b) gets the
Saaty judgment s(|T_a - T_b|) in favour of the higher total.  The mapping is implemented exactly as
published (3 and 4 both "very strong" = 7; >= 5 -> 9; 0 -> 1 "equal").
"""
import numpy as np

from ahp_core import from_upper

DIFF_TO_SAATY = {0: 1.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 7.0}


def diff_to_saaty(d):
    d = int(round(abs(d)))
    return DIFF_TO_SAATY.get(d, 9.0)


def rate(value, reference, tol, higher_is_better=True):
    """+1 above average, -1 below, 0 within +-tol (or value unknown)."""
    if value is None or reference is None or np.isnan(value) or np.isnan(reference):
        return 0
    diff = value - reference if higher_is_better else reference - value
    if diff > tol:
        return 1
    if diff < -tol:
        return -1
    return 0


class PughChart:
    """ratings[criterion][alternative][subcriterion] = -1 / 0 / +1"""

    def __init__(self, criteria, alternatives):
        self.criteria = list(criteria)
        self.alternatives = list(alternatives)
        self.ratings = {c: {a: {} for a in self.alternatives} for c in self.criteria}

    def set(self, criterion, alternative, sub, value):
        if value not in (-1, 0, 1):
            raise ValueError("Pugh ratings are -1, 0 or +1")
        self.ratings[criterion][alternative][sub] = value

    def totals(self, criterion):
        return np.array([sum(self.ratings[criterion][a].values()) for a in self.alternatives], dtype=float)

    def pairwise(self, criterion):
        t = self.totals(criterion)
        n = len(t)
        judg = {}
        for i in range(n):
            for j in range(i + 1, n):
                s = diff_to_saaty(t[i] - t[j])
                judg[(i, j)] = s if t[i] >= t[j] else 1.0 / s
        return from_upper(n, judg)

    def table(self, criterion):
        """Rows = sub-criteria, columns = alternatives (for printing / CSV)."""
        subs = sorted({s for a in self.alternatives for s in self.ratings[criterion][a]})
        rows = []
        for s in subs:
            rows.append([s] + [self.ratings[criterion][a].get(s, 0) for a in self.alternatives])
        rows.append(["Total"] + list(self.totals(criterion).astype(int)))
        return rows
