"""기준 A = EXP-034 b (그리고 c) 구성 — EXP-040 run.py의 A 코드를 그대로 옮긴 것 (EXP-041·042가 A의 평면 그림 · val 판정을 쓰려고).
EXP-040 run.py는 기록 후 고치지 않으므로 복사본이다. 계산은 같다 (재측정이 EXP-034 기록과 같음을 EXP-040이 확인)."""
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from src.exp import ROOT, cached_table, feature_table_ext, make_et
from src.experts import KnockRule, Rule83, RuleOverride, ScoreStack, ae_mean, warm_cache, ae_segerr


class A:
    def __init__(self, X, y, names):
        self.y, self.names, self.n = np.asarray(y), list(names), len(y)
        self.nor = self.y == 0
        self.E = cached_table("ext", feature_table_ext, X).values
        IM = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "2-7_충격울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
        key = [os.path.splitext(os.path.basename(s))[0] for s in names]
        frac = lambda lo, hi: IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float)
        self.R24 = np.nan_to_num(frac(2125, 4125), nan=-1.0); self.RBODY = np.nan_to_num(frac(5500, 7750), nan=-1.0)
        self.RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
        me = self
        E_, R24, RBODY = self.E, self.R24, self.RBODY

        class Res83:
            def __call__(s, te): return me.RES[list(te)] >= 0.7
            def __repr__(s): return "Res83(fit≥0.7)"

        class Either:
            def __init__(s, a, b): s.a, s.b = a, b
            def __repr__(s): return f"{s.a} OR {s.b}"
            def __call__(s, te): return s.a(te) | s.b(te)

        class ETX:
            def __init__(s, inp="ext+r2"): s.inp = inp
            def __repr__(s): return f"ET[{s.inp}]"
            def __call__(s, tr, te):
                tr, te = np.array(tr), np.array(te)
                Mx = np.hstack([E_, R24[:, None], RBODY[:, None]])
                m = make_et(seed=0)().fit(Mx[tr], me.y[tr])
                p = np.clip(m.predict_proba(Mx[te])[:, 1], 1 / 600, 1 - 1 / 600)
                return np.log(p / (1 - p))[:, None]

        self.S83, self.RS, self.KN = Rule83(0.75, 150), Res83(), KnockRule(2.0, 0.7)
        self.ETX = ETX
        kw = dict(experts=(ae_mean, ETX()), combine="lr")
        rules = [self.KN, Either(self.S83, self.RS)]
        self.mkA = lambda: RuleOverride(kw, rules)                                # A = EXP-034 b
        self.rules = rules
        self.Either = Either
        warm_cache([ae_segerr], seeds=range(5), nested=True)
        self.FA = np.arange(self.n)[:, None]

    def oof_plane(self):
        """OOF 5-fold seed 0: (AE z, ET z) (n, 2) · 규칙이 켜진 파일 bool"""
        g = np.arange(self.n)
        Z, rule = np.empty((self.n, 2)), np.zeros(self.n, bool)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, self.y, g):
            Z[te] = ScoreStack(experts=(ae_mean, self.ETX()), combine="lr").fit(self.FA[tr], self.y[tr]).zscores(self.FA[te])[:, :2]
            rule[te] = self.KN.fit(tr)(te) | self.S83(te) | self.RS(te)
        return Z, rule
