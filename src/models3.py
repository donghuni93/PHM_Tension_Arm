"""EXP-046 모델 (사전등록_046). TwoAE(lin) 위에 (a) ET 입력 열 추가 (hf_rep: lo를 fit 안에서 단변량 AUC로, n1_reb: 그대로)
(b) u_ring 을 로지스틱 4번째 입력으로 (키·t 를 fit 안에서 pick_ring_m, m = 0.2). E2 OR 유지, 8.3k 규칙 OR는 or83 로 켜고 끈다.

입력 행 = [M (nM열) | hf 6 | n1 1 | ring 12 | 파일 번호 | r83]  (features3.COLS 순서, 항상 전부 넣고 쓰는 것만 이 클래스가 고른다)
G0 (아무 옵션 없이, or83=True) = EXP-044의 J 와 같은 계산.
fit 안 선택 기록: 환경변수 EXP046_LOG 경로가 있으면 학습 행이 110개 이상인 fit(= 바깥 학습 전체)의 선택을 한 줄 JSON으로 덧붙인다.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.evaluate import best_f2_threshold, ET_N
from src.exp import make_et
from src.features2 import PeakRef, robust_z
from src.features3 import HF_GRID, I_HF, I_N1, I_RING, N_EX, RING_KEYS
from src.models2 import TwoAE, lg


def pick_ring_m(v, y, m):
    """R_step4c_margin.py 22–33행과 같은 선택. v = (n, 12) fit_max (RING_KEYS 순서), y = 학습 라벨. → (키 번호, t)
    켜진 학습 고장 수 최대 → 대역이 좁은 쪽 → ζmax 작은 쪽. 정상 최댓값 mx 에 여유 m: t = max(0.5, mx(1+m)) (mx = 0 이면 0.5(1+m))"""
    nor = y == 0
    best = None
    for j, key in enumerate(RING_KEYS):
        mx = v[nor, j].max()
        t = max(0.5, mx * (1 + m)) if mx > 0 else 0.5 * (1 + m)
        on = v[:, j] >= t
        sc = (int(on[~nor].sum()), -(key[1] - key[0]), -key[2])
        if best is None or sc > best[0]:
            best = (sc, j, t)
    return best[1], float(best[2])


class Models3(TwoAE):
    def __init__(self, hf=False, n1=False, ring=False, or83=True, nM=0, m=0.2, tag="", experts=("E2",), fixed=(), nf=0, df=1.0, peak=(), seed=0):
        super().__init__("lin", experts, fixed=fixed, nf=nf, df=df, peak=peak, seed=seed)
        self.hf, self.n1, self.ring, self.or83, self.nM, self.m, self.tag = hf, n1, ring, or83, nM, m, tag

    def _split(self, F):
        F = np.asarray(F, float)
        return F[:, :self.nM], F[:, self.nM:self.nM + N_EX], F[:, -2].astype(int), F[:, -1]

    def _et_in(self, H, EX):
        cols = [H]
        if self.hf: cols.append(EX[:, [I_HF + self.hf_j_]])
        if self.n1: cols.append(EX[:, [I_N1]])
        return np.hstack(cols)

    def _u(self, EX):
        return np.clip(EX[:, I_RING + self.rk_] / self.rt_, 0, 3)

    def _score(self, a, z, aez, u=None):
        Z = [a, z, np.zeros_like(a) if aez is None else aez]
        if self.ring: Z.append(np.zeros_like(a) if u is None else u)
        return self.c_.decision_function(np.column_stack(Z))

    def fit(self, F, y):
        from src.experts import ae_mean
        y = np.asarray(y)
        M, EX, idx, _ = self._split(F); self.fit_idx_ = idx
        self.pr_ = PeakRef(self.df).fit(M[y == 0, self.nf + 1:])
        H, pk, S = self._parts(M)
        self.zf_ = robust_z(S[y == 0]); z = self.zf_(S)
        rec = dict(tag=self.tag, n=int(len(y)))
        if self.hf:
            au = [roc_auc_score(y, EX[:, I_HF + j]) for j in range(len(HF_GRID))]
            self.hf_j_ = max(range(len(au)), key=lambda j: (round(au[j], 12), HF_GRID[j]))      # 동률 → 큰 lo
            rec["lo"] = HF_GRID[self.hf_j_]; rec["auc"] = [round(a, 4) for a in au]
        Het = self._et_in(H, EX)
        oof, ae = np.empty(len(y)), np.empty(len(y))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(Het, y):
            oof[b] = lg(make_et(n=ET_N, seed=self.seed)().fit(Het[a], y[a]).predict_proba(Het[b])[:, 1])
            ae[b] = ae_mean(idx[a], idx[b])[:, 0]
        self.m_ = make_et(n=ET_N, seed=self.seed)().fit(Het, y)
        self.aef_ = robust_z(ae[y == 0]); aez = self.aef_(ae)
        Z = [oof, z, aez]
        if self.ring:
            self.rk_, self.rt_ = pick_ring_m(EX[:, I_RING:I_RING + len(RING_KEYS)], y, self.m)
            Z.append(self._u(EX)); rec["key"] = list(RING_KEYS[self.rk_]); rec["t"] = round(self.rt_, 4)
        Z = np.column_stack(Z); self.Z_, self.y_ = Z, y
        self.c_ = LogisticRegression(C=1.0, class_weight="balanced").fit(Z, y)
        if self.ring: rec["coef_u"] = float(self.c_.coef_[0][3])
        self.tc_ = float(best_f2_threshold(y, self.c_.decision_function(Z)))
        self.i1_ = self.fixed.index("H1_nimp") if "H1_nimp" in self.fixed else None
        self.t16_ = self._thr(pk[y == 0, 0]); self.t18_ = self._thr(H[y == 0, self.fixed.index("H18_hfl")])
        p = os.environ.get("EXP046_LOG")
        if p and len(y) >= 110:
            with open(f"{p}.{os.getpid()}", "a", encoding="utf-8") as fh: fh.write(json.dumps(rec) + "\n")
        return self

    def fires(self, F):
        from src.experts import ae_mean
        M, EX, te, r83 = self._split(F)
        H, pk, S = self._parts(M)
        a, z = lg(self.m_.predict_proba(self._et_in(H, EX))[:, 1]), self.zf_(S)
        raw = getattr(self, "ae_override_", None)
        aez = self.aef_(raw if raw is not None else ae_mean(self.fit_idx_, te)[:, 0])
        u = self._u(EX) if self.ring else None
        s = self._score(a, z, aez, u)
        out = dict(main=(s >= self.tc_), a=a, z=z, ae=aez, score=s, R83=(r83 > 0.5) if self.or83 else np.zeros(len(a), bool))
        if u is not None: out["u"] = u
        if "E2" in self.experts: out["E2"] = (H[:, self.fixed.index("H18_hfl")] > self.t18_) & (H[:, self.i1_] >= np.log(2) - 1e-9)
        return out

    def decision_function(self, F):
        f = self.fires(F)
        on = f["main"] | f["R83"]
        if "E2" in f: on |= f["E2"]
        return on.astype(float)
