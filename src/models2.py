"""2차 설계 2·3단계 추정기 (EXP-041 · EXP-042). 입력 행 = [FIXED H 피처 | 박자 S | 1 kHz 위 log PSD] (EXP-040의 M과 같음)

Two(comb, experts)
  H 피처(폴드 밖 H + 폴드 안 H16–H17) → ET, 박자 S → 학습 정상 강건 z
  comb 'lin' : [ET 안쪽 4겹 OOF logit, 박자 z] → 로지스틱 (EXP-040 C와 같은 계산)
       'svm' : 같은 입력을 표준화 → RBF SVM (C = 1, gamma = 'scale', class_weight = 'balanced', 고정)
       'max' : 고장 = (ET logit ≥ t_H) OR (박자 z ≥ q99). t_H = fit 행의 ET 안쪽 OOF logit에서 F2 최대 문턱, q99 = fit 행 정상 박자 z의 99 % 분위(고정)
               nested_cv에 맞추려고 점수를 이진(규칙이 켜지면 1, 아니면 0)으로 돌려준다 → 후보 문턱이 {0, 1}뿐이라 바깥에서 문턱이 다시 골라지지 않는다
  experts (3단계, OR로 붙임, 문턱 = fit 행 정상의 99 % 분위, 라벨로 고르지 않음)
       'E1' H16(가장 솟은 좁은 봉우리 높이) > 정상 H16의 99 %        'E2' H18(HFL) > 정상 H18의 99 % AND H1 충격 수 ≥ 1
       전문가가 있으면 점수는 이진: (본 결합의 판정) OR E1 OR E2. 본 결합의 판정 문턱 = fit 행의 안쪽 OOF 점수에서 F2 최대 (lin·svm), max는 위 규칙 그대로
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluate import best_f2_threshold
from src.exp import make_et
from src.features2 import PeakRef, H_FOLD, robust_z


def lg(p):
    p = np.clip(p, 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))


class Two(BaseEstimator):
    def __init__(self, comb="lin", experts=(), fixed=(), nf=0, df=1.0, peak=(), q_exp=99.0, seed=0, scale_thr=1.0):
        self.comb, self.experts, self.fixed, self.nf, self.df, self.peak = comb, tuple(experts), tuple(fixed), nf, df, tuple(peak)
        self.q_exp, self.seed, self.scale_thr = q_exp, seed, scale_thr      # scale_thr: 전문가 문턱을 (분위수 − 정상 중앙값)의 몇 배로 (C3 ±20 % 점검용, 기본 1)

    # ---- 피처
    def _parts(self, F):
        F = np.asarray(F, float)
        H = F[:, :self.nf]
        pk = self.pr_.transform(F[:, self.nf + 1:])
        cols = [H] + ([pk[:, [H_FOLD.index(p) for p in self.peak]]] if self.peak else [])
        return np.hstack(cols), pk, F[:, self.nf]

    def _thr(self, v_normal):
        q = np.percentile(v_normal, self.q_exp); med = np.median(v_normal)
        return med + (q - med) * self.scale_thr

    def fit(self, F, y):
        F, y = np.asarray(F, float), np.asarray(y)
        self.pr_ = PeakRef(self.df).fit(F[y == 0, self.nf + 1:])
        H, pk, S = self._parts(F)
        self.zf_ = robust_z(S[y == 0]); z = self.zf_(S)
        oof = np.empty(len(y))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(H, y):
            oof[b] = lg(make_et(seed=self.seed)().fit(H[a], y[a]).predict_proba(H[b])[:, 1])
        self.m_ = make_et(seed=self.seed)().fit(H, y)
        Z = np.column_stack([oof, z])
        if self.comb == "lin":
            self.c_ = LogisticRegression(C=1.0, class_weight="balanced").fit(Z, y)
        elif self.comb == "svm":
            self.sc_ = StandardScaler().fit(Z)
            self.c_ = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(self.sc_.transform(Z), y)
        else:                                                              # max
            self.tH_ = float(best_f2_threshold(y, oof))
            self.q_ = float(np.percentile(z[y == 0], 99))
            self.sa_, self.sb_ = float(np.std(oof) + 1e-9), float(np.std(z[y == 0]) + 1e-9)
        if self.comb != "max":
            self.tc_ = float(best_f2_threshold(y, self._comb_score(oof, z)))   # 전문가가 OR로 붙을 때 쓰는 결합 문턱 (fit 행 안쪽 OOF 점수에서 F2 최대)
        i1 = self.fixed.index("H1_nimp") if "H1_nimp" in self.fixed else None
        self.i1_ = i1
        self.t16_ = self._thr(pk[y == 0, 0])
        self.t18_ = self._thr(H[y == 0, self.fixed.index("H18_hfl")])
        return self

    # ---- 점수
    def _comb_score(self, a, z):
        """연속 결합 점수 (클수록 고장). max는 두 여유의 최댓값 (0 이상 = 켜짐)"""
        if self.comb == "lin":
            return self.c_.decision_function(np.column_stack([a, z]))
        if self.comb == "svm":
            return self.c_.decision_function(self.sc_.transform(np.column_stack([a, z])))
        return np.maximum((a - self.tH_) / self.sa_, (z - self.q_) / self.sb_)

    def parts(self, F):
        """(ET logit, 박자 z)"""
        H, pk, S = self._parts(F)
        return lg(self.m_.predict_proba(H)[:, 1]), self.zf_(S)

    def fires(self, F):
        """블록별 켜짐 dict: main(결합 판정) · E1 · E2 (bool 배열)"""
        H, pk, S = self._parts(F)
        a, z = lg(self.m_.predict_proba(H)[:, 1]), self.zf_(S)
        s = self._comb_score(a, z)
        main = (s >= 0) if self.comb == "max" else (s >= self.tc_)
        out = dict(main=main, a=a, z=z, score=s)
        if "E1" in self.experts:
            out["E1"] = pk[:, 0] > self.t16_
        if "E2" in self.experts:
            out["E2"] = (H[:, self.fixed.index("H18_hfl")] > self.t18_) & (H[:, self.i1_] >= np.log(2) - 1e-9)
        return out

    def decision_function(self, F):
        f = self.fires(F)
        if self.comb == "max" or self.experts:
            on = f["main"].copy()
            for e in ("E1", "E2"):
                if e in f:
                    on |= f[e]
            return on.astype(float)
        return f["score"]


class TwoAE(Two):
    """EXP-044: Two(comb='lin')에 스펙트럼 AE 점수를 세 번째 입력으로 추가. 입력 행 = [Two의 행 | 파일 번호(마지막 열)].
    결합 입력 = [ET 안쪽 4겹 OOF logit, 박자 z, AE z]. AE = src.experts.ae_mean (정상만 학습, 안쪽 OOF로 점수 → 학습 정상의 강건 z). 판정 문턱·전문가는 Two와 같음"""

    def fit(self, F, y):
        from src.experts import ae_mean
        F, y = np.asarray(F, float), np.asarray(y)
        idx = F[:, -1].astype(int); F = F[:, :-1]; self.fit_idx_ = idx
        self.pr_ = PeakRef(self.df).fit(F[y == 0, self.nf + 1:])
        H, pk, S = self._parts(F)
        self.zf_ = robust_z(S[y == 0]); z = self.zf_(S)
        oof, ae = np.empty(len(y)), np.empty(len(y))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(H, y):
            oof[b] = lg(make_et(seed=self.seed)().fit(H[a], y[a]).predict_proba(H[b])[:, 1])
            ae[b] = ae_mean(idx[a], idx[b])[:, 0]
        self.m_ = make_et(seed=self.seed)().fit(H, y)
        self.aef_ = robust_z(ae[y == 0]); aez = self.aef_(ae)
        Z = np.column_stack([oof, z, aez]); self.Z_, self.y_ = Z, y        # 안쪽 OOF 결합 입력 (다른 결합 모델을 같은 입력으로 비교할 때 쓴다, EXP-045)
        self.c_ = LogisticRegression(C=1.0, class_weight="balanced").fit(Z, y)
        self.tc_ = float(best_f2_threshold(y, self.c_.decision_function(Z)))
        self.i1_ = self.fixed.index("H1_nimp") if "H1_nimp" in self.fixed else None
        self.t16_ = self._thr(pk[y == 0, 0]); self.t18_ = self._thr(H[y == 0, self.fixed.index("H18_hfl")])
        return self

    def _comb_score(self, a, z, aez=None):
        return self.c_.decision_function(np.column_stack([a, z, np.zeros_like(a) if aez is None else aez]))

    def fires(self, F):
        from src.experts import ae_mean
        F = np.asarray(F, float); te = F[:, -1].astype(int); F = F[:, :-1]
        H, pk, S = self._parts(F)
        a, z = lg(self.m_.predict_proba(H)[:, 1]), self.zf_(S)
        raw = getattr(self, "ae_override_", None)          # val · test: 학습 파일 번호로는 못 구하므로 밖에서 신호로 계산한 AE 점수를 넣는다
        aez = self.aef_(raw if raw is not None else ae_mean(self.fit_idx_, te)[:, 0])
        s = self._comb_score(a, z, aez)
        out = dict(main=(s >= self.tc_), a=a, z=z, ae=aez, score=s)
        if "E1" in self.experts: out["E1"] = pk[:, 0] > self.t16_
        if "E2" in self.experts: out["E2"] = (H[:, self.fixed.index("H18_hfl")] > self.t18_) & (H[:, self.i1_] >= np.log(2) - 1e-9)
        return out

    def decision_function(self, F):
        f = self.fires(F)
        if not self.experts: return f["score"]
        on = f["main"].copy()
        for e in ("E1", "E2"):
            if e in f: on |= f[e]
        return on.astype(float)
