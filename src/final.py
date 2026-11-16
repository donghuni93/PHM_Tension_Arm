"""최종 파이프라인 (7단계): 학습 파일 전체로 한 번 학습 → 새 파일(val·test) 판정 + 판정 근거

왜 이 파일이 따로 있나
- src/experts.py는 실험용이다. 학습 150개를 모듈 안에 들고 파일 번호로 주고받는다(분할마다 캐시를 나눠 쓰려고).
  그래서 번호가 없는 새 파일(val·test)은 채점할 수 없다.
- 이 파일은 제출용이다. 신호 배열만 받으면 동작한다. 계산식은 experts.py와 같고,
  같다는 것은 단계/7_최종·제출/검증_동등성.py 가 확인한다 (같은 분할에서 점수 차이 0).

흐름 (위에서 아래로 읽으면 파이프라인 순서다)
  1. signal_tables : 신호 → 숫자 표 (라벨을 쓰지 않는다. 파일 하나씩 따로 계산한다)
  2. Core.fit      : 전문가 2개(AE·ET) 학습 → 정상 기준 z → 로지스틱 결합 + 규칙의 기준값
  3. Core.parts    : 파일별 점수와 규칙 판정
  4. FinalModel    : 임계값(판정 기준선)까지 정해서 0/1과 근거를 낸다

최종 구성 = EXP-025 (EXP-030에서 고정): AE + ET → 로지스틱 + 8.3 kHz 모드 규칙 + 유격 반복 규칙.
주기 잠금·단일 타격은 판정에 쓰지 않고 '관찰' 열로만 남긴다 (use_lock·use_cap 기본값 False).
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor

from src.evaluate import best_f2_threshold
from src.features import extract_features_ext, extract_features_hits, extract_features_knock

FS = 25000
N_SAMPLES = 50000
# 규칙 문턱 (모두 학습 라벨을 본 뒤 정한 '사후' 값. 단계/1_기준·기록/결정.md. val·test를 보고 고치지 않는다)
P83_MIN, W83_MIN = 0.75, 150.0      # 8.3 kHz 모드 규칙: 돌출(dec), 반높이 폭(Hz)
R_MIN, RATIO_MIN = 2.0, 0.7         # 유격 반복 규칙: 정상 중심 거리, 2위/1위 타격 높이
LOCK_MARGIN = 0.1                   # 주기 잠금(관찰용): 학습 정상 최대 + 여유
KEYS = ("LP", "E", "K", "P83", "W83", "n20", "S")


# ---------------------------------------------------------------------------
# 1. 신호 → 표
# ---------------------------------------------------------------------------
def signal_tables(X, cols=None):
    """신호 배열 (파일 수, 50000) → 표 묶음 dict. 라벨을 쓰지 않고 파일마다 독립이라 학습·val·test에 똑같이 쓴다

    LP  (n, 8, 1025) AE 입력: 평균 제거 → 200 Hz 고역통과 → 1초 구간 8개 → log10 Welch PSD
    E   (n, 150)     ET 입력: 확장 피처 (src/features.py extract_features_ext)
    K   (n, 3)       유격 반복: K1 두 번째 타격/σ, K2 저주파 RMS/σ, K3 2위/1위
    P83·W83 (n,)     8.20–8.45 kHz 봉우리의 돌출(dec)과 반높이 폭(Hz)
    n20 (n,)         큰 타격(바탕의 20배 이상) 수   — 관찰용
    S   (n,)         주기 잠금 점수                 — 관찰용
    cols: 학습 때의 E 열 이름. 주면 같은 순서로 맞춘다 (없는 열은 0)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_SAMPLES:
        raise ValueError(f"신호 길이는 {N_SAMPLES}점이어야 한다. 받은 모양 {X.shape}")
    Xc = X - X.mean(1, keepdims=True)

    # AE 입력
    H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
    starts = np.linspace(0, X.shape[1] - FS, 8).astype(int)
    seg = np.stack([H[:, s:s + FS] for s in starts], 1)
    _, P = welch(seg, fs=FS, nperseg=2048, axis=-1)
    LP = np.log10(P + 1e-20)

    # ET 입력, 유격 반복, 큰 타격·주기 잠금
    clean = lambda T: T.replace([np.inf, -np.inf], np.nan).fillna(0)
    E = clean(pd.DataFrame([extract_features_ext(x) for x in X]))
    if cols is not None:
        # 피처 함수가 학습 때와 같으면 열 집합이 같다. 일부 파일에서만 생기는 열(F09·F10 봉우리)이 빠진 경우만 0으로 채운다
        extra = set(E.columns) - set(cols)
        if extra:
            raise ValueError(f"학습 때 없던 피처 열이 생겼다 (피처 함수가 바뀌었는가): {sorted(extra)[:5]}")
        E = E.reindex(columns=cols, fill_value=0.0)
    K = clean(pd.DataFrame([extract_features_knock(x) for x in X]))
    Hit = clean(pd.DataFrame([extract_features_hits(x) for x in X])[["HIT_n20", "LOCK_S"]])

    # 8.3 kHz 봉우리: 돌출 = 55 Hz로 고른 log PSD − 1.2 kHz 이동 중앙값 (주변보다 얼마나 솟았나)
    f, P8 = welch(Xc, fs=FS, nperseg=8192)
    L = np.log10(P8)
    df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450)
    ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]
        h = PR[i, k] / 2
        l = r = k
        while l > 0 and PR[i, l - 1] >= h:
            l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h:
            r += 1
        W83[i] = f[r] - f[l]

    T = dict(LP=LP, E=E.values.astype(float), K=K[["K1_second_snr", "K2_load_to_hf", "K3_top2_ratio"]].values,
             P83=P83, W83=W83, n20=Hit["HIT_n20"].values.astype(int), S=Hit["LOCK_S"].values)
    return T, list(E.columns)


def take(T, idx):
    """표 묶음에서 파일 일부만"""
    return {k: T[k][idx] for k in KEYS}


# ---------------------------------------------------------------------------
# 2–3. 전문가 2개 + 결합 + 규칙
# ---------------------------------------------------------------------------
def _fit_ae(LP_normal, seed=0):
    """정상 구간 스펙트럼만으로 오토인코더(1025-128-16-128-1025) 학습. 반환 (평균, 표준편차, 모델)"""
    S = LP_normal.reshape(-1, LP_normal.shape[-1])
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Z = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(Z, Z)
    return mu, sd, m


def _ae_score(ae, LP):
    """AE 점수 = log10(복원 오차 평균). 정상 스펙트럼 모양에서 벗어날수록 크다"""
    mu, sd, m = ae
    Z = (LP - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return np.log10(((R - Z) ** 2).astype(np.float32).mean((1, 2))).astype(float)


def _fit_et(E, y, seed=0):
    return ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt",
                                random_state=seed, n_jobs=1).fit(E, y)


def _et_score(et, E):
    """ET 점수 = logit(나무 300그루의 고장 확률 평균)"""
    p = np.column_stack([t.predict_proba(E)[:, 1] for t in et.estimators_]).astype(np.float32).mean(1)
    p = np.clip(p, 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p)).astype(float)


class Core:
    """전문가 2개(AE·ET) → 정상 기준 z → 로지스틱 결합 → 규칙. 임계값은 여기서 정하지 않는다 (FinalModel)

    src/experts.py의 RuleOverride(ScoreStack((ae_mean, et_logit), 'lr'), [Rule83, KnockRule]) 와 같은 계산
    """

    def __init__(self, use_cap=False, use_lock=False):
        self.use_cap, self.use_lock = use_cap, use_lock

    @staticmethod
    def _experts(T, y):
        return _fit_ae(T["LP"][y == 0]), _fit_et(T["E"], y)

    @staticmethod
    def _raw(experts, T):
        ae, et = experts
        return np.column_stack([_ae_score(ae, T["LP"]), _et_score(et, T["E"])])

    def fit(self, T, y):
        y = np.asarray(y)
        n = len(y)
        # (1) 전문가 점수를 '자기가 학습에 쓰이지 않은' 상태로 얻는다: 4등분해서 3으로 학습, 1을 채점 (OOF)
        #     이렇게 하지 않으면 ET가 학습 파일을 외워 점수가 부풀고, 그 위의 로지스틱이 잘못 배운다
        raw = np.empty((n, 2))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(np.arange(n), y):
            raw[b] = self._raw(self._experts(take(T, a), y[a]), take(T, b))
        # (2) 정상 기준 z: (점수 − 정상 중앙값) / (정상 90%선 − 정상 중앙값). 1 = 정상의 90%선
        nz = raw[y == 0]
        self.med_ = np.median(nz, 0)
        d = np.abs(np.percentile(nz, 90, 0) - self.med_)
        self.sc_ = np.where(d > 1e-9, d, nz.std(0)) + 1e-6
        self.oof_z_ = (raw - self.med_) / self.sc_
        # (3) 결합: z 두 개 → 로지스틱 회귀 (계수 2개 + 절편)
        self.lr_ = LogisticRegression(C=1.0, class_weight="balanced").fit(self.oof_z_, y)
        self.cap_ = float(np.median(self.lr_.decision_function(self.oof_z_)[y == 0]))
        # (4) 전문가를 학습 파일 전체로 다시 학습 (새 파일 채점용)
        self.experts_ = self._experts(T, y)
        # (5) 규칙의 기준값: 학습 정상만으로
        Kn = T["K"][y == 0][:, :2]
        self.kmed_ = np.median(Kn, 0)
        self.kmad_ = 1.4826 * np.median(np.abs(Kn - self.kmed_), 0) + 1e-12
        self.lock_th_ = float(T["S"][y == 0].max() + LOCK_MARGIN)
        return self

    def parts(self, T):
        """파일별 점수와 규칙 판정 표"""
        raw = self._raw(self.experts_, T)
        Z = (raw - self.med_) / self.sc_
        lr = self.lr_.decision_function(Z)
        r = np.sqrt((((T["K"][:, :2] - self.kmed_) / self.kmad_) ** 2).mean(1))
        rule83 = (T["P83"] >= P83_MIN) & (T["W83"] >= W83_MIN)
        knock = (r >= R_MIN) & (T["K"][:, 2] >= RATIO_MIN)
        lock = T["S"] >= self.lock_th_
        score = lr + 100.0 * rule83 + 100.0 * knock
        if self.use_lock:
            score = score + 100.0 * lock
        if self.use_cap:
            score = np.where((T["n20"] == 1) & (score < 50), np.minimum(score, self.cap_), score)
        return pd.DataFrame(dict(ae_z=Z[:, 0], et_z=Z[:, 1], lr=lr, P83=T["P83"], W83=T["W83"], rule83=rule83,
                                 r=r, K3=T["K"][:, 2], knock=knock, n20=T["n20"], S=T["S"], lock_obs=lock, score=score))


# ---------------------------------------------------------------------------
# 4. 최종 모델: 임계값까지
# ---------------------------------------------------------------------------
def _inner_job(T, y, tr, te, kw):
    return te, Core(**kw).fit(take(T, tr), y[tr]).parts(take(T, te))["score"].values


class FinalModel:
    """학습 파일 전체로 학습하고, 새 신호를 0(정상)/1(고장)로 판정한다

    임계값: 학습 파일을 4등분한 OOF 점수에서 F2가 최대인 값(같으면 낮은 쪽). 분할 seed마다 값이 달라지므로
    thr_seeds 개수만큼 구해 중앙값을 쓴다. 가장 낮은 값–가장 높은 값 사이의 파일은 '경계'로 표시한다
    """

    def __init__(self, use_cap=False, use_lock=False, thr_seeds=(0, 1, 2), n_jobs=-1):
        self.use_cap, self.use_lock, self.thr_seeds, self.n_jobs = use_cap, use_lock, tuple(thr_seeds), n_jobs

    def fit(self, X, y, names=None):
        y = np.asarray(y)
        self.names_ = list(names) if names is not None else [f"train_{i:03d}" for i in range(len(y))]
        T, self.cols_ = signal_tables(X)
        kw = dict(use_cap=self.use_cap, use_lock=self.use_lock)
        idx = np.arange(len(y))
        jobs = [(s, tr, te) for s in self.thr_seeds
                for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(idx, y, idx)]
        out = Parallel(n_jobs=self.n_jobs)(delayed(_inner_job)(T, y, tr, te, kw) for _, tr, te in jobs)
        self.oof_ = np.empty((len(self.thr_seeds), len(y)))
        for (s, _, _), (te, sc) in zip(jobs, out):
            self.oof_[self.thr_seeds.index(s), te] = sc
        self.thr_all_ = np.array([best_f2_threshold(y, o) for o in self.oof_])
        self.thr_ = float(np.median(self.thr_all_))
        self.core_ = Core(**kw).fit(T, y)
        self.y_ = y
        return self

    def predict_table(self, X, names):
        """새 신호 → 파일별 판정표 (pred = 0/1, 근거 열 포함)"""
        T, _ = signal_tables(X, self.cols_)
        D = self.core_.parts(T)
        D.insert(0, "file", list(names))
        D["threshold"] = self.thr_
        D["pred"] = (D["score"] >= self.thr_).astype(int)
        lo, hi = self.thr_all_.min(), self.thr_all_.max()
        D["borderline"] = (D["score"] >= lo) & (D["score"] < hi) if hi > lo else False
        D["single_hit_obs"] = D["n20"] == 1
        # 규칙 문턱까지의 거리: 두 조건 중 덜 채운 쪽의 비율 (1 이상 = 규칙 켜짐). 0.8–1.0이면 문턱 20% 이내 = 확인 대상
        D["rule83_ratio"] = np.minimum(D["P83"] / P83_MIN, D["W83"] / W83_MIN)
        D["knock_ratio"] = np.minimum(D["r"] / R_MIN, D["K3"] / RATIO_MIN)
        D["rule_near"] = (np.maximum(D["rule83_ratio"], D["knock_ratio"]) >= 0.8) & ~D["rule83"] & ~D["knock"]
        D["reason"] = [self._reason(r) for r in D.itertuples()]
        return D

    def _reason(self, r):
        why = []
        if r.rule83:
            why.append(f"8.3 kHz 모드 규칙 (돌출 {r.P83:.2f} dec, 폭 {r.W83:.0f} Hz)")
        if r.knock:
            why.append(f"유격 반복 규칙 (정상 중심 거리 {r.r:.1f}, 2위/1위 {r.K3:.2f})")
        if not why:
            side = "≥" if r.pred else "<"
            why.append(f"결합 점수 {r.lr:+.2f} {side} 임계값 {self.thr_:.2f} (스펙트럼 z {r.ae_z:+.1f}, 충격 z {r.et_z:+.1f})")
        if r.borderline:
            why.append("경계: 임계값 seed에 따라 판정이 바뀜")
        if r.rule_near:
            why.append(f"[확인] 규칙 문턱 20% 이내 (8.3 kHz {r.rule83_ratio:.2f}, 유격 {r.knock_ratio:.2f}; 1 이상이면 켜짐)")
        if r.lock_obs:
            why.append(f"[관찰] 주기 잠금 점수 {r.S:.2f} ≥ {self.core_.lock_th_:.2f}")
        if r.single_hit_obs:
            why.append("[관찰] 큰 타격 정확히 1개")
        return " · ".join(why)


# ---------------------------------------------------------------------------
# 제출 파일
# ---------------------------------------------------------------------------
def _name_key(v):
    """파일 이름 표기 통일: 앞뒤 공백 · 폴더 경로 · 확장자 · 대소문자 차이를 없앤다 ('val/VAL_001.CSV' → 'val_001')"""
    t = str(v).strip().replace(chr(92), "/").rsplit("/", 1)[-1].lower()
    return t[:-4] if t.endswith(".csv") else t


def _trailing_int(k):
    """이름 끝의 숫자 ('val_001' → 1, '001' → 1). 없으면 None"""
    d = ""
    for ch in reversed(k):
        if not ch.isdigit():
            break
        d = ch + d
    return int(d) if d else None


LABEL_NAMES = ("label", "labels", "target", "pred", "prediction", "class", "y", "라벨", "예측", "정답")


def write_submission(D, out_path, template=None, name_col="file", label_col="label"):
    """판정표 D(file·pred 열) → 제출 CSV

    template: 과제에서 준 제출 양식 CSV 경로. 주면 그 파일의 열 이름·행 순서·파일 이름 표기를 그대로 두고 라벨 열만 채운다.
      - 이름 열 = 값이 우리 파일 이름과 가장 많이 겹치는 열 (확장자·대소문자·경로 차이는 무시, 번호만 있는 양식은 번호로 맞춘다)
      - 라벨 열 = 나머지가 한 열이면 그 열, 여러 열이면 이름이 label·target 등인 열 또는 label_col로 지정한 열
      - 양식의 이름이 예측에 하나라도 없으면 멈춘다 (틀린 파일을 조용히 만들지 않는다)
    template이 없으면 name_col·label_col 두 열로 쓴다 — 11/19 양식 공개 전의 임시 형식
    """
    pred = {_name_key(f): int(v) for f, v in zip(D["file"], D["pred"])}
    if len(pred) != len(D):
        raise ValueError("판정표의 파일 이름이 겹친다")
    if template is None:
        sub = pd.DataFrame({name_col: list(D["file"].astype(str)), label_col: list(D["pred"].astype(int))})
        sub.to_csv(out_path, index=False, encoding="utf-8")
        return sub

    sub, used = None, None
    for enc in ("utf-8-sig", "cp949"):
        try:
            sub = pd.read_csv(template, dtype=str, keep_default_na=False, encoding=enc)   # 문자열로 읽어 '001' 같은 표기를 지킨다
            used = enc
            break
        except UnicodeDecodeError:
            continue
    if sub is None:
        raise ValueError(f"제출 양식을 읽을 수 없다 (utf-8 · cp949 모두 실패): {template}")

    hits = {c: sub[c].map(_name_key).isin(pred).mean() for c in sub.columns}
    ncol = max(hits, key=hits.get)
    keys = sub[ncol].map(_name_key)
    if hits[ncol] < 1.0:
        # 번호만 적힌 양식 ('001', '1'): 이름 끝 번호가 양쪽에서 겹치지 않고 모두 대응될 때만 번호로 맞춘다
        by_num = {_trailing_int(k): v for k, v in pred.items()}
        nums = [{c: sub[c].map(_name_key).map(_trailing_int)} for c in sub.columns]
        cand = [(c, n) for d in nums for c, n in d.items() if n.notna().all() and n.is_unique and set(n) <= set(by_num)]
        if None in by_num or len(by_num) != len(pred) or not cand:
            missing = sorted(set(keys) - set(pred))[:5]
            raise ValueError(f"양식의 파일 이름 중 예측에 없는 것이 있다 (열 {ncol}): {missing}")
        ncol, n = cand[0]
        values = n.map(by_num)
    else:
        if not keys.is_unique:
            raise ValueError(f"양식의 파일 이름이 겹친다 (열 {ncol})")
        values = keys.map(pred)
    others = [c for c in sub.columns if c != ncol]
    if len(others) == 1:
        lcol = others[0]
    else:
        named = [c for c in others if c.strip().lower() in LABEL_NAMES or c == label_col]
        if len(named) != 1:
            raise ValueError(f"라벨 열을 정할 수 없다. 양식의 열: {list(sub.columns)} → write_submission(label_col=...)로 지정한다")
        lcol = named[0]
    sub[lcol] = values.astype(int).values
    sub.to_csv(out_path, index=False, encoding="utf-8-sig" if used == "utf-8-sig" and _has_bom(template) else ("cp949" if used == "cp949" else "utf-8"))
    return sub


def _has_bom(path):
    with open(path, "rb") as f:
        return f.read(3) == bytes([0xEF, 0xBB, 0xBF])
