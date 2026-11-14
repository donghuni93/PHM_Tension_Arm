# %% [markdown]
# EXP-013 스펙트럼 AE 정규화 — 가중치 L2(ridge)·조기 종료 vs AE 끝단 ridge 분류기
# - 출발점: EXP-012 (신효섭 보고서 스펙트럼 AE, alpha=0, 60 epoch). F2 0.832, 오경보율 0.068
#   문제: AE가 보지 않은 정상의 복원 오차가 학습 정상의 약 2배 → 정상에 과적합, 오경보는 이 간격에서 생김
# - A. AE 자체 정규화 (점수 = 복원 오차 평균, 정상만 학습)
#   · 가중치 L2 (MLPRegressor alpha = ridge 벌점) 0.01 · 0.1 · 1
#   · 조기 종료 (학습 정상 10%를 검증으로, 10 epoch 개선 없으면 정지, 최대 200 epoch)
# - B. AE 끝단 ridge (보고서 8장 "대역별 AE 오차 + 강한 L2 LR"을 우리 방식으로)
#   · 폴드 안 학습 정상을 4등분 → AE 4개 (각각 정상 3/4로 학습)
#   · 학습 정상의 오차 = 그 파일을 보지 않은 AE 1개의 오차 (OOF) / 학습 고장·테스트 파일 = AE 4개 오차 평균
#   · 오차 벡터 → log → 표준화 → L2 로지스틱 (C는 학습 파일 안 4겹 CV, 로그손실로 선택, class_weight balanced)
#   · 입력 2가지: 옥타브 대역 6개 오차(200–400, …, 6400–12500 Hz, 라벨 보기 전에 고정) / bin 1025개 오차 전체
# - 대표(main) = B의 6대역 (사용자 요청 "끝단 ridge", 미리 지정). 나머지는 비교용이며 최고점을 대표로 바꾸지 않는다
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-013_AE정규화_ridge/run.py

# %% 설정
import os, sys, warnings, hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"
BANDS = [200, 400, 800, 1600, 3200, 6400, 12501]
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ex = Experiment(__file__, "EXP-013", "스펙트럼 AE 정규화 — 가중치 L2·조기 종료 vs 끝단 ridge", [],
                member="이동훈", model="스펙트럼 AE + 끝단 L2 로지스틱",
                changes="EXP-012 AE의 대역별 복원 오차(6개, 폴드 안 OOF) → L2 로지스틱", focus=())
X, y, names = load_train()
g = np.arange(len(y))

# EXP-012와 같은 입력 (200 Hz 고역통과, 1초×8구간 log Welch PSD). EXP-012 run.py가 만든 캐시
z = np.load(os.path.join(CACHE, "seg_logpsd_hp200_8x1s.npz")); f_psd, LP = z["f"], z["LP"]
BAND_IDX = [np.flatnonzero((f_psd >= lo) & (f_psd < hi)) for lo, hi in zip(BANDS[:-1], BANDS[1:])]
ERR_CACHE = os.path.join(CACHE, "exp013_ae_oof"); os.makedirs(ERR_CACHE, exist_ok=True)


# %% 모델
def fit_ae(S, alpha=0.0, early=False, seed=0):
    """S: (구간 수, 1025) 정상 구간. 표준화 통계와 학습된 AE를 돌려준다"""
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Z = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam",
                     learning_rate_init=1e-3, alpha=alpha, random_state=seed,
                     max_iter=200 if early else 60, early_stopping=early,
                     validation_fraction=0.1, n_iter_no_change=10).fit(Z, Z)
    return mu, sd, m


def ae_err(ae, idx):
    """파일 idx의 bin별 복원 제곱오차 (구간 평균) → (파일, 1025)"""
    mu, sd, m = ae
    Z = (LP[idx] - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return ((R - Z) ** 2).mean(1)


class SpecAE(BaseEstimator):
    """A. 정상만 학습한 AE 하나. 점수 = 복원 오차 평균. 입력 F = 파일 번호 1열"""
    def __init__(self, alpha=0.0, early=False):
        self.alpha, self.early = alpha, early

    def fit(self, F, y):
        idx = F[:, 0].astype(int)[np.asarray(y) == 0]
        self.ae_ = fit_ae(LP[idx].reshape(-1, LP.shape[-1]), self.alpha, self.early)
        return self

    def decision_function(self, F):
        return ae_err(self.ae_, F[:, 0].astype(int)).mean(1)


def oof_errors_exact(train_idx, ytr, k=4):
    """학습 파일 train_idx로 AE k개 학습(각각 학습 정상 3/4) → 150개 파일 전부의 bin별 오차.
    학습 정상 = 자기를 보지 않은 AE 1개의 오차, 나머지 = k개 평균. 학습 파일 집합별로 디스크 캐시"""
    key = hashlib.md5(np.sort(train_idx).astype(np.int64).tobytes()).hexdigest()
    p = os.path.join(ERR_CACHE, f"{key}.npy")
    if os.path.exists(p):
        return np.load(p).astype(float)
    nor = train_idx[ytr == 0]
    folds = list(KFold(k, shuffle=True, random_state=0).split(nor))
    Es = []
    for a, b in folds:
        Es.append(ae_err(fit_ae(LP[nor[a]].reshape(-1, LP.shape[-1])), g))
    E = np.mean(Es, 0)
    for (a, b), e in zip(folds, Es):
        E[nor[b]] = e[nor[b]]
    np.save(p, E.astype(np.float32))
    return E


def band_err(E):
    return np.stack([E[:, i].mean(1) for i in BAND_IDX], 1)


class AERidge(BaseEstimator):
    """B. AE 끝단 ridge. head='band6' (대역 6개 오차) 또는 'bin' (1025 bin 오차)"""
    def __init__(self, head="band6"):
        self.head = head

    def _feat(self, E):
        return np.log10((band_err(E) if self.head == "band6" else E) + 1e-12)

    def fit(self, F, y):
        idx, y = F[:, 0].astype(int), np.asarray(y)
        self.E_ = oof_errors_exact(idx, y)
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 9), cv=4, scoring="neg_log_loss", class_weight="balanced",
            max_iter=5000)).fit(self._feat(self.E_[idx]), y)
        return self

    def decision_function(self, F):
        return self.clf_.decision_function(self._feat(self.E_[F[:, 0].astype(int)]))


# %% 중첩 CV
G = g[:, None]
CONFIGS = [
    ("B. 끝단 ridge: 대역 6개 오차 → L2 로지스틱", lambda: AERidge("band6"), True),
    ("B. 끝단 ridge: bin 1025개 오차 → L2 로지스틱", lambda: AERidge("bin"), False),
    ("기준 EXP-012 (alpha 0, 60 epoch)", lambda: SpecAE(0.0), False),
    ("A. AE 가중치 L2 alpha 0.01", lambda: SpecAE(0.01), False),
    ("A. AE 가중치 L2 alpha 0.1", lambda: SpecAE(0.1), False),
    ("A. AE 가중치 L2 alpha 1", lambda: SpecAE(1.0), False),
    ("A. AE 조기 종료", lambda: SpecAE(0.0, early=True), False),
]
R = {}
for name, mk, main in CONFIGS:
    R[name] = nested_cv(G, y, g, mk, n_jobs=-1)
    ex.record(name, R[name], names, main=main)

lines = []
for name, r in R.items():
    mi = [str(names[i]).replace("abn_", "") for i in np.flatnonzero(r["miss_counts"] >= 3)]
    fp = [str(names[i]).replace("nor_", "") for i in np.flatnonzero(r["fp_counts"] >= 3)]
    lines.append(f"{name}\n  미탐: {' '.join(mi) or '—'}\n  오탐: {' '.join(fp) or '—'}")
ex.note("구성별 미탐·오탐 파일 (seed 5번 중 3번 이상)", "\n".join(lines))
main_name = CONFIGS[0][0]
missed = [str(names[i]) for i in np.flatnonzero(R[main_name]["miss_counts"] >= 3)]
ex.focus = missed


# %% 진단: 정상 과적합 간격 (정상 80% 학습, 20%는 보지 않음) — A 구성별
rng = np.random.default_rng(0)
nor_idx = np.flatnonzero(y == 0)
held = np.sort(rng.choice(nor_idx, size=len(nor_idx) // 5, replace=False))
trn = np.setdiff1d(nor_idx, held)
diag, S_all = [], {}
for lab, a, e in (("alpha 0 (기준)", 0.0, False), ("alpha 0.01", 0.01, False), ("alpha 0.1", 0.1, False),
                  ("alpha 1", 1.0, False), ("조기 종료", 0.0, True)):
    s = ae_err(fit_ae(LP[trn].reshape(-1, LP.shape[-1]), a, e), g).mean(1)
    S_all[lab] = s
    gap = np.median(s[held]) / np.median(s[trn])
    auc = roc_auc_score(np.r_[np.zeros(len(held)), np.ones(30)], np.r_[s[held], s[y == 1]])
    diag.append((lab, gap, auc))
ex.note("과적합 간격 (정상 96개 학습)",
        "\n".join(f"{l:14s} 보지 않은 정상/학습 정상 오차 중앙값 비 = {gp:.2f} · 보지 않은 정상 vs 고장 AUC = {au:.3f}"
                  for l, gp, au in diag))

fig, axs = plt.subplots(1, len(S_all), figsize=(18, 4.5), sharey=False)
rj = np.random.default_rng(1)
grp = np.where(y == 1, 2, np.where(np.isin(g, held), 1, 0))
for a, (lab, s), (_, gp, au) in zip(axs, S_all.items(), diag):
    for k, col in enumerate((NOR, NOR, ABN)):
        v = np.log10(s[grp == k])
        a.scatter(k + rj.uniform(-.2, .2, len(v)), v, s=14, color=col, alpha=.35 if k == 0 else .9,
                  facecolor="white" if k == 1 else col, edgecolor=col, lw=1)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["정상\n학습", "정상\n안 봄", "고장"], fontsize=8)
    a.set_title(f"{lab}\n간격 {gp:.2f}배 · AUC {au:.2f}", fontsize=9); a.grid(alpha=.3, axis="y")
axs[0].set_ylabel("log10 복원 오차")
fig.suptitle("AE 가중치 정규화별 복원 오차: 학습 정상과 보지 않은 정상의 간격이 줄어드는가 (간격 = 오차 중앙값 비)", fontsize=11)
fig.tight_layout()
ex.save_fig(fig, "overfit_gap", "정규화를 세게 할수록 학습 정상·보지 않은 정상의 간격(과적합)이 줄고, 고장과의 분리는 유지되는가")


# %% 진단: 끝단 ridge 계수 (전체 150개로 한 번 학습, 해석용)
m6 = AERidge("band6").fit(G, y)
lr = m6.clf_[-1]
labels = [f"{lo}–{min(hi, 12500)}" for lo, hi in zip(BANDS[:-1], BANDS[1:])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
a1.barh(labels, lr.coef_[0], color=[ABN if v > 0 else NOR for v in lr.coef_[0]], height=.6)
a1.axvline(0, color="k", lw=.6); a1.grid(alpha=.3, axis="x")
a1.set_xlabel("표준화 계수 (+ = 이 대역 복원 오차가 클수록 고장)"); a1.set_ylabel("대역 (Hz)")
a1.set_title(f"끝단 ridge 계수 (선택된 C = {lr.C_[0]:.3g})", fontsize=10)
B6 = np.log10(band_err(m6.E_) + 1e-12)
for k, (lab, col) in enumerate((("정상 (OOF)", NOR), ("고장", ABN))):
    m = y == k
    a2.plot(range(6), np.median(B6[m], 0), "o-", color=col, label=f"{lab} 중앙값")
    a2.fill_between(range(6), *np.percentile(B6[m], [5, 95], 0), color=col, alpha=.15)
for nm in missed:
    a2.plot(range(6), B6[names.index(nm)], lw=.8, ls="--", color="k")
    a2.annotate(nm.replace("abn_", ""), (5.05, B6[names.index(nm), 5]), fontsize=8, va="center")
a2.set_xticks(range(6)); a2.set_xticklabels(labels, fontsize=8); a2.set_xlabel("대역 (Hz)")
a2.set_ylabel("log10 대역 복원 오차"); a2.legend(fontsize=8); a2.grid(alpha=.3)
a2.set_title("대역별 복원 오차 (띠 = 5–95%, 점선 = 대표 구성에서 놓친 고장)", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "ridge_head", "끝단 ridge가 어느 대역의 복원 오차를 고장 근거로 쓰는가, 놓친 고장은 대역 오차가 정상 띠 안인가")
ex.note("끝단 ridge 계수 (전체 데이터, 6대역)",
        "\n".join(f"{l:12s} {c:+.3f}" for l, c in zip(labels, lr.coef_[0])) + f"\nC = {lr.C_[0]:.3g}")


# %% 그림: 놓친 고장 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"놓친 고장 {' '.join(missed)}의 PSD vs 정상 5–95%"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done")
