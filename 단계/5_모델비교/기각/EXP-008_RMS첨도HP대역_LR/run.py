# %% [markdown]
# EXP-008 RMS·첨도·고역통과·옥타브 대역 에너지 + 로지스틱 회귀
# - 목적: 진동 진단의 기본 지표 몇 개(13개)만으로 선형 모델이 어디까지 가는가
# - 피처 (라벨을 보기 전에 고정. 대역·차단 주파수를 라벨 보고 고르지 않음)
#   1) 원신호(평균 제거) log RMS, log 첨도(kurtosis, 정규분포 = 3)
#   2) 고역통과(Butterworth 4차, 1 kHz, 영위상) 후 log RMS, log 첨도 → 저주파 운동을 빼고 충격성 성분만
#   3) 옥타브 대역 에너지 비율 log10 (25–50, 50–100, …, 6400–12500 Hz, 9개) → 스펙트럼 모양
# - 모델: StandardScaler + LogisticRegression(L2, C=1, class_weight=balanced)
# - 비교(같은 중첩 CV): 피처 묶음 빼기, C를 안쪽 CV로 고르기, 고역통과 차단 주파수 민감도(500·2000·5000 Hz)
#   대표(main)는 미리 정한 "13개 + C=1 + 1 kHz". 나머지는 해석용이며 최고점을 골라 대표로 바꾸지 않는다
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-008_RMS첨도HP대역_LR/run.py

# %% 설정
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score

FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"
EDGES = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12500]   # 옥타브 대역 경계 (Hz)

ex = Experiment(__file__, "EXP-008", "RMS·첨도·고역통과·대역 에너지 + 로지스틱 회귀", [],
                member="이동훈", model="로지스틱 회귀 (L2, C=1)",
                changes="RMS·첨도·1 kHz 고역통과 RMS·첨도·옥타브 대역 에너지 비율 13개 + 로지스틱",
                focus=())
X, y, names = load_train()
Xc = X - X.mean(1, keepdims=True)
g = np.arange(len(y))


# %% 피처 추출
def highpass(x, fc):
    return sosfiltfilt(butter(4, fc, btype="highpass", fs=FS, output="sos"), x, axis=-1)


def band_fractions(x):
    """옥타브 대역별 에너지 / 25–12500 Hz 전체 에너지 (log10)"""
    P = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], 1 / FS)
    E = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in zip(EDGES[:-1], EDGES[1:])], 1)
    return np.log10(E / E.sum(1, keepdims=True))


def features(Xc, fc=1000):
    H = highpass(Xc, fc)
    T = pd.DataFrame({
        "log_rms": np.log10(np.sqrt((Xc ** 2).mean(1))),
        "log_kurt": np.log10(kurtosis(Xc, axis=1, fisher=False)),
        "log_hp_rms": np.log10(np.sqrt((H ** 2).mean(1))),
        "log_hp_kurt": np.log10(kurtosis(H, axis=1, fisher=False)),
    })
    B = band_fractions(Xc)
    for k, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        T[f"band_{lo}_{hi}"] = B[:, k]
    return T


T = features(Xc)
HP_ALT = {fc: features(Xc, fc)[["log_hp_rms", "log_hp_kurt"]].values for fc in (500, 2000, 5000)}
cols = list(T.columns)
BASIC, HP, BAND = cols[:2], cols[2:4], cols[4:]
T.assign(name=names, y=y).to_csv(os.path.join(ex.dir, "features.csv"), index=False)

auc = pd.DataFrame({"feat": cols, "auc": [roc_auc_score(y, T[c]) for c in cols]})
auc["dir"] = np.where(auc.auc >= .5, "+", "-")
auc["auc"] = np.maximum(auc.auc, 1 - auc.auc)
ex.note("단변량 AUC (참고, 전체 데이터. 방향 + = 고장에서 큼)",
        auc.sort_values("auc", ascending=False).to_string(index=False, float_format="%.3f"))


# %% 모델
def make_lr(C=1.0):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def make_lrcv():
    """C를 바깥 학습 파일 안에서 안쪽 4겹 CV(로그손실)로 고른다"""
    return lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv=4, scoring="neg_log_loss",
        class_weight="balanced", max_iter=5000))


def run(name, F, maker=None, main=False):
    r = nested_cv(np.asarray(F, dtype=float), y, g, maker or make_lr())
    return r, ex.record(name, r, names, main=main)


# %% 중첩 CV
r_main, row_main = run("13개 (RMS·첨도·HP 1k·대역 9) C=1", T.values, main=True)
run("RMS·첨도만 (2)", T[BASIC].values)
run("RMS·첨도 + HP 1k (4)", T[BASIC + HP].values)
run("대역 에너지만 (9)", T[BAND].values)
run("13개, C 안쪽 CV 선택", T.values, make_lrcv())
for fc, A in HP_ALT.items():
    run(f"13개, HP 차단 {fc} Hz", np.hstack([T[BASIC].values, A, T[BAND].values]))

missed = [str(names[i]) for i in np.flatnonzero(r_main["miss_counts"] >= 3)]
fps = [str(names[i]) for i in np.flatnonzero(r_main["fp_counts"] >= 3)]
ex.focus = missed


# %% 그림 1: 피처 분포 (정상·고장, 놓친 고장 표시)
def strip_fig():
    n = len(cols); nc = 5; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(16, 3.0 * nr))
    rng = np.random.default_rng(0)
    for a, c in zip(axs.flat, cols):
        v = T[c].values
        lo, hi = np.percentile(v[y == 0], [5, 95])
        a.axhspan(lo, hi, color=NOR, alpha=.12, lw=0)
        for lab, col, xpos in ((0, NOR, 0), (1, ABN, 1)):
            m = y == lab
            a.scatter(xpos + rng.uniform(-.18, .18, m.sum()), v[m], s=12, color=col, alpha=.7, lw=0)
        for nm in missed:
            i = names.index(nm)
            a.scatter(1.3, v[i], s=40, facecolor="none", edgecolor="k", lw=1)
            a.annotate(nm.replace("abn_", ""), (1.36, v[i]), fontsize=7, va="center")
        a.set_xticks([0, 1]); a.set_xticklabels(["정상", "고장"]); a.set_xlim(-.5, 1.8)
        r = auc.set_index("feat").loc[c]
        a.set_title(f"{c}  (AUC {r.auc:.2f}{r.dir})", fontsize=9); a.grid(alpha=.3, axis="y")
    for a in list(axs.flat)[n:]:
        a.axis("off")
    fig.suptitle("피처 13개 분포. 파란 띠 = 정상 5–95%, 검은 원 = 대표 구성에서 자주 놓친 고장", fontsize=11)
    fig.tight_layout()
    return fig


ex.save_fig(strip_fig(), "feature_strip", "13개 피처에서 정상·고장이 갈리는가, 놓친 고장은 정상 띠 안에 있는가")


# %% 그림 2: 로지스틱 계수 (전체 데이터로 한 번 학습, 해석용. 점수에는 쓰지 않음)
m_all = make_lr()().fit(T.values, y)
coef = pd.Series(m_all[-1].coef_[0], index=cols).sort_values()
fig, a = plt.subplots(figsize=(8, 5))
a.barh(coef.index, coef.values, color=[ABN if v > 0 else NOR for v in coef.values], height=.6)
a.axvline(0, color="k", lw=.6); a.grid(alpha=.3, axis="x")
a.set_xlabel("표준화 계수 (+ = 클수록 고장 쪽)")
a.set_title("로지스틱 회귀 계수 (전체 150개로 학습, 해석용)")
fig.tight_layout()
ex.save_fig(fig, "lr_coef", "모델이 어떤 피처를 고장 쪽 근거로 쓰는가 (상관된 피처끼리는 계수가 나눠지거나 부호가 바뀔 수 있음)")
ex.note("로지스틱 계수 (전체 데이터, 표준화)", coef.iloc[::-1].to_string(float_format="%.3f"))


# %% 그림 3: 고역통과 전후 파형 (정상 중앙 RMS 파일 vs 고장 중 HP 첨도 최대 파일 + 놓친 고장)
i_n = int(np.flatnonzero(y == 0)[np.argsort(T.log_rms[y == 0].values)[(y == 0).sum() // 2]])
i_a = int(np.flatnonzero(y == 1)[np.argmax(T.log_hp_kurt[y == 1].values)])
show = [i_n, i_a] + [names.index(n) for n in missed[:2]]
t = np.arange(X.shape[1]) / FS
w = (t >= 0.5) & (t < 0.7)
H1 = highpass(Xc[show], 1000)
fig, axs = plt.subplots(len(show), 2, figsize=(14, 2.0 * len(show)), sharex=True, squeeze=False)
for k, i in enumerate(show):
    col = NOR if y[i] == 0 else ABN
    axs[k, 0].plot(t[w], Xc[i, w], lw=.5, color=col); axs[k, 1].plot(t[w], H1[k, w], lw=.5, color=col)
    axs[k, 0].set_title(f"{names[i]} 원신호", fontsize=9, loc="left")
    axs[k, 1].set_title(f"{names[i]} 1 kHz 고역통과 (첨도 {10 ** T.log_hp_kurt[i]:.1f})", fontsize=9, loc="left")
    for a in axs[k]: a.grid(alpha=.3)
axs[-1, 0].set_xlabel("초"); axs[-1, 1].set_xlabel("초")
fig.tight_layout()
ex.save_fig(fig, "highpass_wave", "고역통과로 저주파 운동을 빼면 충격이 드러나는가 (0.5–0.7초 확대, 위: 정상 대표·고장 HP 첨도 최대, 아래: 놓친 고장)")


# %% 그림 4: 놓친 고장 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"놓친 고장 {' '.join(missed)}의 PSD vs 정상 5–95%"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
    ex.save_fig(viz.feature_percentile(T, y, names, missed, cols), "missed_percentile",
                "놓친 고장의 13개 피처가 정상 분포에서 몇 백분위인가 (0·100 근처여야 잡을 수 있음)")
if fps:
    ex.save_fig(viz.feature_percentile(T, y, names, fps, cols), "fp_percentile",
                "자주 오경보한 정상 파일의 피처 백분위 (어느 피처가 정상 범위 끝에 있어 고장처럼 보였나)")

ex.note("대표 구성 놓친 고장·오경보", f"자주 놓침(3회+, {len(missed)}개): {' '.join(missed)}\n자주 오경보(3회+, {len(fps)}개): {' '.join(fps)}")
print("done")
