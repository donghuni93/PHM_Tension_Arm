# %% [markdown]
# EXP-012 스펙트럼 MLP 오토인코더 (신효섭 보고서 재현) + t-SNE
# - 출처: 참고자료/팀원 공유 정보/신효섭_텐션암 이상감지 분석 보고서.pdf 5장 "오토인코더 구조"
#   · 입력: 평균 제거 → 200 Hz 고역통과(Butterworth 4차, 양방향) → 파일마다 1초 구간 8개
#           → log Welch PSD (1025 bin) → 학습 정상 기준 bin별 표준화
#   · 구조: 완전연결 1025 → 128 → 16 → 128 → 1025, ReLU. MSE, Adam(학습률 0.001), 60 epoch
#   · 정상만 학습. 점수 = 복원 오차 (파일 점수 = 구간 8개 오차의 평균 — 보고서에 집계 방식이 없어 평균으로 정함)
# - 보고서와 다른 점
#   · 프레임워크: torch 대신 sklearn MLPRegressor(같은 층 구조·활성화·손실·최적화). 병목에도 ReLU
#   · 평가: 보고서는 학습 정상 75%로 학습, 25%+고장으로 임계값. 여기서는 우리 기준(중첩 CV, 임계값 = 안쪽 OOF F2 최대)
#   · 최종안의 "AE OR 피처 12개 LR" 중 LR 쪽은 구현하지 않음 (AE 단독)
# - t-SNE: AE "끝단"을 두 곳에서 본다 — 병목(인코더 끝, 16차원 잠재) / 출력 끝(bin별 복원 오차 1025차원).
#   비교용으로 AE를 거치지 않은 입력 스펙트럼도. 정상 일부를 학습에서 빼서(보지 않은 정상) 고장과 공정하게 비교
# 실행: conda run -n qe-phm python 단계/5_모델비교/유의미/EXP-012_스펙트럼AE_tSNE/run.py

# %% 설정
import os, sys, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"
SEG, NSEG, NPERSEG = FS, 8, 2048          # 1초 구간 8개, Welch 2048 → 1025 bin
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ex = Experiment(__file__, "EXP-012", "스펙트럼 MLP 오토인코더 (신효섭 보고서 재현) + t-SNE", [],
                member="이동훈", model="MLP 오토인코더 (정상만 학습)",
                changes="200 Hz 고역통과 log PSD(1초×8구간) → AE 1025-128-16-128-1025 복원 오차", focus=())
X, y, names = load_train()
g = np.arange(len(y))


# %% 입력: 구간별 log PSD (라벨 안 씀 → 전체에 한 번 계산해도 누수 없음. 표준화는 모델 안에서 학습 정상으로)
def segment_logpsd(X):
    Xc = X - X.mean(1, keepdims=True)
    H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
    starts = np.linspace(0, X.shape[1] - SEG, NSEG).astype(int)       # 0, 3571, …, 25000
    S = np.stack([H[:, s:s + SEG] for s in starts], 1)                # (파일, 8, 25000)
    f, P = welch(S, fs=FS, nperseg=NPERSEG, axis=-1)
    return f, np.log10(P + 1e-20)                                      # (파일, 8, 1025)


p = os.path.join(CACHE, "seg_logpsd_hp200_8x1s.npz")
if os.path.exists(p):
    z = np.load(p); f_psd, LP = z["f"], z["LP"]
else:
    f_psd, LP = segment_logpsd(X); np.savez(p, f=f_psd, LP=LP)


# %% 모델
class SpecAE(BaseEstimator):
    """파일 번호(1열)를 받아 그 파일의 구간 스펙트럼으로 학습·채점하는 오토인코더.
    fit: 받은 파일 중 정상(y=0)의 구간만 사용. 표준화 평균·표준편차도 그 정상으로"""
    def __init__(self, LP, hidden=(128, 16, 128), epochs=60, lr=1e-3, seed=0):
        self.LP, self.hidden, self.epochs, self.lr, self.seed = LP, hidden, epochs, lr, seed

    def _seg(self, idx):
        return self.LP[np.asarray(idx).ravel().astype(int)]           # (파일, 8, 1025)

    def fit(self, F, y):
        S = self._seg(F)[np.asarray(y) == 0].reshape(-1, self.LP.shape[-1])
        self.mu_, self.sd_ = S.mean(0), S.std(0) + 1e-8
        Z = (S - self.mu_) / self.sd_
        self.m_ = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu", solver="adam", learning_rate_init=self.lr,
                               max_iter=self.epochs, alpha=0.0, random_state=self.seed).fit(Z, Z)
        return self

    def _z(self, F):
        return (self._seg(F) - self.mu_) / self.sd_                    # (파일, 8, 1025)

    def err(self, F):
        """bin별 복원 제곱오차, 구간 평균 → (파일, 1025)"""
        Z = self._z(F); n = Z.shape[0]
        R = self.m_.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
        return ((R - Z) ** 2).mean(1)

    def latent(self, F):
        """병목(16차원) 활성값, 구간 평균 → (파일, 16)"""
        Z = self._z(F); h = Z.reshape(-1, Z.shape[-1])
        W, b = self.m_.coefs_, self.m_.intercepts_
        for k in range(len(self.hidden) // 2 + 1):                     # 1025→128→16
            h = np.maximum(h @ W[k] + b[k], 0)
        return h.reshape(Z.shape[0], Z.shape[1], -1).mean(1)

    def decision_function(self, F):
        return self.err(F).mean(1)


make_ae = lambda: SpecAE(LP)
r = nested_cv(g[:, None], y, g, make_ae, n_jobs=-1)
ex.record("스펙트럼 AE (보고서 구조)", r, names, main=True)
missed = [str(names[i]) for i in np.flatnonzero(r["miss_counts"] >= 3)]
fps = [str(names[i]) for i in np.flatnonzero(r["fp_counts"] >= 3)]
ex.focus = missed
ex.note("놓친 고장·오경보 (seed 5번 중 3번 이상)",
        f"자주 놓침({len(missed)}개): {' '.join(missed)}\n자주 오경보({len(fps)}개): {' '.join(fps)}")


# %% t-SNE용 AE 하나: 정상 80%로만 학습. 나머지 정상 24개 + 고장 30개는 모델이 보지 않은 파일
rng = np.random.default_rng(0)
nor_idx = np.flatnonzero(y == 0)
held = np.sort(rng.choice(nor_idx, size=len(nor_idx) // 5, replace=False))
train = np.setdiff1d(nor_idx, held)
ae = SpecAE(LP).fit(train[:, None], np.zeros(len(train)))
Zin = ae._z(g[:, None]).mean(1)            # 입력 (표준화 스펙트럼, 구간 평균)
LAT = ae.latent(g[:, None])                # 병목 16차원
ERR = ae.err(g[:, None])                   # 출력 끝 bin별 복원 오차
score = ERR.mean(1)
grp = np.where(y == 1, 2, np.where(np.isin(g, held), 1, 0))   # 0 학습 정상, 1 보지 않은 정상, 2 고장
auc_hold = roc_auc_score(np.r_[np.zeros(len(held)), np.ones(30)], np.r_[score[held], score[y == 1]])
dead = int((LAT.max(0) <= 0).sum())
ex.note("t-SNE용 AE (정상 96개 학습)",
        f"보지 않은 정상 {len(held)}개 vs 고장 30개 복원오차 AUC = {auc_hold:.3f}\n"
        f"병목 16개 중 항상 0인(죽은) 유닛 = {dead}개")


def tsne(A, seed=0):
    A = (A - A.mean(0)) / (A.std(0) + 1e-8)
    return TSNE(2, perplexity=30, init="pca", random_state=seed).fit_transform(A)


EMB = {"입력: 표준화 log PSD (1025차원)": tsne(Zin),
       "AE 병목: 잠재 벡터 (16차원)": tsne(LAT),
       "AE 출력 끝: bin별 복원 오차 (1025차원)": tsne(np.log10(ERR + 1e-12))}


# %% 그림 1: t-SNE 3개
def tsne_fig(EMB, color_by_score=False):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for a, (title, E) in zip(axs, EMB.items()):
        if color_by_score:
            sc = a.scatter(E[:, 0], E[:, 1], c=np.log10(score), cmap="viridis", s=22,
                           marker="o", edgecolor="none")
            a.scatter(E[y == 1, 0], E[y == 1, 1], s=60, facecolor="none", edgecolor=ABN, lw=1.2, label="고장")
            fig.colorbar(sc, ax=a, label="log10 복원 오차", shrink=.8)
        else:
            a.scatter(*E[grp == 0].T, s=18, color=NOR, alpha=.45, lw=0, label="정상 (AE 학습)")
            a.scatter(*E[grp == 1].T, s=34, facecolor="white", edgecolor=NOR, lw=1.3, label="정상 (보지 않음)")
            a.scatter(*E[grp == 2].T, s=30, color=ABN, lw=0, label="고장 (보지 않음)")
        for nm in missed:
            i = names.index(nm)
            a.annotate(nm.replace("abn_", ""), E[i], xytext=(4, 4), textcoords="offset points", fontsize=8)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    axs[0].legend(fontsize=8, loc="best")
    fig.suptitle("t-SNE (perplexity 30). 번호 = 중첩 CV에서 자주 놓친 고장" +
                 (" · 색 = 복원 오차(이상 점수)" if color_by_score else ""), fontsize=11)
    fig.tight_layout()
    return fig


ex.save_fig(tsne_fig(EMB), "tsne_group",
            "입력 → 병목 → 출력 끝으로 갈수록 고장(주황)이 정상(파랑)과 따로 모이는가. 흰 원 = AE가 보지 않은 정상")
ex.save_fig(tsne_fig(EMB, True), "tsne_score",
            "같은 배치를 복원 오차로 칠함. 오차가 큰 영역과 고장이 겹치는가")


# %% 그림 2: 복원 오차 분포 + 주파수별 복원 오차
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 2]})
rj = np.random.default_rng(1)
for k, (lab, col) in enumerate((("정상 학습", NOR), ("정상 보지 않음", NOR), ("고장", ABN))):
    v = np.log10(score[grp == k])
    a1.scatter(k + rj.uniform(-.2, .2, len(v)), v, s=16, color=col, alpha=.4 if k == 0 else .9,
               facecolor="white" if k == 1 else col, edgecolor=col, lw=1)
a1.axhline(np.log10(np.percentile(score[held], 95)), color="k", lw=.7, ls="--")
a1.text(2.45, np.log10(np.percentile(score[held], 95)), "보지 않은 정상 95%", fontsize=8, va="bottom", ha="right")
for nm in missed:
    i = names.index(nm); a1.annotate(nm.replace("abn_", ""), (2.25, np.log10(score[i])), fontsize=8, va="center")
a1.set_xticks([0, 1, 2]); a1.set_xticklabels(["정상\n(학습)", "정상\n(보지 않음)", "고장"])
a1.set_ylabel("log10 복원 오차"); a1.set_title(f"파일별 복원 오차 (보지 않은 정상 vs 고장 AUC {auc_hold:.2f})", fontsize=10)
a1.grid(alpha=.3, axis="y")
for k, (lab, col, ls) in enumerate((("정상 학습", NOR, "-"), ("정상 보지 않음", NOR, "--"), ("고장", ABN, "-"))):
    a2.plot(f_psd, np.median(np.log10(ERR[grp == k]), 0), color=col, ls=ls, lw=1.2, label=f"{lab} 중앙값")
a2.set_xlabel("Hz"); a2.set_ylabel("log10 복원 제곱오차"); a2.set_xlim(0, FS / 2)
a2.set_title("주파수별 복원 오차: 고장은 어느 대역을 정상처럼 복원하지 못하나", fontsize=10)
a2.legend(fontsize=8); a2.grid(alpha=.3)
fig.tight_layout()
ex.save_fig(fig, "recon_error", "복원 오차가 고장에서 큰가, 어느 주파수에서 큰가 (t-SNE용 AE 기준)")


# %% 그림 3: 놓친 고장 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"놓친 고장 {' '.join(missed)}의 PSD vs 정상 5–95%"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done")
