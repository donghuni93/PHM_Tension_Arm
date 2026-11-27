# %% [markdown]
# 2-22 출렁임·센서 방향 영향 — 원래 평균과 느린 출렁임이 지금 모델 입력(RMS·첨도 등)을 얼마나 흔드나
# - 배운 것: 평균(오프셋) = 중력이 센서 축에 얹힌 값(Stage 1), 가속도 파워의 대역별 몫(Stage 1, 11-20 계산)
# - 비교: 지금 ET 원신호 피처(평균만 뺌) vs 같은 피처를 200 Hz 고역통과 신호로 (피처 후보 C1·C2)
#   200 Hz는 AE 입력과 같은 값으로 미리 정함 (라벨 보고 고르지 않음)
# - "끌려감" = 그 피처와 20 Hz 아래 출렁임 크기의 순위상관. 높으면 고장보다 운전 조건을 잰다
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.dirname(HERE))
from _common import *
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, spearmanr
from sklearn.metrics import roc_auc_score

X, y, names, log, save, flush = setup(HERE)
nor, abn = y == 0, y == 1
raw_mean = X.mean(1)
Xc = X - raw_mean[:, None]
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
LO = sosfiltfilt(butter(4, 20, btype="lowpass", fs=FS, output="sos"), Xc, axis=-1)
sway = LO.std(1)

# %% 1. 원래 평균 = 센서 방향
odd = np.flatnonzero(raw_mean < -0.5)
log("원래 평균 < −0.5 g: " + ", ".join(f"{names[i]} {raw_mean[i]:.2f} g" for i in odd))
rest = np.delete(np.arange(len(X)), odd)
log(f"나머지 {len(rest)}개 범위 {raw_mean[rest].min():.3f} ~ {raw_mean[rest].max():.3f} g")
log(f"원래 평균 AUC {roc_auc_score(y, raw_mean):.2f} · −1 g 파일 4개 빼고 {roc_auc_score(y[rest], raw_mean[rest]):.2f} (사후)")

f, P = welch(Xc, fs=FS, nperseg=16384)
edges = [0, 2, 20, 200, 1000, 5000, 12500]
share = np.stack([P[:, (f >= a) & (f < b)].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
share = share / share.sum(1, keepdims=True)
bl = [f"{a}–{b}" for a, b in zip(edges[:-1], edges[1:])]

fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1, 2]))
ax = axs[0]
rng = np.random.default_rng(0)
for k, (m, c) in enumerate(((nor, NOR), (abn, ABN))):
    ax.scatter(k + rng.uniform(-.15, .15, m.sum()), raw_mean[m], s=14, color=c, alpha=.7)
on = [i for i in odd if y[i] == 0]
ax.annotate("nor_" + "·".join(names[i][-3:] for i in on), (.2, raw_mean[on].mean()), fontsize=8)
for i in odd:
    if y[i] == 1:
        ax.annotate(names[i], (1.2, raw_mean[i]), fontsize=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"]); ax.set_ylabel("원래 평균 (g)")
ax.set_title("원래 평균: 4개만 -1 g 근처 (센서가 중력 방향을 보고 있음)")
ax.grid(alpha=.3)
ax = axs[1]
o = np.r_[np.flatnonzero(nor)[np.argsort(-share[nor, :2].sum(1))], np.flatnonzero(abn)[np.argsort(-share[abn, :2].sum(1))]]
bottom = np.zeros(len(o))
colors = ["#08306b", "#2171b5", "#6baed6", "#fdae6b", "#e6550d", "#a63603"]
for k in range(len(bl)):
    ax.bar(np.arange(len(o)), share[o, k], bottom=bottom, color=colors[k], width=1, label=bl[k] + " Hz")
    bottom += share[o, k]
ax.axvline(nor.sum() - .5, color="k", lw=2)
ax.text(nor.sum() / 2, 1.02, "정상", ha="center"); ax.text(nor.sum() + abn.sum() / 2, 1.02, "고장", ha="center")
ax.set_xlim(-.5, len(o) - .5); ax.set_ylim(0, 1.08); ax.set_xticks([])
ax.set_ylabel("가속도 파워 몫"); ax.legend(fontsize=8, ncol=3, loc="lower right")
ax.set_title("파일마다 파워가 어느 대역에 있나 (그룹 안에서 20 Hz 아래 몫 순)")
save(fig, "1_원래평균_대역별파워몫")
for k, b in enumerate(bl):
    log(f"파워 몫 {b} Hz: 정상 중앙 {np.median(share[nor, k]):.2f} · 고장 중앙 {np.median(share[abn, k]):.2f}")

# %% 2. 같은 피처를 원신호(평균만 뺌) vs 200 Hz 고역통과로
def feats(Z):
    rms = np.sqrt((Z ** 2).mean(1)); peak = np.abs(Z).max(1); mav = np.abs(Z).mean(1)
    w = np.stack([np.sqrt((s ** 2).mean(1)) for s in np.array_split(Z, 10, axis=1)], 1)
    return pd.DataFrame({"RMS": rms, "최고값": peak, "첨도": kurtosis(Z, axis=1, fisher=False),
                         "파고율": peak / rms, "충격계수": peak / mav, "창RMS_흩어짐": w.std(1)})
A, B = feats(Xc), feats(H)
rows = []
for c in A.columns:
    rows.append(dict(피처=c,
                     끌려감_원신호=spearmanr(A[c], sway).correlation, 끌려감_200Hz=spearmanr(B[c], sway).correlation,
                     AUC_원신호=roc_auc_score(y, A[c]), AUC_200Hz=roc_auc_score(y, B[c]),
                     같은값_ρ=spearmanr(A[c], B[c]).correlation))
R = pd.DataFrame(rows)
log("\n피처: 출렁임과 순위상관(원신호 → 200 Hz) · AUC(원신호 → 200 Hz, 사후) · 두 버전 순위상관")
for _, r in R.iterrows():
    log(f"  {r.피처}: 끌려감 {r.끌려감_원신호:+.2f} → {r.끌려감_200Hz:+.2f} · AUC {r.AUC_원신호:.2f} → {r.AUC_200Hz:.2f} · 같은값 {r.같은값_ρ:.2f}")
R.to_csv(os.path.join(HERE, "raw_vs_hp200.csv"), index=False, encoding="utf-8-sig")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
xs = np.arange(len(R)); w = .38
for ax, (a, b, t) in zip(axs, (("끌려감_원신호", "끌려감_200Hz", "출렁임(20 Hz 아래)과의 순위상관 — 0에 가까울수록 운전 조건에 덜 끌려감"),
                               ("AUC_원신호", "AUC_200Hz", "정상·고장 가르는 힘 (AUC, 사후)"))):
    ax.bar(xs - w / 2, R[a], w, color="#999999", label="원신호 (평균만 뺌, 지금)")
    ax.bar(xs + w / 2, R[b], w, color=NOR, label="200 Hz 고역통과")
    ax.set_xticks(xs); ax.set_xticklabels(R.피처, rotation=20); ax.set_title(t, fontsize=10); ax.grid(alpha=.3, axis="y")
axs[1].axhline(.5, color="gray", ls=":"); axs[1].set_ylim(.4, 1)
axs[0].axhline(0, color="k", lw=.8); axs[0].legend(fontsize=8)
save(fig, "2_원신호vs200Hz_피처")

# %% 3. 예: 출렁임이 큰 정상 vs 작은 고장 — RMS 비교
i_hi = np.flatnonzero(nor)[np.argmax(sway[nor])]
i_lo = names.index("abn_017")
fig, axs = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
t = np.arange(X.shape[1]) / FS
for ax, i, c in zip(axs, (i_hi, i_lo), (NOR, ABN)):
    ax.plot(t, Xc[i], color="#bbbbbb", lw=.5, label="원신호 (평균만 뺌)")
    ax.plot(t, H[i], color=c, lw=.5, label="200 Hz 고역통과")
    ax.set_title(f"{names[i]}: RMS 원신호 {A.RMS[i]*1000:.1f} mg → 200 Hz {B.RMS[i]*1000:.1f} mg · 출렁임 {sway[i]*1000:.1f} mg", fontsize=10)
    ax.set_ylabel("g"); ax.legend(fontsize=8, loc="upper right")
axs[1].set_xlabel("시간 (s)")
save(fig, "3_예시_출렁임큰정상_조용한고장")
log(f"\n예시: {names[i_hi]} 출렁임 {sway[i_hi]*1000:.1f} mg (정상 최대) · abn_017 출렁임 {sway[i_lo]*1000:.1f} mg")
flush()
