# %% [markdown]
# 14. 감쇠비 정밀 구분: (울림 주파수 f, 감쇠비 ζ) 평면에서 "고장이 한 번도 가지 않은 영역"을 찾아 정상 판정에 쓸 수 있는가
#     + 앞쪽 대역(f < 2.7 kHz)에서 정상과 고장이 섞이는데, ζ 말고 어떤 지표가 둘을 가르는가
# 기존(src/final.py impact_rings)의 한계: 감쇠비 격자가 0.01·0.02·0.04·0.08 네 개뿐이고 창이 8 ms라, 9.4 kHz 긴 울림(ζ ≈ 0.001)을 표현할 수 없다.
# 여기서는 ζ 격자를 0.001–0.064(7단계, 2배 간격), f 격자를 800–12000 Hz(100 Hz 간격), 창을 16 ms로 늘렸다. 충격 검출은 기존과 같다.
# 영역 검증(누수 시험): 5겹 층화 교차검증에서 학습 파일로 (f, ζ) 칸을 정하고, 보류한 파일의 충격이 그 칸에 떨어지는지 센다.
#   후보 칸 = 학습 정상 충격이 N_MIN개 이상 있고 학습 고장 충격이 0개인 칸. 파일 내림 = 충격 ≥ 3개이고 90% 이상이 후보 칸.
#   (기존 본체 모드 내림 = 충격 ≥ 3 & 5.5–7.75 kHz 울림 ≥ 90%. 기존 규칙과 비교해 얼마나 더 정상을 내리는가를 본다)
# 보기용이다. 사후 영역을 규칙으로 쓰려면 새 EXP로 폴드 안에서 검증한다. 실행: python "3_단계/2_EDA/아카이브/휴대폰신호해부/2-18_1119_휴대폰_원시신호분석/14_감쇠비_정밀구분/run.py"

# %% 설정
import os, sys
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000; SEG = int(0.016 * FS); FIT_MIN = 0.7
ZETAS = np.array([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
FREQS = np.arange(800, 12001, 100)
FEDGE = np.arange(800, 12001, 500)                 # 영역 칸: f 500 Hz × ζ 7단계
N_MIN = 15
X, y, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))

FP_NORMAL = [116, 59, 55, 106]
FN_FAULT = [22, 14, 17, 2]
def find(num, label):
    for i, (n, l) in enumerate(zip(names, y)):
        if l == label and int("".join(c for c in n if c.isdigit()) or -1) == num:
            return i
    raise KeyError(f"{num} (label {label}) 없음. names 예: {names[:3]}")
IDX_FP = [find(n, 0) for n in FP_NORMAL]; IDX_FN = [find(n, 1) for n in FN_FAULT]

# %% 정밀 울림 맞춤: 충격마다 (f, ζ, fit, 높이)
t = np.arange(SEG) / FS
atoms, par = [], []
for f in FREQS:
    for z in ZETAS:
        w = 2 * np.pi * f; wd = w * np.sqrt(1 - z ** 2); a = np.exp(-z * w * t)
        Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1)); atoms.append(Q); par.append((f, z))
atoms = np.stack(atoms); par = np.array(par)
X1k = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
rows = []
for i, x in enumerate(X1k):
    env = sosfiltfilt(lp500, np.abs(hilbert(x)))
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    pk = pk[pk + SEG <= len(x)]
    if not len(pk): continue
    S = np.stack([x[p:p + SEG] for p in pk]); amp = np.abs(S[:, :50]).max(1)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2); b = fit.argmax(1)
    rows += [(i, par[bb, 0], par[bb, 1], ft, am) for bb, ft, am in zip(b, fit.max(1), amp)]
I = pd.DataFrame(rows, columns=["file", "f", "zeta", "fit", "amp"])
I["y"] = y[I["file"].values]
I.to_csv(os.path.join(HERE, "impacts_fz.csv"), index=False)
print("충격 수: 정상", int((I.y == 0).sum()), "고장", int((I.y == 1).sum()), "| fit ≥ 0.7:", int((I.fit >= FIT_MIN).sum()))
G = I[I.fit >= FIT_MIN].copy()

# %% 그림 1: 충격 단위 (f, ζ) 산점도 — 정상 vs 고장 + 경계 파일
fig, ax = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
rng = np.random.default_rng(0)
zi = np.log2(G.zeta.values / 0.001) + rng.uniform(-0.18, 0.18, len(G))        # ζ는 격자라 지터를 줌
for a, lab, c in ((ax[0], 0, "gray"), (ax[1], 1, "tab:red")):
    m = (G.y == lab).values; a.scatter(G.f[m] / 1000, zi[m], s=8, color=c, alpha=0.35)
for i in IDX_FP:
    m = (G.file == i).values; ax[0].scatter(G.f[m] / 1000, zi[m], s=30, color="tab:blue", label=names[i])
for i in IDX_FN:
    m = (G.file == i).values; ax[1].scatter(G.f[m] / 1000, zi[m], s=30, label=names[i])
for a, tt in zip(ax, ("정상 충격 (파랑 = 오탐 경계 정상)", "고장 충격 (색 = 미탐 경계 고장)")):
    a.set_title(tt); a.set_xlabel("울림 주파수 (kHz)"); a.legend(fontsize=7)
    a.set_yticks(range(len(ZETAS))); a.set_yticklabels([f"{z:g}" for z in ZETAS])
ax[0].set_ylabel("감쇠비 ζ (위로 갈수록 감쇠 큼)")
fig.savefig(os.path.join(FIG, "fz_scatter.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 칸 정의와 누수 시험 (5겹 × 5회)
def cell(f, z):
    return np.digitize(f, FEDGE) * 10 + np.searchsorted(ZETAS, z - 1e-9)
G["cell"] = cell(G.f.values, G.zeta.values)
nf = {i: g for i, g in G.groupby("file")}
res = []; flagged = {i: 0 for i in range(len(y))}; flag_by_seed = []
for seed in range(5):
    leak_n = leak_f = tot_n = tot_f = 0
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        gtr = G[G.file.isin(tr)]
        cn = gtr[gtr.y == 0].cell.value_counts(); cf = set(gtr[gtr.y == 1].cell)
        cand = {c for c, n in cn.items() if n >= N_MIN and c not in cf}
        for i in te:
            g = nf.get(i)
            if g is None: continue
            inn = g.cell.isin(cand)
            if y[i] == 0: leak_n += inn.sum(); tot_n += len(g)
            else: leak_f += inn.sum(); tot_f += len(g)
            if len(g) >= 3 and inn.mean() >= 0.9: flagged[i] += 1
    res.append((seed, leak_n / tot_n, leak_f / tot_f, len(cand)))
R = pd.DataFrame(res, columns=["seed", "보류 정상 충격 중 후보 칸 비율", "보류 고장 충격 중 후보 칸 비율", "후보 칸 수(마지막 겹)"])
print(R.round(3).to_string(index=False)); print("평균 정상 {:.3f} / 고장 {:.3f}".format(R.iloc[:, 1].mean(), R.iloc[:, 2].mean()))
fl = pd.DataFrame({"파일": names, "라벨": y, "내림 횟수(5회 중)": [flagged[i] for i in range(len(y))]})
fl.to_csv(os.path.join(HERE, "zone_flag_by_file.csv"), index=False, encoding="utf-8-sig")
hit = fl[fl["내림 횟수(5회 중)"] >= 3]
print("5회 중 3회 이상 내림 판정: 정상", int((hit.라벨 == 0).sum()), "/120, 고장", int((hit.라벨 == 1).sum()), "/30 | 고장 목록:", list(hit[hit.라벨 == 1].파일))
print("경계 정상 4개:", {names[i]: flagged[i] for i in IDX_FP})

# %% 기존 본체 모드 내림과 비교 (5.5–7.75 kHz 울림 ≥ 90%, 충격 ≥ 3, 격자만 기존 대신 이번 정밀 맞춤)
old = {}
for i, g in nf.items():
    if len(g) >= 3 and ((g.f >= 5500) & (g.f < 7750)).mean() >= 0.9: old[i] = 1
print("기존식 본체 내림(이번 정밀 맞춤 기준): 정상", sum(1 for i in old if y[i] == 0), "고장", sum(1 for i in old if y[i] == 1),
      "| 경계 정상:", {names[i]: int(i in old) for i in IDX_FP})

# %% 그림 2: 칸별 정상/고장 충격 수 (후보 칸 = 정상만 있는 칸)
fb = len(FEDGE) + 1
Hn = np.zeros((len(ZETAS), fb)); Hf = np.zeros_like(Hn)
for c, n in G[G.y == 0].cell.value_counts().items(): Hn[c % 10, c // 10] = n
for c, n in G[G.y == 1].cell.value_counts().items(): Hf[c % 10, c // 10] = n
fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), sharey=True)
ext = [-0.5, fb - 0.5, -0.5, len(ZETAS) - 0.5]
for a, H, tt, cm in ((ax[0], Hn, "정상 충격 수 (fit ≥ 0.7)", "Greys"), (ax[1], Hf, "고장 충격 수", "Reds")):
    a.imshow(np.log10(H + 1), origin="lower", aspect="auto", cmap=cm, extent=ext)
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            if H[r, c] > 0: a.text(c, r, int(H[r, c]), ha="center", va="center", fontsize=6, color="tab:blue" if (a is ax[0] and Hf[r, c] == 0 and H[r, c] >= N_MIN) else "k")
    a.set_title(tt + ("  (파랑 숫자 = 정상만 N≥%d 있는 칸)" % N_MIN if a is ax[0] else ""))
    a.set_xticks(range(fb)); a.set_xticklabels([f"{(FEDGE[c-1] if c>0 else 0)/1000:.1f}" for c in range(fb)], fontsize=6, rotation=90)
    a.set_xlabel("f 칸 시작 (kHz)")
ax[0].set_yticks(range(len(ZETAS))); ax[0].set_yticklabels([f"{z:g}" for z in ZETAS]); ax[0].set_ylabel("ζ")
fig.savefig(os.path.join(FIG, "fz_cells.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 2부: 앞쪽 대역(f < 2.7 kHz)에서 ζ 말고 정상/고장을 가르는 지표 (파일 단위, AUC)
low = G[G.f < 2700]
rows = []
for i, g in low.groupby("file"):
    rows.append({"file": i, "y": y[i], "중앙 ζ": g.zeta.median(), "중앙 fit": g.fit.median(), "중앙 높이(g)": g.amp.median(),
                 "최대 높이(g)": g.amp.max(), "앞쪽 충격 수": len(g), "앞쪽 충격 비율": len(g) / len(nf[i]), "RMS(g)": X[i].std(),
                 "높이/RMS": g.amp.median() / X[i].std(), "중앙 f(kHz)": g.f.median() / 1000})
Lw = pd.DataFrame(rows)
print("앞쪽 대역 보유 파일: 정상", int((Lw.y == 0).sum()), "고장", int((Lw.y == 1).sum()))
aucs = {c: roc_auc_score(Lw.y, Lw[c]) for c in Lw.columns if c not in ("file", "y")}
A = pd.Series(aucs).sort_values(key=lambda s: (s - 0.5).abs(), ascending=False); print(A.round(3).to_string())
A.round(3).to_csv(os.path.join(HERE, "low_band_auc.csv"), header=["AUC(고장이 크면 1)"], encoding="utf-8-sig")
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for a, c in zip(ax, list(A.index[:4])):
    for lab, col, off in ((0, "gray", 0), (1, "tab:red", 1)):
        v = Lw[Lw.y == lab][c].values; a.scatter(rng.uniform(-0.15, 0.15, len(v)) + off, v, s=14, color=col, alpha=0.6)
    for i in IDX_FP + IDX_FN:
        r = Lw[Lw.file == i]
        if len(r): a.scatter([0 if y[i] == 0 else 1], r[c], s=60, color="tab:blue" if y[i] == 0 else "darkred", zorder=5); a.annotate(names[i][-3:], (0 if y[i] == 0 else 1, r[c].values[0]), fontsize=7)
    a.set_xticks([0, 1]); a.set_xticklabels(["정상", "고장"]); a.set_title(f"{c} (AUC {aucs[c]:.2f})", fontsize=9)
    if c in ("최대 높이(g)", "중앙 높이(g)", "RMS(g)"): a.set_yscale("log")
fig.savefig(os.path.join(FIG, "low_band_indicators.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 합본 (휴대폰용)
import matplotlib.image as mpimg
imgs = [mpimg.imread(os.path.join(FIG, p))[..., :3] for p in ("fz_scatter.png", "fz_cells.png", "low_band_indicators.png")]
Wd = max(im.shape[1] for im in imgs)
big = np.concatenate([np.pad(im, ((0, 0), ((Wd - im.shape[1]) // 2, Wd - im.shape[1] - (Wd - im.shape[1]) // 2), (0, 0)), constant_values=1.0) for im in imgs])
plt.imsave(os.path.join(FIG, "fz_all.png"), big); print("저장:", FIG)
