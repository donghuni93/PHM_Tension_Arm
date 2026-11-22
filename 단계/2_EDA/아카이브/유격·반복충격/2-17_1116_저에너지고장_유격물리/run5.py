# %% [markdown]
# 2-17 5차 — 접촉 시간(충격의 날카로움)
# 4차 그림: 고장 충격은 한 점 스파이크가 포락선 위로 솟는 날카로운 모양, 정상의 큰 사건은 수 ms 둥근 덩어리
# 물리: 단단한 금속 접촉은 접촉 시간이 짧다(Hertz) → 날카로운 충격. 고무·노면 경로는 접촉이 길다 → 둥근 덩어리
#   울림(구조 응답, 2-16)이 아니라 충격이 "들어오는" 모양을 본다. 2-6·D8(타격 지속)은 울림까지 포함해 실패
# 정의 (라벨 보기 전에 고정). 충격 = 2·3차와 같음(가린 σ, |h| > kσ, 10 ms 안은 한 사건), 큰 순 상위 m개
#   CR  국소 파고율   = |h(p)| / RMS(h, p ± w)            (w = 2 ms) — 짧은 충격일수록 큼 (순수 임펄스면 √(창 표본 수))
#   RT  반상승 시간   = 포락선(0.1 ms 평균)이 봉우리의 50%를 마지막으로 넘은 때 → 봉우리 (ms) — 짧을수록 날카로움
#   SE  스파이크/포락선 = |h(p)| / 포락선(0.5 ms 평균)(p)   — 한 점이 주변 평균보다 얼마나 솟나
#   파일 값 = 상위 m개 충격의 중앙값. 충격 0개 → NaN
# 주의: 25 kHz 표본(0.04 ms)과 센서·장착 응답이 금속 접촉(수십 µs)보다 느릴 수 있다 → 측정되는 것은 "접촉 + 전달 경로"의 날카로움
# 라벨 없는 확인: 날카로움이 충격 크기(|h|/σ)와 따로 움직이는가 (같이 움직이면 K1을 다시 잰 것)
# 견고성: w ∈ {1, 2, 4} ms, m ∈ {3, 5, 10}, k ∈ {6, 8, 12}
# 실행: conda run -n qe-phm python "단계/2_EDA/아카이브/유격·반복충격/2-17_1116_저에너지고장_유격물리/run5.py"

# %% 설정
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.features import feature_table_knock
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs5"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = np.array([os.path.splitext(os.path.basename(str(n)))[0] for n in names])
lab = lambda s: s.replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

GAP = int(0.010 * FS)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
E0 = np.abs(hilbert(H, axis=-1))
mov = lambda a, n: np.convolve(a, np.ones(n) / n, "same")
E01 = np.array([mov(e, 3) for e in E0])      # 0.1 ms
E05 = np.array([mov(e, 12) for e in E0])     # 0.5 ms
K1 = feature_table_knock(X).K1_second_snr.values

def detect(h, k):
    s0 = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    m = np.ones(len(h), bool)
    for p in np.flatnonzero(np.abs(h) > k * s0):
        m[max(p - GAP, 0):p + GAP] = False
    hb = h[m] if m.sum() > len(h) // 4 else h
    sig = 1.4826 * np.median(np.abs(hb - np.median(hb))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > k * sig)
    if not len(idx):
        return sig, np.array([], int)
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > GAP) + 1)
    pk = np.array([g[np.argmax(np.abs(h[g]))] for g in gs], int)
    return sig, pk[np.argsort(-np.abs(h[pk]))]          # 큰 순

def sharp(n, p, w):
    h = H[n]; W = int(w * 1e-3 * FS)
    a, b = max(p - W, 0), min(p + W + 1, len(h))
    cr = abs(h[p]) / (np.sqrt(np.mean(h[a:b] ** 2)) + 1e-12)
    e = E01[n]; q = a + int(np.argmax(e[a:b]))          # 포락선 봉우리 (창 안)
    lo = max(q - int(0.005 * FS), 0)
    below = np.flatnonzero(e[lo:q] < 0.5 * e[q])
    rt = (q - (lo + below[-1])) / FS * 1e3 if len(below) else 5.0
    se = abs(h[p]) / (E05[n][p] + 1e-12)
    return cr, rt, se

def run(k=8, w=2, m=5, keep=False):
    rows, imp = [], []
    for n in range(N):
        sig, pk = detect(H[n], k)
        vals = [sharp(n, p, w) for p in pk[:m]]
        if keep:
            imp += [dict(file=short[n], y=int(y[n]), rank=r, snr=abs(H[n][p]) / sig, CR=v[0], RT=v[1], SE=v[2])
                    for r, (p, v) in enumerate(zip(pk[:m], vals))]
        v = np.array(vals) if vals else np.full((1, 3), np.nan)
        rows.append(dict(file=short[n], y=int(y[n]), n_imp=len(pk), CR=np.median(v[:, 0]), RT=np.median(v[:, 1]),
                         SE=np.median(v[:, 2]), sigma_mg=sig * 1e3))
    return pd.DataFrame(rows), pd.DataFrame(imp)

def auc(v, yy=None, flip=False):
    yy = y if yy is None else yy
    v = np.asarray(v, float); v = -v if flip else v
    fill = np.nanmin(v) - 1 if np.isfinite(np.nanmin(v)) else 0
    return roc_auc_score(yy, np.where(np.isnan(v), fill, v))

# %% 기본
D, I = run(keep=True)
D["K1"] = K1
QUIET = np.zeros(N, bool); QUIET[np.argsort(D.sigma_mg.values)[:40]] = True; D["quiet"] = QUIET
D.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")
I.to_csv(os.path.join(FIG, "impacts.csv"), index=False, encoding="utf-8-sig")
say("== 기본 (k=8σ, w=±2 ms, 상위 5 충격 중앙값) ==")
for g, gn in [(0, "정상"), (1, "고장")]:
    d = D[D.y == g]
    say(f"{gn}: CR 중앙 {d.CR.median():.2f} · RT 중앙 {d.RT.median():.3f} ms · SE 중앙 {d.SE.median():.2f} (충격 0개 {int(d.CR.isna().sum())})")
say(f"AUC 150개: CR {auc(D.CR):.3f} · RT(짧을수록 고장) {auc(D.RT, flip=True):.3f} · SE {auc(D.SE):.3f} · K1 {auc(K1):.3f}")
q = QUIET
say(f"AUC 조용한40: CR {auc(D.CR[q], y[q]):.3f} · RT {auc(D.RT[q], y[q], True):.3f} · SE {auc(D.SE[q], y[q]):.3f} · K1 {auc(K1[q], y[q]):.3f}")
# 충격 0개 파일을 빼고 (날카로움은 충격이 있어야 정의됨)
h = D.CR.notna().values
say(f"AUC 충격 있는 파일만 ({h.sum()}): CR {auc(D.CR[h], y[h]):.3f} · RT {auc(D.RT[h], y[h], True):.3f} · SE {auc(D.SE[h], y[h]):.3f} · K1 {auc(K1[h], y[h]):.3f}")

say("\n== 라벨 없는 확인: 충격 단위로 날카로움 vs 크기(|h|/σ) 순위상관 ==")
for c in ["CR", "RT", "SE"]:
    say(f"  {c}: 전체 {spearmanr(I[c], I.snr)[0]:.2f} · 정상 {spearmanr(I[c][I.y == 0], I.snr[I.y == 0])[0]:.2f} · 고장 {spearmanr(I[c][I.y == 1], I.snr[I.y == 1])[0]:.2f}")
say(f"  파일 단위 K1과 순위상관: CR {spearmanr(D.CR, K1, nan_policy='omit')[0]:.2f} · RT {spearmanr(D.RT, K1, nan_policy='omit')[0]:.2f} · SE {spearmanr(D.SE, K1, nan_policy='omit')[0]:.2f}")
# 크기를 맞춘 비교: 20–60σ 충격만
s = (I.snr >= 20) & (I.snr <= 60)
say(f"  크기 20–60σ 충격만: 정상 {int((s & (I.y == 0)).sum())} · 고장 {int((s & (I.y == 1)).sum())} | CR 중앙 정상 {I.CR[s & (I.y == 0)].median():.2f} · 고장 {I.CR[s & (I.y == 1)].median():.2f}"
    f" | RT 정상 {I.RT[s & (I.y == 0)].median():.3f} · 고장 {I.RT[s & (I.y == 1)].median():.3f} | SE 정상 {I.SE[s & (I.y == 0)].median():.2f} · 고장 {I.SE[s & (I.y == 1)].median():.2f}")

say("\n== 저에너지 고장 + 기준 정상 분위 ==")
for c in ["CR", "RT", "SE"]:
    nv = D[D.y == 0][c]
    say(f"  정상 {c}: 5% {nv.quantile(.05):.3f} · 50% {nv.median():.3f} · 95% {nv.quantile(.95):.3f}")
say(D[QUIET & (y == 1)].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
say("\n== 관심 파일 ==")
say(D[D.file.isin(["abn_014", "abn_017", "abn_022", "abn_002", "abn_005", "nor_009", "nor_054", "nor_086", "nor_025", "nor_055", "nor_116", "nor_059"])]
    .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% 견고성
rows = []
for k, w, m in itertools.product([6, 8, 12], [1, 2, 4], [3, 5, 10]):
    d, _ = run(k, w, m)
    rows.append(dict(k=k, w_ms=w, m=m, auc_CR=auc(d.CR), auc_RT=auc(d.RT, flip=True), auc_SE=auc(d.SE)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(FIG, "robust.csv"), index=False, encoding="utf-8-sig")
say("\n== 견고성 27가지 (AUC) ==")
say(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
say(f"범위: CR {R.auc_CR.min():.3f}–{R.auc_CR.max():.3f} · RT {R.auc_RT.min():.3f}–{R.auc_RT.max():.3f} · SE {R.auc_SE.min():.3f}–{R.auc_SE.max():.3f}")

# %% 그림 1: 충격 크기 vs 날카로움
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
for ax, c, yl in zip(axs, ["CR", "RT", "SE"], ["국소 파고율 CR", "반상승 시간 RT (ms)", "스파이크/포락선 SE"]):
    for g, col in [(0, NOR), (1, ABN)]:
        d = I[I.y == g]; ax.scatter(d.snr, d[c], s=8, c=col, alpha=.4, label="정상" if g == 0 else "고장")
    ax.set_xscale("log"); ax.set_xlabel("충격 크기 |h|/σ"); ax.set_ylabel(yl); ax.legend()
    if c == "RT": ax.set_yscale("log")
fig.suptitle("충격 단위 (파일마다 큰 순 5개) — 날카로움이 크기와 따로 움직이는가")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_크기대날카로움.png"), dpi=80); plt.close(fig)

# %% 그림 2: 가장 큰 충격 모양 겹쳐 보기 (봉우리 정렬, -2..+4 ms, 최대 1로)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
tt = np.arange(-50, 101) / FS * 1e3
for ax, g, col in [(axs[0], 0, NOR), (axs[1], 1, ABN)]:
    for n in np.flatnonzero(y == g):
        sig, pk = detect(H[n], 8)
        if not len(pk) or pk[0] < 50 or pk[0] + 101 > H.shape[1]: continue
        e = E01[n][pk[0] - 50:pk[0] + 101]; ax.plot(tt, e / e.max(), color=col, lw=.6, alpha=.35)
    ax.set_title(f"{'정상' if g == 0 else '고장'}: 가장 큰 충격의 포락선(0.1 ms) — 봉우리 정렬"); ax.set_xlabel("ms")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_큰충격모양.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
