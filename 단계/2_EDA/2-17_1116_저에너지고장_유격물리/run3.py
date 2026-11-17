# %% [markdown]
# 2-17 3차 — K1(충격성)을 물리적으로 다듬고, 운전 조건(하중 무리)별 정상 기준으로 문턱을 잡는다
# 2차 결론: 위상 잠금은 이 센서로 안 보임, 충격/하중 비는 K1보다 약함, 바퀴 튐 무리에서 고장이 뚜렷이 갈림
# 3차 정의 (라벨 보기 전에 고정)
#   σ    = 200 Hz 고역 h에서 충격(|h| > kσ0) ±10 ms를 가린 바탕의 1.4826·MAD
#   충격 = |h| > kσ, 10 ms 안은 한 사건 (k = 8 기본)
#   다발 = 하중 주기 T0 = 1/f0의 1/d 안에 이어지는 충격 (d = 4 기본). f0 = 1–20 Hz 최대 주파수
#   K1b  = log10(두 번째로 큰 "다발"의 최대 |h| / σ)  — 서로 다른 하중 사건 두 번 이상 바탕 위로 튀어나온 타격
#          다발이 1개 이하면 log10(k) (문턱 바닥)
#   무리 = f0 < B Hz 차체 / ≥ B Hz 바퀴 튐 (B = 6 기본, 두 물리 대역 1–3 · 10–20 Hz 사이)
#   문턱 = 그 무리 정상의 분위수 (정상 자신은 빼고 계산 = leave-one-out, 오경보를 정직하게)
# 견고성: k ∈ {4, 8, 12}, d ∈ {2, 4, 8}, B ∈ {3, 6, 9} (각 ±50%) 27가지
# 실행: conda run -n qe-phm python "단계/2_EDA/2-17_1116_저에너지고장_유격물리/run3.py"

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
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.features import feature_table_knock
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs3"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
lab = lambda s: s.replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

GAP = int(0.010 * FS)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
sp = np.abs(np.fft.rfft(Xc * np.hanning(X.shape[1]), n=4 * X.shape[1], axis=1)) ** 2
fq = np.fft.rfftfreq(4 * X.shape[1], 1 / FS); bb = (fq >= 1) & (fq <= 20)
F0 = fq[bb][np.argmax(sp[:, bb], 1)]
del sp
KNOCK = feature_table_knock(X)
QUIET = np.zeros(N, bool)

def k1b(h, f0, k=8, d=4):
    s0 = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    m = np.ones(len(h), bool)
    for p in np.flatnonzero(np.abs(h) > k * s0):
        m[max(p - GAP, 0):p + GAP] = False
    hb = h[m] if m.sum() > len(h) // 4 else h
    sig = 1.4826 * np.median(np.abs(hb - np.median(hb))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > k * sig)
    if not len(idx):
        return sig, 0, float(np.log10(k))
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > GAP) + 1)
    pk = np.array([g[np.argmax(np.abs(h[g]))] for g in gs])
    bu = np.split(pk, np.flatnonzero(np.diff(pk) > int(FS / f0 / d)) + 1)
    top = np.sort([np.abs(h[b]).max() / sig for b in bu])[::-1]
    return sig, len(bu), float(np.log10(top[1])) if len(top) > 1 else float(np.log10(k))

def loo_thr(v, reg, q):
    """정상마다 자신을 뺀 같은 무리 정상의 q 분위수, 고장은 같은 무리 정상 전체의 q 분위수"""
    t = np.full(N, np.nan)
    for i in range(N):
        ref = (y == 0) & (reg == reg[i]); ref[i] = False
        t[i] = np.quantile(v[ref], q) if q < 1 else v[ref].max()
    return t

def run(k=8, d=4, B=6, verbose=False):
    r = [k1b(H[i], F0[i], k, d) for i in range(N)]
    sig = np.array([a[0] for a in r]); nb = np.array([a[1] for a in r]); v = np.array([a[2] for a in r])
    reg = (F0 >= B).astype(int)
    # 무리별 로버스트 z (정상 중앙값·MAD) — 무리끼리 같은 눈금으로
    z = np.empty(N)
    for g in (0, 1):
        ref = (y == 0) & (reg == g); med = np.median(v[ref]); mad = 1.4826 * np.median(np.abs(v[ref] - med)) + 1e-9
        z[reg == g] = (v[reg == g] - med) / mad
    out = dict(k=k, d=d, B=B, n_wheel_nor=int(((reg == 1) & (y == 0)).sum()), n_wheel_abn=int(((reg == 1) & (y == 1)).sum()),
               auc_K1b=roc_auc_score(y, v), auc_K1b_z=roc_auc_score(y, z))
    res = {}
    for q, qn in [(0.95, "q95"), (1.0, "max")]:
        tg = loo_thr(v, np.zeros(N, int), q)         # 무리 무시
        tr = loo_thr(v, reg, q)                       # 무리별
        for nm, t in [("전역", tg), ("무리별", tr)]:
            on = v > t
            out[f"{qn}_{nm}_TP"] = int((on & (y == 1)).sum()); out[f"{qn}_{nm}_FP"] = int((on & (y == 0)).sum())
            res[(qn, nm)] = on
    return out, v, z, reg, nb, sig, res

# %% 기본 설정
o, v, z, reg, nb, sig, res = run()
QUIET[np.argsort(sig)[:40]] = True
D = pd.DataFrame(dict(file=short, y=y, f0=F0, regime=np.where(reg, "바퀴", "차체"), sigma_mg=sig * 1e3, n_burst=nb,
                      K1b=v, K1b_z=z, K1=KNOCK.K1_second_snr.values, quiet=QUIET))
D.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")
say("== 기본 (k=8, d=4, B=6 Hz) ==")
say(f"바퀴 무리: 정상 {o['n_wheel_nor']} · 고장 {o['n_wheel_abn']}")
say(f"AUC  K1(현행) {roc_auc_score(y, D.K1):.3f} · K1b {o['auc_K1b']:.3f} · K1b 무리별 z {o['auc_K1b_z']:.3f}")
say(f"조용한 40 AUC  K1 {roc_auc_score(y[QUIET], D.K1[QUIET]):.3f} · K1b {roc_auc_score(y[QUIET], v[QUIET]):.3f} · z {roc_auc_score(y[QUIET], z[QUIET]):.3f}")
for qn in ["q95", "max"]:
    for nm in ["전역", "무리별"]:
        on = res[(qn, nm)]
        say(f"  문턱 {qn} {nm:3}: 고장 {o[f'{qn}_{nm}_TP']:2}/30 · 정상 {o[f'{qn}_{nm}_FP']:2}/120 | "
            f"고장 {' '.join(lab(s) for s in np.array(short)[on & (y == 1)])} | 정상 {' '.join(lab(s) for s in np.array(short)[on & (y == 0)])}")
say("(정상 오경보는 leave-one-out: 정상 자신을 뺀 정상으로 문턱. 기대 오경보 q95 ≈ 5%, max ≈ 1/n)")
say("\n참고 — 현행 유격 규칙이 켜는 고장: 003 004 008 010 011 014 017 019 028 030")
say("\n== 무리별 값 (z 큰 순, 상위 25) ==")
say(D.sort_values("K1b_z", ascending=False).head(25).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
say("\n== 바퀴 무리 전체 ==")
say(D[D.regime == "바퀴"].sort_values("K1b", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
say("\n== 저에너지 고장 (조용한 40 안의 고장) ==")
say(D[QUIET & (y == 1)].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# %% 견고성 27가지
rows = [run(k, d, B)[0] for k, d, B in itertools.product([4, 8, 12], [2, 4, 8], [3, 6, 9])]
R = pd.DataFrame(rows); R.to_csv(os.path.join(FIG, "robust.csv"), index=False, encoding="utf-8-sig")
say("\n== 견고성 (k 충격 배수, d 다발 = T0/d, B 무리 경계 Hz) ==")
say(R[["k", "d", "B", "n_wheel_nor", "n_wheel_abn", "auc_K1b", "auc_K1b_z", "max_전역_TP", "max_전역_FP", "max_무리별_TP", "max_무리별_FP",
       "q95_무리별_TP", "q95_무리별_FP"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# %% 그림: 무리별 K1b 분포
fig, axs = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(width_ratios=[3, 2]))
rng = np.random.default_rng(0)
for ax, (g, gn) in zip(axs, [(0, "차체 출렁임 무리 (f0 < 6 Hz)"), (1, "바퀴 튐 무리 (f0 ≥ 6 Hz)")]):
    for c, yy in [(NOR, 0), (ABN, 1)]:
        s = (reg == g) & (y == yy)
        xx = yy + rng.uniform(-.15, .15, s.sum())
        ax.scatter(xx, v[s], c=c, s=np.where(QUIET[s], 45, 15), edgecolors=np.where(QUIET[s], "k", "none"))
        for xi, vi, f in zip(xx, v[s], np.array(short)[s]):
            if yy == 1 or vi > np.quantile(v[(reg == g) & (y == 0)], .9):
                ax.annotate(lab(f), (xi, vi), fontsize=7)
    ref = v[(reg == g) & (y == 0)]
    ax.axhline(np.quantile(ref, .95), color="0.5", ls="--", label="정상 95%")
    ax.axhline(ref.max(), color="k", ls=":", label="정상 최대")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"정상 {int(((reg == g) & (y == 0)).sum())}", f"고장 {int(((reg == g) & (y == 1)).sum())}"])
    ax.set_ylabel("K1b = log10(두 번째 다발 최대 |h| / σ)"); ax.set_title(gn); ax.legend(fontsize=8)
fig.suptitle("K1b — 서로 다른 하중 사건 두 번 이상 바탕 위로 튀어나온 타격 (테두리 = 조용한 40)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_무리별_K1b.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
