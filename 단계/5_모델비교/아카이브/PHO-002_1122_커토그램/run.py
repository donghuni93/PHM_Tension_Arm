# %% [markdown]
# PHO-002 (휴대폰 세션, 11-22) 커토그램: 파일마다 충격이 가장 뾰족한 대역을 자동으로 찾아 그 대역 포락선으로 충격·박자를 다시 계산
# - 커토그램(라벨 없음, 실행 전 고정): 대역폭 4000·2000·1000·500·250 Hz, 중심은 폭의 절반 간격, 대역 범위 1–12.4 kHz
#   대역마다 4차 Butterworth 대역통과(양방향) → 복소 포락선 c = hilbert → 첨도 K = E|c|⁴ / (E|c|²)² − 2 (Antoni 2007, 가우스 잡음 = 0)
#   최적 대역 = K 최대. 보조: 2초를 0.5초 4조각으로 나눈 K의 중앙값(큰 충격 하나에 끌리는지 확인용)
# - 그 대역 신호로 충격(src.impacts.detect_impacts, 새 정의 그대로) → 충격 수 · 박자 S(src.features2.lock_S) 재계산
# - quick_cv (기준 W = PHO-001과 같음), 하나씩: K1 = + [K_max, log2(fc)] 2열 / K2 = 박자 S를 커토그램 대역 S로 교체
# 실행: conda run -n qe-phm python "단계/5_모델비교/PHO-002_1122_커토그램/run.py"

# %% 설정
import os, sys, json, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from src.exp import load_train
from src.features2 import feat2_table, lock_S
from src.impacts import detect_impacts
from src.models2 import Two
from src.evaluate import quick_cv, separation
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000; CACHE = os.path.join(_d, "outputs", "cache")
LOG = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; n = len(y); g = np.arange(n); nor = y == 0

# %% 커토그램
BANDS = [(fc - bw / 2, fc + bw / 2, fc, bw) for bw in (4000, 2000, 1000, 500, 250)
         for fc in np.arange(1000 + bw / 2, 12400 - bw / 2 + 1, bw / 2)]
SOS = [butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos") for lo, hi, _, _ in BANDS]

def kurt_env(c):
    a = np.abs(c) ** 2; return float(np.mean(a ** 2) / np.mean(a) ** 2 - 2)

def kgram(x):
    x = x - x.mean(); K, Kmed = [], []
    for s in SOS:
        c = hilbert(sosfiltfilt(s, x)); K.append(kurt_env(c))
        Kmed.append(np.median([kurt_env(p) for p in np.array_split(c, 4)]))
    return np.array(K), np.array(Kmed)

t0 = time.time()
p = os.path.join(CACHE, "pho002_kgram.npz")
if os.path.exists(p):
    z = np.load(p); KG, KM = z["K"], z["Km"]
else:
    out = Parallel(n_jobs=-1)(delayed(kgram)(x) for x in X); KG, KM = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    np.savez(p, K=KG, Km=KM)
B = np.array(BANDS); best = KG.argmax(1); bestm = KM.argmax(1)
fc, bw, Kmax = B[best, 2], B[best, 3], KG.max(1)
say(f"커토그램 {len(BANDS)}대역 × {n}파일 · {time.time() - t0:.0f}s")

# %% 최적 대역 신호로 충격 · 박자 S 다시 계산
def redo(i):
    x = X[i]; lo, hi = B[best[i], 0], B[best[i], 1]
    xb = sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos"), x - x.mean())
    t = detect_impacts(x, _hp=xb)["t"]
    return len(t), lock_S(x, t)[0]
RD = np.array(Parallel(n_jobs=-1)(delayed(redo)(i) for i in range(n)))
z = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T = z["T"].item(); L = z["L"]; fL = z["f"]
n_old, S_old = np.expm1(np.asarray(T["H1_nimp"], float)), np.asarray(T["L1_S"], float)
n_new, S_new = RD[:, 0], RD[:, 1]

# %% EDA 요약
say("\n=== 최적 대역 (정상 / 고장) ===")
for lab, m in (("정상", nor), ("고장", ~nor)):
    say(f"{lab}: 중심 중앙 {np.median(fc[m]) / 1000:.2f} kHz · 폭 중앙 {np.median(bw[m]):.0f} Hz · K_max 중앙 {np.median(Kmax[m]):.1f} · "
        f"중심 ≥ 8 kHz 비율 {(fc[m] >= 8000).mean():.2f} · 4조각 중앙값 방식과 같은 대역 {(best[m] == bestm[m]).mean():.2f}")
auc = lambda v: roc_auc_score(y, v)
say(f"단독 AUC: K_max {auc(Kmax):.2f} · log2(fc) {auc(np.log2(fc)):.2f} · 충격 수(기존) {auc(n_old):.2f} → (커토그램) {auc(n_new):.2f} · 박자 S(기존) {auc(S_old):.2f} → (커토그램) {auc(S_new):.2f}")
pct = lambda v, i: (v[nor] < v[i]).mean() * 100
say("\n파일 | 최적 대역 | K_max (정상 중 %) | 충격 수 기존 → 커토그램 | S 기존 → 커토그램")
FOC = ["abn_002", "abn_014", "abn_017", "abn_022", "abn_015", "abn_020", "nor_116", "nor_065", "nor_059", "nor_025", "nor_027"]
for fn in FOC:
    i = names.index(fn)
    say(f"{fn} | {fc[i] / 1000:5.2f} kHz ± {bw[i] / 2:4.0f} | {Kmax[i]:7.1f} ({pct(Kmax, i):3.0f}%) | {int(n_old[i]):2d} → {int(n_new[i]):2d} | {S_old[i]:.3f} → {S_new[i]:.3f}")
say(f"정상 S 최대: 기존 {S_old[nor].max():.3f} → 커토그램 {S_new[nor].max():.3f}")

# %% 그림 1: 최적 대역 지도 (중심 × K_max)
fig, ax = plt.subplots(figsize=(10, 6))
for m, c, lab in ((nor, "#2a78d6", "정상"), (~nor, "#eb6834", "고장")):
    ax.scatter(fc[m] / 1000, Kmax[m], s=np.sqrt(bw[m]) * 1.5, color=c, alpha=.6, label=lab, edgecolor="k", lw=.3)
for fn in FOC:
    i = names.index(fn); ax.annotate(fn.replace("abn_", "A").replace("nor_", "N"), (fc[i] / 1000, Kmax[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("최적 대역 중심 (kHz)"); ax.set_ylabel("K_max (포락선 첨도, log)"); ax.grid(alpha=.3); ax.legend()
ax.set_title("파일별 커토그램 최적 대역 (점 크기 = 대역폭)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "best_band.png"), dpi=120); plt.close(fig)

# %% 그림 2: 커토그램 지도 예시
def kmap(ax, i):
    for li, bwv in enumerate((4000, 2000, 1000, 500, 250)):
        sel = np.flatnonzero(B[:, 3] == bwv)
        for j in sel:
            ax.add_patch(plt.Rectangle((B[j, 0] / 1000, li), (B[j, 1] - B[j, 0]) / 1000 / 2, 1, color=plt.cm.magma(min(KG[i, j] / max(KG[i].max(), 1e-9), 1))))
    ax.set_xlim(1, 12.4); ax.set_ylim(0, 5); ax.set_yticks(np.arange(5) + .5); ax.set_yticklabels(["4000", "2000", "1000", "500", "250"]); ax.invert_yaxis()
    ax.set_title(f"{names[i]} · 최적 {fc[i] / 1000:.2f} kHz ± {bw[i] / 2:.0f} · K {Kmax[i]:.1f}", fontsize=10); ax.set_xlabel("주파수 (kHz)")
ex = ["nor_001", "nor_116", "nor_059", "abn_002", "abn_014", "abn_022"]
fig, axs = plt.subplots(2, 3, figsize=(16, 7))
for ax, fn in zip(axs.ravel(), ex):
    kmap(ax, names.index(fn))
axs[0, 0].set_ylabel("대역폭 (Hz)"); axs[1, 0].set_ylabel("대역폭 (Hz)")
fig.suptitle("커토그램 (색 = 그 파일의 K / K_max, 밝을수록 뾰족)"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "kurtogram_examples.png"), dpi=110); plt.close(fig)

# %% quick_cv (하나씩)
H = np.column_stack([np.asarray(T[c], float) for c in FIXED]); df = float(fL[1] - fL[0]); nf = len(FIXED)
mk = lambda fx, k: (lambda: Two("lin", (), fixed=tuple(fx), nf=k, df=df, peak=tuple(PEAK)))
R = {}
def run(tag, F, make):
    q = quick_cv(F, y, g, make); oof = q["oof"]; sp = separation(oof, y, names)
    R[tag] = dict(pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"],
                  rank={k: q["miss_rank"][names.index(k)] for k in ("abn_002", "abn_014", "abn_017", "abn_022")})
    say(f"{tag:12s} PR-AUC {R[tag]['pr_auc']:.3f} ± {R[tag]['pr_sd']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | "
        + " ".join(f"{k[-3:]} {v:.2f}" for k, v in R[tag]["rank"].items()))
say("\n=== quick_cv (5-fold × seed 3, PR-AUC) ===")
run("기준 W", np.column_stack([H, S_old, L]), mk(FIXED, nf))
run("K1 +K·fc", np.column_stack([H, Kmax, np.log2(fc), S_old, L]), mk(FIXED + ["K_max", "K_fc"], nf + 2))
run("K2 S 교체", np.column_stack([H, S_new, L]), mk(FIXED, nf))
json.dump(dict(rule="quick_cv 5-fold × seed 3, PR-AUC. 하나씩", results=R,
               files={fn: dict(fc=float(fc[names.index(fn)]), bw=float(bw[names.index(fn)]), K=float(Kmax[names.index(fn)]),
                               n_old=int(n_old[names.index(fn)]), n_new=int(n_new[names.index(fn)]), S_old=float(S_old[names.index(fn)]), S_new=float(S_new[names.index(fn)])) for fn in FOC}),
          open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say(f"완료 {time.time() - t0:.0f}s")
