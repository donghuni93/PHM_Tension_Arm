# %% [markdown]
# Q2 저주파 L 블록 후보 피처 분포 (사령탑 확인 요청, 11-21)
# 피처 (모두 라벨 무관 정의, 라벨은 색칠·비교에만):
#   r_LP  = 30 Hz 저역 신호의 자기반복 상관 최대값 (지연 0.3–1.2 s)          [2-13 정의 그대로]
#   hit   = 200 Hz 고역 8σ 충격 중 주기 T ± 3%에 짝이 있는 비율                [2-13 정의 그대로]
#   S     = r_LP × hit                                                           [2-13 정의 그대로]
#   HFL   = log10(200 Hz 이상 RMS / 20 Hz 이하 RMS)  하중 대비 고주파
#   SHP   = 0–200 Hz 모양 거리: 3–200 Hz 로그 PSD(8192점, detrend 끔)에서 그 파일 평균을 빼 크기를 없앤 뒤,
#           정상 중앙 모양과의 RMS 거리(정상은 자기 제외 중앙). 모양 곡선은 그림 2
#   LV    = log10(200 Hz 이상 RMS) 전체 크기. 피처가 크기에 끌려가는지 보려는 참고 값
# 정상 99% 선 = 정상 120개의 99 백분위. 기준 파일 목록은 result.txt
# 실행: conda run -n qe-phm python run.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; D = 50; fs2 = FS / D
NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
Xt, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
Xv, vn = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
X = np.vstack([Xt, Xv]); N = len(Xt); allnames = names + vn
X = X - X.mean(1, keepdims=True)
grp = np.array(["정상"] * 0 + ["x"] * 0)
lab = np.concatenate([y, np.full(len(Xv), 2)])      # 0 정상, 1 고장, 2 val
nor = lab == 0

# %% 2-13 정의
HP = sosfiltfilt(butter(4, 200, "high", fs=FS, output="sos"), X, axis=1)
L30 = sosfiltfilt(butter(4, 30, "low", fs=FS, output="sos"), X, axis=1)[:, ::D]
L20 = sosfiltfilt(butter(4, 20, "low", fs=FS, output="sos"), X, axis=1)

def pear_r(s, k): return np.corrcoef(s[:len(s) - k], s[k:])[0, 1]

def events(h, k_sigma=8, merge_s=0.010):
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.flatnonzero(np.abs(h) > k_sigma * sig); ev = []
    for j in idx:
        if not ev or j - ev[-1][-1] > int(merge_s * FS): ev.append([j])
        else: ev[-1].append(j)
    return np.array([e[np.argmax(np.abs(h[e]))] for e in ev]) / FS if ev else np.array([])

ks = np.arange(int(0.3 * fs2), int(1.2 * fs2) + 1, 2)
rows = []
for i in range(len(X)):
    r = np.array([pear_r(L30[i], k) for k in ks]); T = ks[r.argmax()] / fs2
    ev = events(HP[i])
    if len(ev) >= 2:
        d = np.abs(ev[None, :] - ev[:, None]); hit = (np.abs(d - T) <= 0.03 * T).any(1).mean()
    else: hit = 0.0
    rows.append((r.max(), T, hit, len(ev)))
F = pd.DataFrame(rows, columns=["r_LP", "T", "hit", "n_ev"], index=allnames)
F["S"] = F.r_LP * F.hit
rms = lambda a: np.sqrt((a ** 2).mean(1))
F["HFL"] = np.log10(rms(HP) / rms(L20)); F["LV"] = np.log10(rms(HP))

# %% 0–200 Hz 모양
f, P = welch(X, fs=FS, nperseg=8192, noverlap=4096, window="hann", detrend=False, axis=-1)
m = (f >= 3) & (f <= 200); fm = f[m]
LP_ = np.log10(P[:, m] + 1e-20); LP_ = LP_ - LP_.mean(1, keepdims=True)    # 크기 제거
ref = np.median(LP_[nor], 0)
shp = np.zeros(len(X))
for i in range(len(X)):
    if nor[i]:
        idx = np.flatnonzero(nor); idx = idx[idx != i]; r_ = np.median(LP_[idx], 0)
    else: r_ = ref
    shp[i] = np.sqrt(((LP_[i] - r_) ** 2).mean())
F["SHP"] = shp
F["lab"] = lab; F.to_csv(os.path.join(HERE, "features.csv"))

# %% 숫자
FEATS = [("S", "S = r_LP × hit"), ("r_LP", "r_LP 느린 파형 반복"), ("hit", "hit 충격 주기 잠김"),
         ("HFL", "HFL log(고역/저역 RMS)"), ("SHP", "SHP 0–200 Hz 모양 거리"), ("LV", "LV 고역 RMS(크기, 참고)")]
out = ["정상 99% 선 = 정상 120개 99 백분위. AUC는 사후 관찰(고장 vs 정상), 판단에 쓰지 않는다", ""]
thr = {}
for c, t in FEATS:
    v = F[c].values; th = np.percentile(v[nor], 99); thr[c] = th
    auc = roc_auc_score(lab[lab < 2], v[lab < 2])
    ab = [allnames[i] for i in np.flatnonzero((lab == 1) & (v > th))]
    vv = [allnames[i] for i in np.flatnonzero((lab == 2) & (v > th))]
    nn = [allnames[i] for i in np.flatnonzero(nor & (v > th))]
    rho, _ = spearmanr(v, F.LV.values)
    out += [f"[{c}] {t}", f"  정상 중앙 {np.median(v[nor]):.3f} · 95% {np.percentile(v[nor],95):.3f} · 99% {th:.3f} · 최대 {v[nor].max():.3f} | 고장 중앙 {np.median(v[lab==1]):.3f} | val 중앙 {np.median(v[lab==2]):.3f}",
            f"  AUC(고장 vs 정상) {auc:.2f} | 크기 LV와 Spearman {rho:+.2f}",
            f"  99% 선 초과: 고장 {len(ab)}/30 {ab}", f"             val {len(vv)}/40 {vv}", f"             정상 {len(nn)}/120 {nn}", ""]
out.append("T (자기반복 주기, s) 중앙: 정상 %.2f · 고장 %.2f · val %.2f" % tuple(F["T"][lab == k].median() for k in (0, 1, 2)))
out.append("abn_022: " + ", ".join(f"{c} {F.loc['abn_022', c]:.3f}" for c, _ in FEATS))
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))

# %% 그림 1: 피처 6개 분포 (띠 + 정상 99% 선 + 초과 파일 이름)
rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, (c, t) in zip(axs.ravel(), FEATS):
    v = F[c].values
    ax.axhspan(np.percentile(v[nor], 5), np.percentile(v[nor], 95), color=NOR, alpha=.15, label="정상 5–95%")
    ax.axhline(thr[c], color="k", ls="--", lw=1, label="정상 99%")
    for k, col, mk, lbl in ((0, NOR, "o", "정상"), (1, ABN, "o", "고장"), (2, VAL, "^", "val")):
        s = lab == k; xs = k + rng.uniform(-.17, .17, s.sum())
        ax.scatter(xs, v[s], s=16 if k == 0 else 34, c=col, marker=mk, alpha=.6 if k == 0 else .95, label=lbl)
        if k > 0:
            for x_, val_, n_ in zip(xs, v[s], np.array(allnames)[s]):
                if val_ > thr[c]: ax.annotate(n_[-3:], (x_, val_), fontsize=6.5, color=col, xytext=(2, 2), textcoords="offset points")
    ax.set_xticks([0, 1, 2], ["정상 120", "고장 30", "val 40"]); ax.set_title(t, fontsize=10); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Q2 저주파 L 후보 피처 분포 (점선 = 정상 99%, 번호 = 선 위 파일)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_features.png"), dpi=90); plt.close(fig)

# %% 그림 2: 0–200 Hz 모양 곡선 (정상 중앙 기준)
fig, ax = plt.subplots(figsize=(11, 5))
D_ = LP_ - ref
ax.fill_between(fm, np.percentile(D_[nor], 5, 0), np.percentile(D_[nor], 95, 0), color=NOR, alpha=.18, label="정상 5–95%")
ax.plot(fm, np.percentile(D_[nor], 99, 0), color=NOR, lw=.8, ls="--"); ax.plot(fm, np.percentile(D_[nor], 1, 0), color=NOR, lw=.8, ls="--", label="정상 1·99%")
ax.plot(fm, np.median(D_[lab == 1], 0), color=ABN, lw=1.8, label="고장 중앙"); ax.plot(fm, np.median(D_[lab == 2], 0), color=VAL, lw=1.8, ls="--", label="val 중앙")
top = F.index[np.argsort(-F.SHP.values)][:4]
for n in top:
    i = allnames.index(n); ax.plot(fm, D_[i], lw=.9, alpha=.8, label=f"{n} (SHP {F.SHP[n]:.2f})")
ax.axhline(0, color="k", lw=.6); ax.set_xticks(np.arange(0, 201, 10)); ax.set_xlim(3, 200)
ax.set_xlabel("주파수 (Hz)"); ax.set_ylabel("모양 - 정상 중앙 모양 (dec)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
ax.set_title("0–200 Hz 정규화 로그 PSD 모양 (파일 평균 제거, Welch 8192점) — SHP 상위 4개 파일 겹침")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q2_shape200.png"), dpi=90); plt.close(fig)
print("done")
