# %% [markdown]
# 2-15 (11-23) 대역 경계 지도 — 경계를 어디에 두면 정상 · 고장이 잘 갈리나 (EDA, 모델 실험 없음)
# 질문 1: 경계를 바꿔야 하는가? CV 성능이 아니라 "구분이 잘 되는가"(분리도)로 본다.
# 방법: 대역 [아래, 위]를 격자로 모두 훑어 피처 4종의 분리도 지도를 그리고, 옥타브 · 팀원 · 모드 경계를 지도 위에 찍는다.
#   A 울림 비율   = 파일 안 충격 중 울림 주파수가 대역 안인 비율 (H7–H10 방식, 충격 없음 = −1)
#   B 충격 직후 몫 = 큰 충격 20개의 직후 12 ms 에너지 중 대역 몫, log10 중앙값 (소린 방식의 분자 한쪽)
#   C 대역 첨도   = 대역통과 신호의 log10 Pearson 첨도 (H3–H6 방식)
#   D 전체 몫     = 파일 전체 Welch 파워 중 대역 몫(1 kHz 위 합 기준), log10 (H12–H15 방식)
# 분리도 = |AUC − 0.5| × 2 (학습 150, 이상치 정상 포함). 붓스트랩 300번으로 '가장 좋은 대역'이 흔들리는 범위를 본다.
# 실행: conda run -n qe-phm python run.py (첫 실행은 대역 첨도 계산 몇 분, 이후 figs/*.npy 캐시)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal, stats
from src.exp2util import NS
from src.exp import OUTLIER_NORMALS
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
INK, MUTED = "#1d2430", "#5b6575"
FS = 25000; ns = NS(); y = ns.y; names = list(ns.names); n = len(y)
E14 = os.path.join(_d, "3_단계", "2_EDA", "2-14_1123_울림모드경계", "figs")   # 2-14 충격별 캐시 (학습 + val)

# 비교할 경계 묶음
SETS = {
    "옥타브 (지금)": ([(1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500)], "s", "#ffffff"),
    "팀원 (효섭 · 소린)": ([(1000, 2500), (2000, 5000), (2500, 5000), (5000, 8000), (8000, 11000), (11000, 12500)], "^", "#f2c14e"),
    "울림 모드 (2-14)": ([(1000, 1600), (1600, 4500), (2200, 4500), (4500, 5500), (5500, 8000), (8000, 12500)], "o", "#e8443a"),
}

# %% 빠른 분리도: 순위합 AUC (열마다), NaN 열은 NaN
def auc_cols(V, yy):
    R = stats.rankdata(V, axis=0); n1 = yy.sum(); n0 = len(yy) - n1
    return (R[yy == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
sep_cols = lambda V, yy: np.abs(auc_cols(V, yy) - .5) * 2

# %% 격자
def grid(step, lo0=1000, hi0=12500):
    e = np.arange(lo0, hi0 + 1, step); return [(a, b) for i, a in enumerate(e) for b in e[i + 1:]], e

# %% A · B: 2-14 충격 캐시에서 (학습 파일만)
D = pd.read_csv(os.path.join(E14, "impacts.csv")); SP = np.load(os.path.join(E14, "impact_spec250.npy"))
EB = np.arange(1000, 12501, 250)[:-1]
tr = D.file.isin(names).values; D, SP = D[tr].reset_index(drop=True), SP[tr]
fid = D.file.map({f: i for i, f in enumerate(names)}).values
BA, eA = grid(250)
VA = np.full((n, len(BA)), -1.0)                       # 충격 없음 = −1
cnt = np.bincount(fid, minlength=n)
for j, (a, b) in enumerate(BA):
    m = ((D.f50 >= a) & (D.f50 < b)).values
    k = np.bincount(fid, weights=m, minlength=n); VA[cnt > 0, j] = k[cnt > 0] / cnt[cnt > 0]
top = D.assign(i=np.arange(len(D))).sort_values("hg", ascending=False).groupby("file").head(20)
ti, tf = top.i.values, fid[top.i.values]
CS = np.cumsum(np.c_[np.zeros(len(SP)), SP], 1); tot = CS[:, -1]
VB = np.full((n, len(BA)), np.nan)
for j, (a, b) in enumerate(BA):
    sh = (CS[ti, (b - 1000) // 250] - CS[ti, (a - 1000) // 250]) / tot[ti]
    s = pd.Series(np.log10(sh + 1e-6)).groupby(tf).median(); VB[s.index.values, j] = s.values
for j in range(len(BA)):                                 # 충격 없음 = 정상 중앙값
    c = VB[:, j]; c[np.isnan(c)] = np.nanmedian(c[(y == 0) & ~np.isnan(c)])

# %% C · D: 신호에서 (500 Hz 격자, 캐시)
BC, eC = grid(500)
KC = os.path.join(FIG, "kurt500.npy")
if not os.path.exists(KC):
    VC = np.zeros((n, len(BC)))
    for j, (a, b) in enumerate(BC):
        sos = signal.butter(4, [a, min(b, 12499)], "bandpass", fs=FS, output="sos")
        xb = signal.sosfiltfilt(sos, ns.X - ns.X.mean(1, keepdims=True), axis=1)[:, 500:-500]
        VC[:, j] = np.log10(stats.kurtosis(xb, axis=1, fisher=False))
    np.save(KC, VC)
VC = np.load(KC)
f, P = signal.welch(ns.X - ns.X.mean(1, keepdims=True), FS, nperseg=4096, axis=1)
Pc = np.cumsum(np.c_[np.zeros(n), P], 1); idx = lambda hz: np.searchsorted(f, hz)
VD = np.stack([np.log10((Pc[:, idx(b)] - Pc[:, idx(a)]) / (Pc[:, idx(12500)] - Pc[:, idx(1000)])) for a, b in BC], 1)

np.savez(os.path.join(FIG, "격자피처.npz"), VA=VA, VB=VB, VC=VC, VD=VD, BA=np.array(BA), BC=np.array(BC), cnt=cnt, names=np.array(names))   # 4-14가 씀
FEATS = {"A 울림 비율 (충격이 어느 대역으로 울렸나)": (VA, BA, eA), "B 충격 직후 12 ms 에너지 몫 (소린 방식)": (VB, BA, eA),
         "C 대역 첨도 (충격의 뾰족함)": (VC, BC, eC), "D 파일 전체 에너지 몫": (VD, BC, eC)}

# %% 분리도 · 붓스트랩 (층화, 300번)
rng = np.random.default_rng(0); I0, I1 = np.where(y == 0)[0], np.where(y == 1)[0]
BS = [np.r_[rng.choice(I0, len(I0)), rng.choice(I1, len(I1))] for _ in range(300)]
RES, ROWS = {}, []
for name, (V, B, e) in FEATS.items():
    s = sep_cols(V, y)
    boot = np.stack([sep_cols(V[b], y[b]) for b in BS])          # (300, 대역)
    best = boot.argmax(1); RES[name] = (s, boot, best)
    j0 = s.argmax(); lo5, hi95 = np.percentile(boot[:, j0], [5, 95])
    say(f"\n[{name}] 최고 대역 {B[j0][0]}–{B[j0][1]} Hz 분리도 {s[j0]:.3f} (90% {lo5:.2f}–{hi95:.2f})")
    bb = pd.Series([B[k] for k in best]).value_counts().head(5)
    say("  붓스트랩 최고 대역 상위: " + " · ".join(f"{a}–{b} {c / 3:.0f}%" for (a, b), c in bb.items()))
    for setn, (bands, _, _) in SETS.items():
        for a, b in bands:
            if (a, b) not in B:   # 격자 밖이면 가장 가까운 격자
                a2, b2 = e[np.abs(e - a).argmin()], e[np.abs(e - b).argmin()]
            else: a2, b2 = a, b
            j = B.index((a2, b2)); lo5, hi95 = np.percentile(boot[:, j], [5, 95])
            # 그 대역이 붓스트랩 최고 대역과 다를 확률이 아니라, 최고 대역 대비 얼마나 낮은지
            gap = np.percentile(boot.max(1) - boot[:, j], 50)
            ROWS.append(dict(피처=name[0], 묶음=setn, 대역=f"{a}–{b}", 격자=f"{a2}–{b2}", 분리도=round(s[j], 3), 구간90=f"{lo5:.2f}–{hi95:.2f}", 최고대비=round(gap, 3)))
T = pd.DataFrame(ROWS); say("\n경계 묶음별 분리도 (최고대비 = 붓스트랩마다 '그 회차 최고 대역 − 이 대역'의 중앙값)"); say(T.to_string(index=False))
T.to_csv(os.path.join(FIG, "경계별_분리도.csv"), index=False, encoding="utf-8-sig")

# %% 그림 1: 분리도 지도 4장 (가로 = 아래 경계, 세로 = 위 경계)
fig, axs = plt.subplots(2, 2, figsize=(15, 13))
for ax, (name, (V, B, e)) in zip(axs.ravel(), FEATS.items()):
    s, boot, best = RES[name]; M = np.full((len(e), len(e)), np.nan); ix = {v: i for i, v in enumerate(e)}
    for j, (a, b) in enumerate(B): M[ix[b], ix[a]] = s[j]
    st = e[1] - e[0]
    im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=.85, extent=(e[0] - st / 2, e[-1] + st / 2, e[0] - st / 2, e[-1] + st / 2), aspect="auto")
    bx = np.array([B[k] for k in best]); ax.scatter(bx[:, 0] + rng.normal(0, st * .15, len(bx)), bx[:, 1] + rng.normal(0, st * .15, len(bx)), s=6, color="#ff66cc", alpha=.35, lw=0)
    for setn, (bands, mk, c) in SETS.items():
        ax.scatter([a for a, _ in bands], [b for _, b in bands], marker=mk, s=70, facecolor="none", edgecolor=c, lw=1.8, label=setn)
    j0 = s.argmax(); ax.scatter(*B[j0], marker="*", s=220, color="#ff66cc", edgecolor=INK, lw=.8, label=f"최고 {B[j0][0]}–{B[j0][1]} ({s[j0]:.2f})")
    ax.set_title(name, loc="left", fontsize=11); ax.set_xlabel("아래 경계 (Hz)", color=MUTED); ax.set_ylabel("위 경계 (Hz)", color=MUTED)
    ax.legend(fontsize=8, loc="lower right", framealpha=.85)
fig.colorbar(im, ax=axs, shrink=.6, label="분리도 |AUC-0.5|×2")
fig.suptitle("그림 1 — 대역 [아래, 위]마다 분리도. 분홍 점 = 붓스트랩 300번의 최고 대역(흩어진 범위 = 경계의 불확실성), 기호 = 경계 묶음", x=.01, ha="left", fontsize=12)
fig.savefig(os.path.join(FIG, "1_분리도지도.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
OUT.close()
