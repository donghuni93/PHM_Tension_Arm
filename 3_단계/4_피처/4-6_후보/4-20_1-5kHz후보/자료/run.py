# %% 4-20 1–5 kHz(연결부 대역) 후보 C1–C4 EDA (피처후보 P1–P3 + 충격 대 배경 비, 11-25)
# 정의 = src.features2.band15_table (라벨 보기 전에 고정). 충격 = 최종 모델과 같은 all_impacts(fmax=11000)
# 관문(계산 전 고정): AUC 95 % CI 하한 > 0.6 · 기존 J 열과 max |ρ| < 0.8 · val 이동 AUC가 고장 0–40 % 섞음 기준 범위 안
#   · 오경보 정상 059 065 116 095 110이 정상 99 % 위로 가지 않음 · 반쪽(앞 1초 vs 뒤 1초) Spearman ρ(정상) ≥ 0.5
# 결과는 하위 폴더 C1–C4마다 figs/수치.txt · figs/files.csv · 그림 1장, 요약은 figs/관문.csv
import os, sys
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.exp2util import NS
from src.impacts import all_impacts
from src.features2 import band15_table, B15_COLS
from src.final import J_FIXED, _PeakRef
FS = 25000; CACHE = os.path.join(ROOT, "4_outputs", "cache")
ns = NS(); y = ns.y; names = np.array(ns.names + [f"val_{v[-3:]}" for v in ns.nv])
X = np.vstack([ns.X, ns.Xv]); grp = np.array(["normal" if v == 0 else "abnormal" for v in y] + ["val"] * len(ns.nv))
N, A, V = grp == "normal", grp == "abnormal", grp == "val"; TR = N | A

# %% 피처 (전체 · 반쪽), 캐시 4_outputs/cache/b15_*.csv
pc = os.path.join(CACHE, "b15_all.csv")
if os.path.exists(pc):
    F = pd.read_csv(pc, index_col=0); F1 = pd.read_csv(os.path.join(CACHE, "b15_h1.csv"), index_col=0); F2 = pd.read_csv(os.path.join(CACHE, "b15_h2.csv"), index_col=0)
else:
    idx = [im["idx"] for im in all_impacts(X, fmax=11000)]
    F = band15_table(X, idx).set_axis(names); h = X.shape[1] // 2
    F1 = band15_table(X[:, :h], [i[i < h] for i in idx]).set_axis(names)
    F2 = band15_table(X[:, h:], [i[i >= h] - h for i in idx]).set_axis(names)
    F["n_imp"] = [len(i) for i in idx]
    for d, k in ((F, "all"), (F1, "h1"), (F2, "h2")): d.to_csv(os.path.join(CACHE, f"b15_{k}.csv"))

# J 입력 열(최종 모델과 같은 fmax=11000 표, EXP-065 캐시): H 13열 + 봉우리 2열(학습 정상 기준) + L1_lock
T = np.load(os.path.join(CACHE, "jtab_train_11k_065.npz"))
J = pd.DataFrame(T["H"], columns=J_FIXED)
pk = _PeakRef(float(T["df"])).fit(T["L"][y == 0]).transform(T["L"]); J["H16_peak_h"], J["H17_peak_w"] = pk[:, 0], pk[:, 1]; J["L1_lock"] = T["S"]
load = sosfiltfilt(butter(4, 20, "lowpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=1).std(1)

rng = np.random.default_rng(0)
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
def ci(a, b, B=2000):
    s = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]; return np.percentile(s, [2.5, 97.5])
def mix_range(v, frac, B=1000):
    """학습 정상 + 고장 섞음(frac)으로 val 크기(40)를 재추출했을 때 '재추출 vs 정상' AUC의 90 % 범위 (4-19와 같음)"""
    n, a = v[N], v[A]; k = int(round(40 * frac)); out = []
    for _ in range(B):
        s = np.r_[rng.choice(n, 40 - k), rng.choice(a, k)] if k else rng.choice(n, 40); out.append(auc(s, n))
    return np.percentile(out, [5, 95])

GROUPS = {"C1_대역포락선첨도대비": ["C1_kenv_1.6-4.5k", "C1_kc_1.6-4.5k", "C1_kc_2-4k", "H4_kurt_2-4k"],
          "C2_TKEO파고율": ["C2_tk_1.6-4.5k", "C2_tk_2-4k"],
          "C3_WPD상대에너지": ["C3_w1_all", "C3_w2_all", "C3_w1_imp", "C3_w2_imp"],
          "C4_충격대배경비": ["C4_ib_1.6-4.5k", "C4_ib_2-4k"]}
HARD = ["abn_021", "abn_028", "abn_014", "abn_022", "abn_002", "abn_017", "nor_059", "nor_065", "nor_116", "nor_025", "nor_095", "nor_110", "nor_046"]
FA = ["nor_059", "nor_065", "nor_116", "nor_095", "nor_110"]
pos = {f: i for i, f in enumerate(names)}
ROWS = []
for c in B15_COLS:
    v = F[c].values; n, a = v[N], v[A]; lo, hi = ci(a, n); q99 = np.percentile(n, 99); q10a = np.percentile(a, 10)
    rv = auc(v[V], n); m = [mix_range(v, f) for f in (0, .2, .4)]; env = (min(r[0] for r in m), max(r[1] for r in m))
    r = {j: spearmanr(v[TR], J[j].values)[0] for j in J.columns}; jb = max(r, key=lambda k: abs(r[k]))
    fa_up = [f for f in FA if v[pos[f]] > q99]; hs = spearmanr(F1[c].values[N], F2[c].values[N])[0]
    ok = dict(CI=lo > .6, 중복=abs(r[jb]) < .8, val이동=env[0] <= rv <= env[1], 오경보정상=not fa_up, 반쪽=hs >= .5)
    ROWS.append(dict(열=c, AUC=round(auc(a, n), 3), CI_lo=round(lo, 3), CI_hi=round(hi, 3), 고장_정상99초과=" ".join(x[-3:] for x in names[A & (v > q99)]) or "-",
                     n고장_정상99초과=int((A & (v > q99)).sum()), 정상_고장10초과=f"{int((N & (v > q10a)).sum())}/120",
                     val_AUC=round(rv, 3), 기준0=f"{m[0][0]:.2f}–{m[0][1]:.2f}", 기준20=f"{m[1][0]:.2f}–{m[1][1]:.2f}", 기준40=f"{m[2][0]:.2f}–{m[2][1]:.2f}",
                     rho_하중=round(spearmanr(n, load[N])[0], 3), 최대중복=f"{jb} {r[jb]:+.2f}", 최대중복_abs=round(abs(r[jb]), 3),
                     오경보정상_99초과=" ".join(f[-3:] for f in fa_up) or "-", 반쪽rho=round(hs, 3),
                     val_정상99초과=" ".join(x[-3:] for x in names[V & (v > q99)]) or "-",
                     **{f"관문_{k}": bool(t) for k, t in ok.items()}, 통과=all(ok.values())))
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
G = pd.DataFrame(ROWS).set_index("열"); G.to_csv(os.path.join(HERE, "figs", "관문.csv"), encoding="utf-8-sig")
P = pd.DataFrame({c: [(F[c].values[N] < F.loc[f, c]).mean() * 100 for f in HARD] for c in B15_COLS}, index=HARD).round(0)
RHO = pd.DataFrame(spearmanr(F.loc[TR, B15_COLS].values)[0], index=B15_COLS, columns=B15_COLS).round(2)
RHO.to_csv(os.path.join(HERE, "figs", "후보끼리_rho.csv"), encoding="utf-8-sig")
noimp = f"충격 없는 파일: 정상 {int((F.n_imp[N] == 0).sum())} · 고장 {int((F.n_imp[A] == 0).sum())} · val {int((F.n_imp[V] == 0).sum())}"

# %% 하위 폴더마다 수치 · 파일별 · 그림
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"; C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}
LAB = {"normal": "정상 120", "abnormal": "고장 30", "val": "val 40"}
for gname, cols in GROUPS.items():
    D = os.path.join(HERE, gname, "figs"); os.makedirs(D, exist_ok=True)
    L = [f"{gname} — 관문 표 (학습 150 · val 40, 충격 = all_impacts(fmax=11000))", noimp, "",
         G.loc[cols].T.to_string(), "", "헷갈리는 파일의 정상 내 백분위", P[cols].to_string(), "",
         "후보끼리 Spearman ρ (학습 150)", RHO.loc[cols, B15_COLS].to_string()]
    open(os.path.join(D, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
    F[cols + ["n_imp"]].assign(group=grp, **{f"{c}_앞1초": F1[c] for c in cols}, **{f"{c}_뒤1초": F2[c] for c in cols}).to_csv(os.path.join(D, "files.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.8), facecolor=SURF)
    for a_, c in zip(np.atleast_1d(ax), cols):
        a_.set_facecolor(SURF); a_.grid(True, axis="y", color=GRID); a_.set_axisbelow(True)
        for s in ("top", "right"): a_.spines[s].set_visible(False)
        for k, g in enumerate(("normal", "abnormal", "val")):
            v = F[c][grp == g].values
            a_.scatter(k + rng.uniform(-.25, .25, len(v)), v, s=13, color=C[g], alpha=.6, lw=0)
            a_.hlines(np.median(v), k - .32, k + .32, color=INK, lw=2)
        for f in ["abn_021", "abn_028", "nor_059", "nor_065", "nor_116"]:
            k = 1 if f.startswith("abn") else 0
            a_.annotate(f[-3:], (k, F.loc[f, c]), xytext=(k + .36, F.loc[f, c]), fontsize=8, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
        a_.axhline(np.percentile(F[c][N], 99), color=C["normal"], ls="--", lw=1)
        a_.set_xticks(range(3)); a_.set_xticklabels([LAB[g] for g in ("normal", "abnormal", "val")], fontsize=8.5)
        r = G.loc[c]; a_.set_title(f"{c}\nAUC {r.AUC:.2f} [{r.CI_lo:.2f}, {r.CI_hi:.2f}] · val {r.val_AUC:.2f} · {'통과' if r.통과 else '미통과'}", loc="left", color=INK, fontsize=9.5)
    fig.text(.01, .01, "검은 선 = 중앙값, 파란 점선 = 정상 99 %. 이름 = 고장 021 · 028, 오경보 정상 059 · 065 · 116", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(os.path.join(D, "분포.png"), dpi=130, facecolor=SURF); plt.close(fig)
open(os.path.join(HERE, "figs", "관문.txt"), "w", encoding="utf-8").write(noimp + "\n\n" + G.to_string() + "\n\n헷갈리는 파일 백분위\n" + P.to_string() + "\n\n후보끼리 ρ\n" + RHO.to_string())
print("done")
