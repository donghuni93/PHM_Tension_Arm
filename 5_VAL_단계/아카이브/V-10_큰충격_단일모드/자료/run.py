# %% [markdown]
# V-10 가장 큰 충격이 한 모드로 깨끗하게 우는가 (11-27)
# - 관찰(V-2 그림 4 · 옛 V-3): train 정상의 큰 6–7 kHz 강타는 여러 모드가 섞여 길게 울고 맥놀이(fit 0.46–0.68),
#   val 고립 강타 012 · 032 · 034는 한 모드로 깨끗(fit 0.86–0.96). 옛 V-3에서 ζ 차이는 착시였고 fit 차이만 남았다
# - 피처 하나(계산 전에 고정): VF1 = 파일에서 가장 큰 충격(우리 검출기 포락선/바탕 배수 최대)의 단일 모드 fit
#   (VP, 봉우리 + 0.2 ms부터 8 ms, V-8 캐시). 충격 없는 파일 = 0. 문턱 · 띠 없음
# - 미리 적은 예측: 고장 > 정상. 반대면 쓰지 않는다
#   conda run -n qe-phm python 5_VAL_단계/아카이브/V-10_큰충격_단일모드/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
import src.viz  # noqa: F401
import matplotlib.pyplot as plt
from src.exp import md_table
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
A = pd.read_csv(os.path.join(ROOT, "4_outputs", "cache", "val_eda_ringall.csv"), encoding="utf-8-sig")
Ttr, _, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
core = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")).core_
Etr, _ = core._et_input(Ttr); Hcols = list(core.cols) + list(core.peak); oof = core.Z_[:, 0]

# %% 피처
top = A.loc[A.groupby("file").hx.idxmax()].set_index("file")
def vf(n):
    return (float(top.loc[n, "fit"]), float(top.loc[n, "hx"]), float(top.loc[n, "f"])) if n in top.index else (0.0, 0.0, np.nan)
Ftr = pd.DataFrame([vf(n) for n in ntr], index=ntr, columns=["VF1", "hx", "f"]); Fva = pd.DataFrame([vf(n) for n in nva], index=nva, columns=["VF1", "hx", "f"])
v = Ftr.VF1.values

# %% 관문
auc = roc_auc_score(y, v)
rng = np.random.default_rng(0); boot = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if len(set(y[i])) == 2:
        boot.append(roc_auc_score(y[i], v[i]))
rh = [abs(spearmanr(v, Etr[:, j]).correlation) for j in range(Etr.shape[1])]; jm = int(np.argmax(rh))
cvr = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
def cvs(X):
    au, ap = [], []
    for a, b in cvr.split(X, y):
        p = LogisticRegression(C=1.0, class_weight="balanced").fit(X[a], y[a]).decision_function(X[b])
        au.append(roc_auc_score(y[b], p)); ap.append(average_precision_score(y[b], p))
    return np.mean(au), np.mean(ap)
b_au, b_ap = cvs(oof[:, None]); a_au, a_ap = cvs(np.column_stack([oof, v]))
nv = v[y == 0]; md = np.median(nv); sc = 1.4826 * np.median(np.abs(nv - md)) + 1e-9
z = lambda x: (x - md) / sc
has = (Ftr.hx > 0).values
gate = dict(AUC=round(auc, 3), AUC_부트_5_95=[round(float(np.percentile(boot, 5)), 3), round(float(np.percentile(boot, 95)), 3)],
            AUC_충격있는파일=round(roc_auc_score(y[has], v[has]), 3),
            H최대ρ=f"{max(rh):.2f} ({Hcols[jm]})", ρ_ET_OOF=round(spearmanr(v, oof).correlation, 2), ρ_최대충격크기=round(spearmanr(v, Ftr.hx).correlation, 2),
            ΔAUC=round(a_au - b_au, 4), ΔPRAUC=round(a_ap - b_ap, 4),
            정상_중앙=round(float(md), 3), 고장_중앙=round(float(np.median(v[y == 1])), 3), val_중앙=round(float(Fva.VF1.median()), 3),
            val중앙z=round(float(z(Fva.VF1.median())), 2))
SHOW = ["val_012", "val_032", "val_034", "val_036", "val_039", "val_016", "val_004", "nor_046", "nor_060", "nor_006", "nor_038", "nor_055", "nor_089", "abn_017"]
pos = pd.DataFrame([dict(file=n, VF1=round(float((Ftr if n in ntr else Fva).loc[n, "VF1"]), 2), z=round(float(z((Ftr if n in ntr else Fva).loc[n, "VF1"])), 1),
                         최대충격=round(float((Ftr if n in ntr else Fva).loc[n, "hx"]), 0), f=round(float((Ftr if n in ntr else Fva).loc[n, "f"]), 0)) for n in SHOW])
# 크기 구간별 정상 · 고장 fit (큰 충격일수록 섞이는가)
G = pd.DataFrame({"VF1": v, "hx": Ftr.hx, "y": y})
G["크기"] = pd.cut(G.hx, [0, 10, 30, 100, 1e9], labels=["<10", "10–30", "30–100", "≥100"])
by = G.groupby(["크기", "y"], observed=True).VF1.agg(["count", "median"]).round(2).reset_index()

# %% 그림
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
COL = {0: "C0", 1: "C1"}
for yy, lab in [(0, "train 정상"), (1, "train 고장")]:
    m = (y == yy) & has; axs[0].scatter(Ftr.hx[m], v[m], s=18, alpha=.6, color=COL[yy], label=lab)
axs[0].scatter(Fva.hx[Fva.hx > 0], Fva.VF1[Fva.hx > 0], s=18, alpha=.6, color="C4", label="val")
for n in SHOW[:4]:
    axs[0].annotate(n[4:], (Fva.loc[n, "hx"], Fva.loc[n, "VF1"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
axs[0].set_xscale("log"); axs[0].set_xlabel("가장 큰 충격 (배, 로그)"); axs[0].set_ylabel("그 충격의 단일 모드 fit (VF1)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
jit = np.random.default_rng(1); labs = []
for k, (lab, vals, c) in enumerate([("train 정상", v[(y == 0) & has], "C0"), ("train 고장", v[(y == 1) & has], "C1"), ("val", Fva.VF1[Fva.hx > 0].values, "C4")]):
    axs[1].scatter(k + jit.uniform(-.18, .18, len(vals)), vals, s=16, alpha=.6, color=c); labs.append(f"{lab}\n(n={len(vals)})")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(labs); axs[1].set_ylabel("VF1 (충격 있는 파일)"); axs[1].grid(alpha=.3, axis="y")
fig.suptitle(f"VF1 = 가장 큰 충격의 단일 모드 fit · train AUC {auc:.3f} (예측: 고장 > 정상)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "VF1.png"), dpi=130); plt.close(fig)

with open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8") as fh:
    fh.write("## 관문\n\n" + md_table(pd.DataFrame([gate])) + "\n\n## 파일별 위치 (z = train 정상 기준)\n\n" + md_table(pos) +
             "\n\n## 최대 충격 크기 구간별 VF1 (y=0 정상, 1 고장)\n\n" + md_table(by) + "\n")
json.dump(gate, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done")
