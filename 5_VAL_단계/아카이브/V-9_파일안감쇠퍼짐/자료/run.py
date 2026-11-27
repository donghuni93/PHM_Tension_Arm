# %% [markdown]
# V-9 파일 안 울림 퍼짐 (11-27)
# - 가설(H5 변형): 헐거운 연결부는 충격마다 접촉 상태가 달라 같은 모드라도 주파수 · 감쇠가 충격마다 흔들린다.
#   측정 조건 차이(val)는 모든 충격을 같이 옮기므로 파일 안 퍼짐에는 안 나타난다 → 이동에 강한 피처
# - 피처 2개(계산 전에 고정, 예측 = 고장 > 정상. 반대면 쓰지 않는다)
#   Vd1 = 본체 띠(5.5–8 kHz, 2-14 라벨 없는 모드 지도) 충격들의 울림 주파수 IQR (Hz)
#   Vd2 = 같은 충격들의 ln ζ IQR
#   충격 = V-8 느슨 기준(fit ≥ 0.6). 본체 충격 3개 미만이면 0. 묶기 · 가중치 · 새 문턱 없음
# - 입력: V-8 캐시 4_outputs/cache/val_eda_ringall.csv (모든 충격의 f · ζ · fit, VP 추정)
#   conda run -n qe-phm python 5_VAL_단계/아카이브/V-9_파일안감쇠퍼짐/자료/run.py

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
B = A[(A.f >= 5500) & (A.f < 8000) & (A.fit >= 0.6)].copy(); B["lnz"] = np.log(B.z)
iqr = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))
def feats(n):
    d = B[B.file == n]
    if len(d) < 3:
        return dict(Vd1=0.0, Vd2=0.0, n본체=len(d))
    return dict(Vd1=iqr(d.f), Vd2=iqr(d.lnz), n본체=len(d))
Ftr = pd.DataFrame([feats(n) for n in ntr], index=ntr); Fva = pd.DataFrame([feats(n) for n in nva], index=nva)
pd.concat([Ftr.assign(그룹=["train 정상" if v == 0 else "train 고장" for v in y]), Fva.assign(그룹="val")]).to_csv(
    os.path.join(HERE, "파일별_Vd.csv"), encoding="utf-8-sig")

# %% 커버리지 · 관문
has_tr = Ftr.n본체 >= 3; has_va = Fva.n본체 >= 3
cov = {"train 정상": f"{int(has_tr[y == 0].sum())}/120", "train 고장": f"{int(has_tr[y == 1].sum())}/30", "val": f"{int(has_va.sum())}/40"}
cvr = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
def cvs(X):
    au, ap = [], []
    for a, b in cvr.split(X, y):
        p = LogisticRegression(C=1.0, class_weight="balanced").fit(X[a], y[a]).decision_function(X[b])
        au.append(roc_auc_score(y[b], p)); ap.append(average_precision_score(y[b], p))
    return np.mean(au), np.mean(ap)
b_au, b_ap = cvs(oof[:, None])
nmv = Ftr[(y == 0) & has_tr.values]
rows = []
for c in ["Vd1", "Vd2"]:
    v = Ftr[c].values; md, sc = np.median(nmv[c]), 1.4826 * np.median(np.abs(nmv[c] - np.median(nmv[c]))) + 1e-9
    rh = [abs(spearmanr(v, Etr[:, j]).correlation) for j in range(Etr.shape[1])]; j = int(np.argmax(rh))
    au, ap = cvs(np.column_stack([oof, v]))
    hv = has_tr.values
    rows.append(dict(피처=c, AUC_전체=round(roc_auc_score(y, v), 3), AUC_값있는파일=round(roc_auc_score(y[hv], v[hv]), 3) if len(set(y[hv])) == 2 else None,
                     정상_중앙=round(float(nmv[c].median()), 3), 고장_중앙=round(float(Ftr[(y == 1) & hv][c].median()), 3),
                     val_중앙=round(float(Fva[has_va][c].median()), 3), val중앙z=round(float((Fva[has_va][c].median() - md) / sc), 2),
                     H최대ρ=f"{max(rh):.2f} ({Hcols[j]})", ρ_n본체=round(spearmanr(v, Ftr.n본체).correlation, 2),
                     ΔAUC=round(au - b_au, 4), ΔPRAUC=round(ap - b_ap, 4)))
T = pd.DataFrame(rows)

# %% 그림: 값이 있는 파일만, 그룹별
COL = {"train 정상": "C0", "train 고장": "C1", "val": "C4"}; rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for a, c, lab in zip(axs, ["Vd1", "Vd2"], ["Vd1 본체 울림 주파수 IQR (Hz)", "Vd2 본체 ln ζ IQR"]):
    for k, (g, v) in enumerate([("train 정상", Ftr[(y == 0) & has_tr.values][c]), ("train 고장", Ftr[(y == 1) & has_tr.values][c]), ("val", Fva[has_va][c])]):
        a.scatter(k + rng.uniform(-.18, .18, len(v)), v, s=20, alpha=.65, color=COL[g]); a.text(k, a.get_ylim()[0], f"n={len(v)}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(3)); a.set_xticklabels(list(COL)); a.set_ylabel(lab); a.grid(alpha=.3, axis="y")
fig.suptitle("파일 안 울림 퍼짐 (본체 5.5–8 kHz, fit ≥ 0.6, 본체 충격 3개 이상인 파일만)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "파일안_퍼짐.png"), dpi=130); plt.close(fig)

with open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8") as fh:
    fh.write(f"본체 충격 3개 이상인 파일: {cov}\n\n기준: ET OOF만 로지스틱 AUC {b_au:.4f} · PR-AUC {b_ap:.4f}\n\n" + md_table(T) + "\n")
json.dump(dict(커버리지=cov, 표=rows), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done")
