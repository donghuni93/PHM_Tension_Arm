# %% [markdown]
# V-1 val 미탐 원인 진단 (11-27)
# - 질문: val F2 0.513(TP 8 · 오경보 2 · 미탐 9, 고장 17/40)에서 미탐이 왜 많은가. 가설 "train이 쉬워서 ET가 피처 몇 개만 썼다"
# - 입력: 제출 모델 EXP-062 구성 3(제출_EXP-062/모델/final_model_062.joblib), train 150, val 40. 학습하지 않는다
# - 그룹(둘다고장 · 8조만 · 우리만 · 둘다정상)은 다른 조 결과로 추론한 것 → 진단에만 쓰고 모델 튜닝에 쓰지 않는다
# - 출력: 자료/result.txt (README 표의 근거)
#   conda run -n qe-phm python 5_VAL_단계/V-1_미탐원인진단/자료/run.py

# %% 설정
import os, sys, io
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from src.data import load_dataset, load_folder
from src.final import _j_et

SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
m = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")); core = m.core_
cache = os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib")      # j_tables(train, val) — V-2도 같이 쓴다
if os.path.exists(cache):
    Ttr, Tva, y, ntr, nva = joblib.load(cache)
else:
    Xtr, y, ntr = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
    Xva, nva = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
    Ttr, Tva = m._tables(Xtr), m._tables(Xva)
    joblib.dump((Ttr, Tva, y, ntr, nva), cache)

cols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); Eva, _ = core._et_input(Tva)
out = io.StringIO(); P = lambda *a: print(*a, file=out)
GROUPS = {"둘다고장": ["004", "013", "016", "027", "031", "035", "039"], "8조만": ["010", "034", "036"], "우리만": ["017", "024", "025"]}
tag = {f"val_{k}": g for g, ks in GROUPS.items() for k in ks}
grp = np.array([tag.get(n, "둘다정상") for n in nva])
nm = y == 0

# %% A. 단일 피처 AUC · ET 중요도(MDI)
P("== A. train 단일 피처 AUC (정상 120 vs 고장 30) · ET 중요도(MDI) ==")
imp = core.et_.feature_importances_
rows = []
for j, c in enumerate(cols):
    a = roc_auc_score(y, Etr[:, j]); rows.append((c, max(a, 1 - a), "+" if a >= .5 else "-", imp[j]))
A = pd.DataFrame(rows, columns=["feat", "AUC", "dir", "MDI"]).sort_values("MDI", ascending=False)
P(A.to_string(index=False, float_format="%.3f"))
P(f"MDI 상위 3개 합 = {A.MDI.head(3).sum():.2f}, 상위 5개 합 = {A.MDI.head(5).sum():.2f}")
order = list(A.feat)

# %% B. 얕은 나무로 train 난이도
P("\n== B. train 난이도: 깊이 1·2 나무 (5-fold OOF) ==")
for d in (1, 2):
    pr = cross_val_predict(DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0), Etr, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    pred = pr >= .5; tp = (pred & (y == 1)).sum(); fp = (pred & nm).sum(); fn = (~pred & (y == 1)).sum()
    P(f"깊이 {d}: 재현율 {tp}/30, 오경보 {fp}/120, F2 {5*tp/(5*tp+4*fn+fp):.3f}, AUC {roc_auc_score(y, pr):.3f}")

# %% C. 상위 k개 피처만 쓴 ET
P("\n== C. ET 피처 수를 줄였을 때 (train 4-fold OOF AUC) ==")
for k in (1, 2, 3, 5, len(cols)):
    idx = [cols.index(c) for c in order[:k]]; oof = np.empty(len(y))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(Etr, y):
        oof[b] = _j_et(0).fit(Etr[a][:, idx], y[a]).predict_proba(Etr[b][:, idx])[:, 1]
    P(f"상위 {k:2d}개: OOF AUC {roc_auc_score(y, oof):.3f}")

# %% D. val 피처 위치 (train 정상 중앙 · MAD 기준 z), 그룹 중앙값
med = np.median(Etr[nm], 0); mad = 1.4826 * np.median(np.abs(Etr[nm] - med), 0)
mad[mad < 1e-9] = Etr[nm].std(0)[mad < 1e-9] + 1e-9
Ztr, Zva = (Etr - med) / mad, (Eva - med) / mad
P("\n== D. val 피처 위치 (train 정상 중앙·MAD 기준 z) — 그룹 중앙값 ==")
tab = pd.DataFrame({"feat": cols, "train정상": np.median(Ztr[nm], 0), "train고장": np.median(Ztr[~nm], 0)})
for g in ["둘다고장", "8조만", "우리만", "둘다정상"]:
    tab[g] = np.median(Zva[grp == g], 0)
P(tab.set_index("feat").loc[order].to_string(float_format="%+.2f"))

# %% E. ET logit 구간별 파일 수 (train OOF vs val)
P("\n== E. ET logit 구간별 파일 수 ==")
oof = core.Z_[:, 0]
p = np.clip(core.et_.predict_proba(Eva)[:, 1], 1 / 600, 1 - 1 / 600); lv = np.log(p / (1 - p))
for lo, hi in [(-9, -2), (-2, -1), (-1, 0), (0, 1), (1, 9)]:
    k = (oof >= lo) & (oof < hi)
    P(f"  [{lo:+d},{hi:+d}) train 정상 {int((k & nm).sum()):3d} · 고장 {int((k & ~nm).sum()):2d} | val {int(((lv >= lo) & (lv < hi)).sum())}")

# %% F. 두 징후(H3 > 3 · H8 > 5) 동시 출현
i3, i8 = cols.index("H3_kurt_1-2k"), cols.index("H8_ring_2-4k")
P("\n== F. 두 징후 H3_kurt_1-2k z > 3 · H8_ring_2-4k z > 5 ==")
for nmz, ZZ, msk in [("train정상", Ztr, nm), ("train고장", Ztr, ~nm), ("val전체", Zva, np.ones(len(Zva), bool))]:
    a = ZZ[msk, i3] > 3; b = ZZ[msk, i8] > 5
    P(f"{nmz:8s} n={msk.sum():3d}: 둘 다 {(a & b).mean():.0%} · 하나만 {(a ^ b).mean():.0%} · H3만 {(a & ~b).sum()} · H8만 {(~a & b).sum()}")
P("val 하나만:", [n for n, a, b in zip(nva, Zva[:, i3] > 3, Zva[:, i8] > 5) if a ^ b])
P("val 둘 다:", [n for n, a, b in zip(nva, Zva[:, i3] > 3, Zva[:, i8] > 5) if a and b])

# %% G. val 파일별 표 (결합 점수 순)
ev = pd.read_csv(os.path.join(SUB, "출력", "근거표_submission_val_9조.csv"))
F = pd.DataFrame({"file": nva, "grp": grp, "ET_p": p, "H3z": Zva[:, i3], "H8z": Zva[:, i8],
                  "H16z": Zva[:, cols.index("H16_peak_h")], "score": ev.score.values, "ae_z": ev.ae_z.values, "n_imp": ev.n_imp.values})
P("\n== G. val 파일별 (결합 점수 순) ==")
P(F.sort_values("score", ascending=False).to_string(index=False, float_format="%+.2f"))

open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write(out.getvalue())
print("done")
