# %% [markdown]
# V-12 피처 묶음 교체 비교: 우리 H 15열 vs 효섭 13개 (11-27)
# - 질문: 우리 ET 피처가 val 고장을 못 잡는 것은 피처 묶음 탓인가? 피처를 근본적으로 바꾸면 "train에 없던 유형"을 잡나?
# - 방법: 모델은 고정(ET · RF, 300그루), 입력 피처 묶음만 바꾼다(우리 H15 · 효섭13 · 합집합 28) + 지도 없는 정상 기준 점수
#   A. train 그룹 CV OOF (AUC · PR-AUC)
#   B. 유형 빼기: 2-1 고장 유형(L1 · L2 · L3 · 기타) 하나를 학습에서 통째로 빼고, 그 유형이 정상보다 높게 나오나(AUC) → 새 유형 일반화
#   C. 피처 × 유형 AUC (단일 피처가 어느 유형을 가르나)
#   D. val 순위: train 전체로 학습 → val 40 점수. 그룹(추론)은 진단에만, 모델 · 문턱에 쓰지 않는다
# - 입력 캐시: val_eda_tabs.joblib(제출 모델 j_tables) · val_eda_hy.joblib(효섭 피처). 제출 모델 EXP-062 구성 3의 ET 입력을 그대로 씀
#   (울림 축소 p0 · 봉우리 기준은 제출 모델이 train 전체 정상으로 잡은 값 → CV에 약간 새어 들어감, 두 묶음 비교에는 영향 작음)
#   conda run -n qe-phm python 5_VAL_단계/V-12_피처묶음_교체비교/자료/run.py

# %% 설정
import os, sys, json, importlib.util
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf
from src.final import _j_et
from src.exp import md_table
import src.viz  # noqa: F401  (한글 글꼴)
import matplotlib.pyplot as plt

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2]

# 우리 H: 제출 모델 ET 입력
SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
core = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")).core_
Ttr, Tva, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
ntr, nva = [str(n) for n in ntr], [str(n) for n in nva]
Etr, _ = core._et_input(Ttr); Eva, _ = core._et_input(Tva)
OUR = list(core.cols) + list(core.peak)

# 효섭: 3-2와 같은 처리(AR 잔차 · 결측 = train 150 기준)
spec = importlib.util.spec_from_file_location("hy", os.path.join(ROOT, "1_참고자료", "팀원 공유 정보", "신효섭", "1119_최종", "7_코드", "최종모델.py"))
hy = importlib.util.module_from_spec(spec); spec.loader.exec_module(hy)
Xtr_h, ACtr, _, Xva_h, ACva, _, y2, ntr2, nva2 = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_hy.joblib"))
assert [str(n) for n in ntr2] == ntr and [str(n) for n in nva2] == nva and (np.asarray(y2) == y).all()
XA = pd.concat([Xtr_h, Xva_h], ignore_index=True); ACA = np.vstack([ACtr, ACva]); yA = np.r_[y, np.zeros(len(nva), int)]
XA = hy.prep(XA, ACA, yA, np.arange(len(y)))
HY = ["A_segk", "b25_5_segk", "D_segk", "I_density", "D_nimp", "SK_200_3k", "b25_5_kurt", "A_kurt",
      "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58", "R_811_over_25", "AR_resid"]
Htr, Hva = XA.iloc[:len(y)][HY].values, XA.iloc[len(y):][HY].values

SETS = {"우리 H15": (Etr, Eva, OUR), "효섭 13": (Htr, Hva, HY),
        "합집합 28": (np.hstack([Etr, Htr]), np.hstack([Eva, Hva]), OUR + HY)}
MODELS = {"ET": lambda s: _j_et(s),
          "RF": lambda s: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=s, n_jobs=1)}

# CV 그룹: 097 · 101 · 120 한 그룹, 나머지 파일당 1그룹
grp = np.arange(len(y)); grp[[ntr.index(n) for n in ("nor_097", "nor_101", "nor_120")]] = -1
nm = y == 0

# 2-1 고장 유형 (L4 1개 + 미분류 3개 = 기타)
ty = pd.read_csv(os.path.join(ROOT, "3_단계", "2_EDA", "2-1_고장유형분류", "자료", "figs", "types.csv"), index_col=0)["type"]
ftype = np.array([("기타" if ty[n] in ("L4", "U") else ty[n]) if y[i] == 1 else "정상" for i, n in enumerate(ntr)])
TYPES = ["L1", "L2", "L3", "기타"]
TYPE_NOTE = {"L1": "국소 · 고주파 (에너지 정상 수준)", "L2": "출렁임 + 가려진 충격", "L3": "금속 타격 (연발 · 강타)", "기타": "L4 저주파 1 + 미분류 3"}

def maha(Xa, Xb):
    """지도 없음: 학습 정상만으로 강건 z → Ledoit-Wolf 마할라노비스 거리"""
    med = np.median(Xa, 0); mad = 1.4826 * np.median(np.abs(Xa - med), 0); mad[mad < 1e-9] = Xa.std(0)[mad < 1e-9] + 1e-9
    Za, Zb = (Xa - med) / mad, (Xb - med) / mad
    Za, Zb = np.clip(Za, -20, 20), np.clip(Zb, -20, 20)
    lw = LedoitWolf().fit(Za); d = Zb - lw.location_
    return np.einsum("ij,jk,ik->i", d, lw.precision_, d)

# %% A · B. train OOF (기본) · 유형 빼기
def oof_run(X, mk, drop_type=None, seed=0):
    """그룹 5-fold. drop_type이면 그 유형 고장을 학습에서 빼고, 그 유형 고장은 모든 폴드 모델 점수의 평균으로 채점"""
    s = np.zeros(len(y)); cnt = np.zeros(len(y))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grp):
        if drop_type is not None:
            a = a[ftype[a] != drop_type]
            b = np.r_[b[ftype[b] != drop_type], np.where(ftype == drop_type)[0]]
        if mk == "MAHA":
            p = maha(X[a][y[a] == 0], X[b])
        else:
            p = MODELS[mk](seed).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        s[b] += p; cnt[b] += 1
    return s / cnt

def type_auc(s):
    """유형별 AUC = 그 유형 고장 vs 정상 120"""
    return {t: roc_auc_score(np.r_[np.zeros(nm.sum()), np.ones((ftype == t).sum())], np.r_[s[nm], s[ftype == t]]) for t in TYPES}

rowsA, rowsB, oof_keep = [], [], {}
for sn, (Xt, _, _) in SETS.items():
    for mk in list(MODELS) + ["MAHA"]:
        base = [oof_run(Xt, mk, None, sd) for sd in SEEDS]
        auc = [roc_auc_score(y, s) for s in base]; ap = [average_precision_score(y, s) for s in base]
        rowsA.append(dict(묶음=sn, 모델=mk, AUC=np.mean(auc), AUC_sd=np.std(auc), PR_AUC=np.mean(ap), PR_AUC_sd=np.std(ap)))
        oof_keep[(sn, mk)] = np.mean(base, 0)
        bt = pd.DataFrame([type_auc(s) for s in base]).mean()
        for t in TYPES:
            held = [type_auc(oof_run(Xt, mk, t, sd))[t] for sd in SEEDS] if mk != "MAHA" else [bt[t]]
            rowsB.append(dict(묶음=sn, 모델=mk, 유형=t, n=int((ftype == t).sum()), 같이학습=bt[t], 빼고학습=np.mean(held)))
A = pd.DataFrame(rowsA); B = pd.DataFrame(rowsB); B["손실"] = B["같이학습"] - B["빼고학습"]

# %% C. 단일 피처 × 유형 AUC (방향은 train 전체 AUC로 정함)
rowsC = []
for sn in ("우리 H15", "효섭 13"):
    Xt, _, cols = SETS[sn]
    for j, c in enumerate(cols):
        a = roc_auc_score(y, Xt[:, j]); sgn = 1 if a >= .5 else -1
        r = dict(묶음=sn, 피처=c, 전체=max(a, 1 - a), 방향="+" if sgn > 0 else "-")
        r.update({t: roc_auc_score(np.r_[np.zeros(nm.sum()), np.ones((ftype == t).sum())], sgn * np.r_[Xt[nm, j], Xt[ftype == t, j]]) for t in TYPES})
        rowsC.append(r)
C = pd.DataFrame(rowsC)

# %% D. val: train 전체로 학습 → val 40 점수 (seed 평균), 그룹은 추론 · 진단용
VG = {"둘다고장(확정 추론)": ["004", "013", "016", "027", "031", "035", "039"], "024·025 중 하나": ["024", "025"],
      "017 (오경보 추론)": ["017"], "놓친 후보 9": ["001", "009", "010", "012", "029", "032", "034", "036", "038"]}
vtag = {f"val_{k}": g for g, ks in VG.items() for k in ks}
vgrp = np.array([vtag.get(n, "나머지 21") for n in nva])
rowsD, vscore = [], {}
for sn, (Xt, Xv, _) in SETS.items():
    for mk in list(MODELS) + ["MAHA"]:
        if mk == "MAHA":
            p = maha(Xt[nm], Xv); pt = oof_keep[(sn, mk)]
        else:
            p = np.mean([MODELS[mk](sd).fit(Xt, y).predict_proba(Xv)[:, 1] for sd in SEEDS], 0); pt = oof_keep[(sn, mk)]
        # train 정상 OOF 점수 안 백분위 (모델 사이 눈금을 맞추려고)
        pct = np.array([(pt[nm] < v).mean() * 100 for v in p])
        vscore[(sn, mk)] = pct
        rk = pd.Series(p).rank(ascending=False).values
        cand = vgrp == "놓친 후보 9"; rest = vgrp == "나머지 21"; sure = vgrp == "둘다고장(확정 추론)"
        rowsD.append(dict(묶음=sn, 모델=mk,
                          후보9_순위중앙=float(np.median(rk[cand])), 후보9_상위17안=int((rk[cand] <= 17).sum()),
                          확정7_상위17안=int((rk[sure] <= 17).sum()), 후보9_vs_나머지21_AUC=roc_auc_score(np.r_[np.zeros(rest.sum()), np.ones(cand.sum())], np.r_[p[rest], p[cand]]),
                          후보9_정상99위=int((pct[cand] >= 99).sum()), 나머지21_정상99위=int((pct[rest] >= 99).sum()),
                          val_정상99위_전체=int((pct >= 99).sum())))
D = pd.DataFrame(rowsD)
Dfile = pd.DataFrame({"file": nva, "그룹": vgrp, **{f"{sn}|{mk}": np.round(v, 0) for (sn, mk), v in vscore.items()}})

# %% 그림 1: 유형 빼기 (ET · MAHA)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
xs = np.arange(len(TYPES)); w = .26
for ax, mk in zip(axs, ["ET", "RF", "MAHA"]):
    for k, (sn, col) in enumerate(zip(SETS, ["#2a6fdb", "#e07b1a", "#5a5a5a"])):
        b = B[(B.묶음 == sn) & (B.모델 == mk)].set_index("유형").loc[TYPES]
        if mk != "MAHA":
            ax.bar(xs + (k - 1) * w, b["같이학습"], w, color=col, alpha=.28, edgecolor=col)
        ax.bar(xs + (k - 1) * w, b["빼고학습"], w * .62, color=col, label=sn)
    ax.axhline(.5, color="0.6", lw=.8, ls=":"); ax.set_xticks(xs); ax.set_xticklabels([f"{t}\n(n={int((ftype == t).sum())})" for t in TYPES])
    ax.set_title({"ET": "ET: 그 유형을 빼고 학습", "RF": "RF: 그 유형을 빼고 학습", "MAHA": "지도 없음: 정상만으로 거리"}[mk], fontsize=10, loc="left")
    ax.set_ylim(.4, 1.0); ax.grid(axis="y", alpha=.3)
axs[0].set_ylabel("그 유형 고장 vs 정상 120 AUC"); axs[0].legend(fontsize=8, loc="lower left")
fig.text(.01, -.02, "옅은 막대 = 그 유형도 학습에 있을 때(기본 OOF) · 진한 막대 = 그 유형을 학습에서 뺐을 때. 지도 없음은 고장을 안 보므로 한 막대", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "유형빼기.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

# %% 그림 2: 단일 피처 × 유형 AUC
fig, axs = plt.subplots(1, 2, figsize=(11, 6.2), gridspec_kw=dict(width_ratios=[1, 1]))
for ax, sn in zip(axs, ("우리 H15", "효섭 13")):
    c = C[C.묶음 == sn].sort_values("전체", ascending=False)
    M = c[["전체"] + TYPES].values
    im = ax.imshow(M, cmap="RdBu_r", vmin=.3, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(c))); ax.set_yticklabels([f"{p} ({d})" for p, d in zip(c.피처, c.방향)], fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["전체"] + TYPES, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j] - .65) > .25 else "black")
    ax.set_title(sn, fontsize=10, loc="left")
fig.colorbar(im, ax=axs, shrink=.6, label="AUC (그 유형 고장 vs 정상 120)")
fig.savefig(os.path.join(FIG, "피처x유형.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

# %% 그림 3: val 파일 × 묶음 (ET), 점수 = train 정상 OOF 안 백분위
order_g = list(VG) + ["나머지 21"]
Dv = Dfile.copy(); Dv["g"] = Dv["그룹"].map({g: i for i, g in enumerate(order_g)})
Dv = Dv.sort_values(["g", "우리 H15|ET"], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 10.5))
for k, (sn, col, mkr) in enumerate(zip(SETS, ["#2a6fdb", "#e07b1a", "#5a5a5a"], ["o", "s", "D"])):
    ax.scatter(Dv[f"{sn}|ET"], np.arange(len(Dv)) + (k - 1) * .22, s=22, color=col, marker=mkr, label=sn, zorder=3)
ax.axvline(99, color="crimson", lw=.8, ls="--"); ax.text(99, -1.3, "train 정상 99 %", color="crimson", fontsize=8, ha="right")
ax.set_yticks(range(len(Dv))); ax.set_yticklabels([n.replace("val_", "") for n in Dv.file], fontsize=8); ax.invert_yaxis()
for g in order_g:
    idx = np.where(Dv["그룹"] == g)[0]
    ax.axhspan(idx.min() - .5, idx.max() + .5, color="0.93" if order_g.index(g) % 2 else "white", zorder=0)
    ax.text(101, idx.mean(), g, fontsize=8, va="center")
ax.set_xlim(0, 100.5); ax.set_xlabel("ET 확률의 train 정상 OOF 안 백분위"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("val 40: 같은 ET, 피처 묶음만 바꿈 (그룹은 추론 · 진단용)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "val_묶음비교.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

# %% 저장
r3 = lambda D_: json.loads(D_.round(3).to_json(orient="records", force_ascii=False))
json.dump(dict(A_train=r3(A), B_유형빼기=r3(B), C_피처유형=r3(C), D_val요약=r3(D), D_val파일=r3(Dfile),
               유형=dict(zip(ntr, ftype.tolist()))), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, "표.md"), "w", encoding="utf-8") as fh:
    for nmx, Dx in [("A train OOF", A), ("B 유형 빼기", B), ("C 피처 × 유형", C), ("D val 요약", D), ("D val 파일 (백분위)", Dfile)]:
        fh.write(f"## {nmx}\n\n" + md_table(Dx.round(3)) + "\n\n")
print("done")
