# %% [markdown]
# V-4 시작점별 ζ 변화(감쇠 드리프트)를 피처로 쓸 수 있나 (11-27)
# - 가설: H5 변형 — 진폭에 따라 감쇠가 달라진다(접촉 · 마찰 같은 비선형 감쇠, 또는 감쇠가 다른 두 모드).
#         지수 감쇠 하나면 창 시작을 늦춰도 ζ가 같다 → drift = ln(ζ_late / ζ_early) ≈ 0. 앞이 빨리 · 뒤가 천천히 죽으면 drift < 0
# - 순서: ① 합성: drift가 의도한 현상에만 반응하나(잡음 바닥 편향 확인) ② train · val 강한 충격 전체 drift
#         ③ 파일 피처(가장 큰 충격 · 중앙값 · 최솟값) ④ train에서 단변량 AUC · 기존 ET와 중복 · ET에 더했을 때 CV ⑤ val 후보 위치
# - 기준(강한 충격 · fit)은 train 정상만으로 정한다. 실험(중첩 CV)은 하지 않는다(EDA 단계)
#   conda run -n qe-phm python 5_VAL_단계/아카이브/V-4_시작점감쇠변화/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
import src.viz  # noqa: F401
import matplotlib.pyplot as plt
from src.ringfit import zeta_drift, OFF, N8, FS
from src.exp import md_table
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
CAND = ["val_012", "val_032", "val_034", "val_036", "val_010", "val_014"]
SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
m = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")); core = m.core_
Itr, Iva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
Ttr, Tva, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))

# %% ① 합성: 실제 바탕(충격 없는 train 정상) + 울림 → 1–11 kHz 대역통과 → 포락선 봉우리 → drift
sos_bp = butter(4, (1000, 11000), btype="bandpass", fs=FS, output="sos"); sos_lp = butter(4, 1000, btype="lowpass", fs=FS, output="sos")
quiet = [im["hp"] for im, yy in zip(Itr, y) if yy == 0 and len(im["idx"]) == 0][:20]
rng = np.random.default_rng(0)

def synth(kind, f0, z0, k):
    t = np.arange(1500) / FS; w = 2 * np.pi * f0
    s = np.exp(-z0 * w * t) * np.sin(w * np.sqrt(1 - z0 ** 2) * t)
    if kind == "두 단계(빠른 ζ + 느린 ζ 0.006, 20 %)":
        w2 = 2 * np.pi * (f0 + 150); s += 0.2 * np.exp(-0.006 * w2 * t) * np.sin(w2 * t + 1)
    if kind == "맥놀이(+400 Hz, 같은 ζ, 50 %)":
        w2 = 2 * np.pi * (f0 + 400); s += 0.5 * np.exp(-z0 * w2 * t) * np.sin(w2 * t + 1)
    return s

syn = []
for kind in ["한 모드", "두 단계(빠른 ζ + 느린 ζ 0.006, 20 %)", "맥놀이(+400 Hz, 같은 ζ, 50 %)"]:
    for f0 in (1900, 6800):
        for z0 in (0.01, 0.033, 0.08):
            for amp_x in (15, 40, 120):                       # 바탕 표준편차 대비 충격 진폭 (실제 강한 충격 11–450배 범위)
                for rep in range(4):
                    bg = quiet[(rep * 3 + int(z0 * 100)) % len(quiet)][4000:6000].copy()
                    raw = bg.copy(); raw[300:1800] += amp_x * bg.std() * synth(kind, f0, z0, rep)
                    xh = sosfiltfilt(sos_bp, raw); e = sosfiltfilt(sos_lp, np.abs(hilbert(xh))); p = 300 + int(np.argmax(e[300:400]))
                    d = zeta_drift(xh, p)
                    syn.append(dict(종류=kind, f0=f0, z0=z0, 진폭배=amp_x, drift=d["drift"], fit=d["fit"]))
SY = pd.DataFrame(syn)
syn_tab = SY.groupby(["종류", "z0", "진폭배"]).drift.median().unstack().round(2)

# %% ② 실제 강한 충격 전체
hx_all = np.concatenate([np.asarray(im["height_x"]) for im, yy in zip(Itr, y) if yy == 0])
HX_MIN = float(np.percentile(hx_all, 75))
R = []
for I, names, labs in [(Itr, ntr, ["train 정상" if v == 0 else "train 고장" for v in y]), (Iva, nva, ["val"] * len(nva))]:
    for n, im, g in zip(names, I, labs):
        for p, hx in zip(im["idx"], im["height_x"]):
            if hx < HX_MIN:
                continue
            d = zeta_drift(im["hp"], int(p))
            if d is None:
                continue
            R.append(dict(file=n, 그룹=g, t=round(p / FS, 4), hx=float(hx), **d))
T = pd.DataFrame(R); T.to_csv(os.path.join(HERE, "충격_drift.csv"), index=False, encoding="utf-8-sig")
FIT_MIN = float(np.percentile(T[T.그룹 == "train 정상"].fit, 50))   # 믿을 만한 충격 = train 정상 강한 충격 fit 중앙값 이상

# %% ③ 파일 피처 (강한 충격이 없거나 믿을 만한 충격이 없으면 0 = 중립, 지수 감쇠 하나와 같은 값)
def file_feats(n):
    d = T[T.file == n]
    if len(d) == 0:
        return dict(DR_top=0.0, DR_med=0.0, DR_min=0.0, has=0)
    top = d.loc[d.hx.idxmax()]; g = d[d.fit >= FIT_MIN]
    return dict(DR_top=top.drift if top.fit >= FIT_MIN else 0.0, DR_med=g.drift.median() if len(g) else 0.0,
                DR_min=g.drift.min() if len(g) else 0.0, has=int(len(g) > 0))
Ftr = pd.DataFrame([file_feats(n) for n in ntr], index=ntr); Fva = pd.DataFrame([file_feats(n) for n in nva], index=nva)

# %% ④ train 평가
ev = {}
oof = core.Z_[:, 0]                                                  # 제출 모델의 ET OOF logit (학습에 안 쓰인 점수)
cols = list(core.cols) + list(core.peak); Etr, _ = core._et_input(Ttr)
for c in ["DR_top", "DR_med", "DR_min"]:
    v = Ftr[c].values; a = roc_auc_score(y, -v)                      # drift가 작을수록(음수) 고장이라는 방향을 가정 → AUC > 0.5면 가설 방향
    has = Ftr.has.values == 1
    rho_et = spearmanr(v, oof).correlation
    rho_h = max(abs(spearmanr(v, Etr[:, j]).correlation) for j in range(Etr.shape[1]))
    ev[c] = dict(AUC_가설방향=round(a, 3), AUC_값있는파일만=round(roc_auc_score(y[has], -v[has]), 3) if has.sum() > 5 and len(set(y[has])) == 2 else None,
                 값있는파일_정상=int((has & (y == 0)).sum()), 값있는파일_고장=int((has & (y == 1)).sum()),
                 rho_ET_OOF=round(rho_et, 3), max_rho_H피처=round(rho_h, 3))
# ET OOF logit에 더했을 때 (로지스틱, 반복 5-fold) — 기존 결합기와 같은 형태의 간단한 확인
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
def cv_score(X):
    au, ap = [], []
    for a, b in cv.split(X, y):
        p = LogisticRegression(C=1.0, class_weight="balanced").fit(X[a], y[a]).decision_function(X[b])
        au.append(roc_auc_score(y[b], p)); ap.append(average_precision_score(y[b], p))
    return round(float(np.mean(au)), 4), round(float(np.std(au)), 4), round(float(np.mean(ap)), 4)
add = {"ET OOF만": cv_score(oof[:, None])}
for c in ["DR_top", "DR_med", "DR_min"]:
    add[f"ET OOF + {c}"] = cv_score(np.column_stack([oof, Ftr[c].values]))

# %% 그림
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for k, kind in enumerate(SY.종류.unique()):
    d = SY[SY.종류 == kind].groupby(["z0", "진폭배"]).drift.median().unstack()
    for j, ax_ in enumerate(d.columns):
        axs[0].bar(np.arange(len(d)) + k * 0.28 + (j - 1) * 0.08, d[ax_].values, width=0.08, color=f"C{k}", alpha=.4 + .2 * j,
                   label=kind if j == 2 else None)
axs[0].set_xticks(np.arange(3) + .28); axs[0].set_xticklabels([f"ζ {z}" for z in (0.01, 0.033, 0.08)]); axs[0].axhline(0, color="k", lw=.8)
axs[0].set_ylabel("drift = ln(ζ_늦은 / ζ_이른) 중앙값"); axs[0].set_title("① 합성 (막대 3개 = 바탕 대비 15 · 40 · 120배)", fontsize=10)
axs[0].legend(fontsize=7.5, loc="lower left"); axs[0].grid(alpha=.3, axis="y")
COL = {"train 정상": "C0", "train 고장": "C1", "val": "C4"}
for g in COL:
    d = T[T.그룹 == g]; axs[1].scatter(d.fit, d.drift, s=8 + d.hx.clip(upper=200) / 4, alpha=.45, color=COL[g], label=g)
axs[1].axvline(FIT_MIN, color="C0", ls="--", lw=1); axs[1].axhline(0, color="k", lw=.8)
axs[1].set_xlabel("단일 모드 fit (이른 창)"); axs[1].set_ylabel("drift"); axs[1].set_title(f"② 강한 충격 {len(T)}개 (점선 = 믿을 만한 fit 기준 {FIT_MIN:.2f})", fontsize=10)
axs[1].legend(fontsize=8); axs[1].set_ylim(-4, 4); axs[1].grid(alpha=.3)
jit = lambda n: rng.uniform(-.15, .15, n)
for i, (lab, v) in enumerate([("train 정상", Ftr.DR_top[y == 0]), ("train 고장", Ftr.DR_top[y == 1]), ("val", Fva.DR_top)]):
    axs[2].scatter(i + jit(len(v)), v, s=14, alpha=.6, color=list(COL.values())[i])
for c in CAND:
    v = Fva.loc[c, "DR_top"]; axs[2].scatter(2, v, s=60, facecolor="none", edgecolor="k"); axs[2].annotate(c[4:], (2, v), fontsize=8, xytext=(6, 0), textcoords="offset points")
axs[2].set_xticks([0, 1, 2]); axs[2].set_xticklabels(["train 정상", "train 고장", "val"]); axs[2].axhline(0, color="k", lw=.8)
axs[2].set_ylabel("DR_top (가장 큰 충격의 drift, 믿을 만하지 않으면 0)"); axs[2].set_title("③ 파일 피처", fontsize=10); axs[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "드리프트.png"), dpi=130); plt.close(fig)

# %% 저장
cand_tab = Fva.loc[CAND].round(3).reset_index().rename(columns={"index": "file"})
with open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8") as fh:
    fh.write("## ① 합성 drift 중앙값 (열 = 바탕 대비 충격 진폭배)\n\n" + md_table(syn_tab.reset_index()) + "\n\n")
    fh.write(f"## ④ train 평가 (강한 충격 ≥ {HX_MIN:.1f}배, 믿을 만한 fit ≥ {FIT_MIN:.2f})\n\n" + md_table(pd.DataFrame(ev).T.reset_index()) + "\n\n")
    fh.write("## ④ ET OOF logit에 더했을 때 (로지스틱, 5-fold × 20회: AUC 평균 · sd · PR-AUC 평균)\n\n" +
             md_table(pd.DataFrame(add, index=["AUC", "sd", "PR-AUC"]).T.reset_index()) + "\n\n")
    fh.write("## ⑤ val 후보\n\n" + md_table(cand_tab))
json.dump(dict(HX_MIN=HX_MIN, FIT_MIN=FIT_MIN, 평가=ev, 더하기=add), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print("done")
