# %% [markdown]
# V-8 모드별 감쇠비 (11-27)
# - 가설: H5 변형 — 연결부가 헐거워지면 감쇠가 커진다. 감쇠비는 모드마다 다르므로 모드별로 따로 보고 OR로 판단한다.
#         초기 고장은 정상과 같은 모드(본체 5.5–8 kHz)로 울되 ζ만 커질 수 있다(사용자 11-27)
# - 방법: 검출된 모든 충격(train 150 · val 40, 제출 모델과 같은 검출기 · 1–11 kHz)의 f · ζ · fit을 VP 연속 최적화로 추정(src/ringfit.py, V-3에서 검증)
#         충격 고르는 기준 2가지: 엄격(크기 ≥ train 정상 충격 75% 분위 · fit ≥ 0.8) · 느슨(검출된 충격 전부 · fit ≥ 0.6)
#         모드: 연결부 1.6–4.5 kHz · 본체 5.5–8 kHz (전이 · 고주파는 참고). 파일 · 모드마다 ζ 중앙값
#         정상 기준 = train 정상 파일들의 ζ 분포(라벨 = 정상만 씀) → 95 % 분위 넘으면 이상, 모드 OR
# - 감쇠 드리프트는 쓰지 않는다(V-4, 사용자 11-27)
#   conda run -n qe-phm python 5_VAL_단계/아카이브/V-8_모드별감쇠/자료/run.py

# %% 설정
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import src.viz  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.ringfit import est_vp, OFF, N8
from src.exp import md_table
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
Itr, Iva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
_, _, y, ntr, nva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
MODES = {"연결부 1.6–4.5k": (1600, 4500), "전이 4.5–5.5k": (4500, 5500), "본체 5.5–8k": (5500, 8000), "고주파 8–11k": (8000, 11000)}
MAIN = ["연결부 1.6–4.5k", "본체 5.5–8k"]

# %% ① 모든 충격의 f · ζ · fit (캐시)
cache = os.path.join(ROOT, "4_outputs", "cache", "val_eda_ringall.csv")
def file_rings(name, grp, im):
    out = []
    for p, hx in zip(im["idx"], im["height_x"]):
        s = im["hp"][p + OFF:p + OFF + N8]
        if len(s) < N8:
            continue
        v = est_vp(s); out.append(dict(file=name, 그룹=grp, t=round(p / 25000, 4), hx=float(hx), f=v["f"], z=v["z"], fit=v["fit"]))
    return out
if os.path.exists(cache):
    R = pd.read_csv(cache, encoding="utf-8-sig")
else:
    jobs = [(n, "train 정상" if yy == 0 else "train 고장", im) for n, yy, im in zip(ntr, y, Itr)] + [(n, "val", im) for n, im in zip(nva, Iva)]
    R = pd.DataFrame([r for rs in Parallel(n_jobs=-1)(delayed(file_rings)(*j) for j in jobs) for r in rs])
    R.to_csv(cache, index=False, encoding="utf-8-sig")
R["모드"] = pd.cut(R.f, [m[0] for m in MODES.values()][:1] + [m[1] for m in MODES.values()], labels=list(MODES), right=False)
HX75 = float(np.percentile(R[R.그룹 == "train 정상"].hx, 75))
CRIT = {"엄격": (R.hx >= HX75) & (R.fit >= 0.8), "느슨": (R.fit >= 0.6)}

# %% ② 파일 · 모드별 ζ 중앙값 → 정상 기준(95 % 분위) → 모드 OR
allf = pd.DataFrame({"file": list(ntr) + list(nva), "그룹": ["train 정상" if v == 0 else "train 고장" for v in y] + ["val"] * len(nva)})
summ, flags, tabs = {}, {}, {}
for cn, mask in CRIT.items():
    d = R[mask & R.모드.notna()]
    F = d.groupby(["file", "모드"], observed=True).z.median().unstack()
    F = allf.set_index("file").join(F)
    q95 = {m: float(np.nanpercentile(F.loc[F.그룹 == "train 정상", m], 95)) if F.loc[F.그룹 == "train 정상", m].notna().sum() >= 3 else np.nan for m in MODES}
    over = pd.DataFrame({m: F[m] > q95[m] for m in MODES if not np.isnan(q95[m])})
    F["OR_주요"] = over[[m for m in MAIN if m in over]].any(axis=1)
    rows = []
    for m in MODES:
        r = dict(모드=m, 기준_정상q95=None if np.isnan(q95[m]) else round(q95[m], 4))
        for g, n in [("train 정상", 120), ("train 고장", 30), ("val", 40)]:
            v = F.loc[F.그룹 == g, m]; r[f"{g} 파일수"] = f"{v.notna().sum()}/{n}"; r[f"{g} ζ중앙"] = round(float(v.median()), 4) if v.notna().any() else None
        both = F[F.그룹 != "val"][m].dropna(); yy = (F.loc[both.index, "그룹"] == "train 고장").astype(int)
        r["AUC(값있는 train)"] = round(roc_auc_score(yy, both), 3) if yy.nunique() == 2 else None
        rows.append(r)
    tabs[cn] = pd.DataFrame(rows)
    tn = F[F.그룹 == "train 정상"]; tf = F[F.그룹 == "train 고장"]; va = F[F.그룹 == "val"]
    flags[cn] = dict(train정상_이상=sorted(tn.index[tn.OR_주요]), train고장_이상=sorted(tf.index[tf.OR_주요]), val_이상=sorted(va.index[va.OR_주요]))
    summ[cn] = dict(충격수=int(mask.sum()), 정상오경보=f"{int(tn.OR_주요.sum())}/120", 고장검출=f"{int(tf.OR_주요.sum())}/30", val이상=f"{int(va.OR_주요.sum())}/40")
    F.to_csv(os.path.join(HERE, f"파일별_ζ_{cn}.csv"), encoding="utf-8-sig")
    # 그림: 주요 모드 두 개, 그룹별 파일 ζ
    if cn == "엄격":
        FS_ = {}
    FS_[cn] = (F, q95)

# %% ③ 확인: 같은 모드 안에서 ζ가 충격 크기 · fit에 따라 달라지나 (train 정상, 느슨 기준) — 크면 ζ가 달라지면 크기 차이가 섞인다
chk = {}
for m in MAIN:
    d = R[CRIT["느슨"] & (R.모드 == m) & (R.그룹 == "train 정상")]
    if len(d) > 10:
        chk[m] = dict(충격수=len(d), ρ_ζ_크기=round(spearmanr(d.z, d.hx).correlation, 2), ρ_ζ_fit=round(spearmanr(d.z, d.fit).correlation, 2))

# %% 그림
G = FuncFormatter(lambda v, _: f"{v:g}")
COL = {"train 정상": "C0", "train 고장": "C1", "val": "C4"}; rng = np.random.default_rng(0)
fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
for i, cn in enumerate(CRIT):
    F, q95 = FS_[cn]
    for j, m in enumerate(MAIN):
        a = axs[i, j]
        for k, g in enumerate(COL):
            v = F.loc[F.그룹 == g, m].dropna()
            a.scatter(k + rng.uniform(-.18, .18, len(v)), v, s=18, alpha=.65, color=COL[g])
            a.text(k, 0.0032, f"n={len(v)}", ha="center", fontsize=8)
        if not np.isnan(q95[m]):
            a.axhline(q95[m], color="C0", ls="--", lw=1); a.text(2.45, q95[m], " 정상 95%", color="C0", fontsize=8, va="bottom")
        a.set_yscale("log"); a.yaxis.set_major_formatter(G); a.set_ylim(0.003, 0.4)
        a.set_xticks(range(3)); a.set_xticklabels(list(COL)); a.grid(alpha=.3, axis="y")
        a.set_title(f"{cn} · {m}", fontsize=10)
    axs[i, 0].set_ylabel("파일별 ζ 중앙값 (로그)")
fig.suptitle(f"모드별 감쇠비 — 엄격: 크기 ≥ {HX75:.1f}배 · fit ≥ 0.8 / 느슨: 검출 충격 전부 · fit ≥ 0.6", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "모드별_감쇠비.png"), dpi=130); plt.close(fig)

# %% 저장
with open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8") as fh:
    fh.write(f"충격 전체 {len(R)}개 (train 정상 {int((R.그룹 == 'train 정상').sum())} · 고장 {int((R.그룹 == 'train 고장').sum())} · val {int((R.그룹 == 'val').sum())}). 엄격 크기 기준 {HX75:.1f}배\n\n")
    for cn in CRIT:
        fh.write(f"## {cn} — 충격 {summ[cn]['충격수']}개 · 주요 모드 OR: 정상 오경보 {summ[cn]['정상오경보']} · 고장 검출 {summ[cn]['고장검출']} · val 이상 {summ[cn]['val이상']}\n\n")
        fh.write(md_table(tabs[cn]) + "\n\n")
        fh.write(f"- train 정상 이상: {flags[cn]['train정상_이상']}\n- train 고장 이상: {flags[cn]['train고장_이상']}\n- val 이상: {flags[cn]['val_이상']}\n\n")
    fh.write("## 확인: train 정상 · 느슨 기준, 같은 모드 안에서 ζ와 충격 크기 · fit의 상관\n\n" + json.dumps(chk, ensure_ascii=False) + "\n")
json.dump(dict(HX75=HX75, 요약=summ, 이상파일=flags, 상관=chk), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done")
