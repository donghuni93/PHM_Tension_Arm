# EXP-044 질문답용 (run.py는 고치지 않음, 11-22). 라벨 없음(val), train 라벨은 J 학습에만
# ① val AE z를 아래로 옮기면(이동 보정) val 판정이 얼마나 바뀌나 — J · J′(H10 · H15 제거) · ET 단독
# ② val ET 점수가 정상보다 높은 것(이동)은 어느 피처에서 오나 — 피처별 백분위 + 분위 맞춤 기여
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/4_J모델/EXP-044_AE추가/자료/분석/val이동_민감도.py"
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE, lg
from src.evaluate import best_f2_threshold
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

ns = NS(); y, names = ns.y, ns.names; nor = y == 0; nv = [s[-3:] for s in ns.nv]
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); specv, resv = rule83_flags(ns.Xv); r83v = (specv | resv).astype(float)
idx = np.arange(ns.n, dtype=float)
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); raw_v = _ae_score(ae_tr, LPv)
DROP = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]


def build(fixed):
    """전체 150으로 J(또는 J′) 학습 → (모델, val 입력, train 입력)"""
    di = [ns.FIXED.index(c) for c in ns.FIXED if c not in fixed]
    M, Mv = np.delete(ns.M, di, axis=1), np.delete(ns.Mv, di, axis=1)
    mk = lambda: TwoAE("lin", ("E2",), fixed=tuple(fixed), nf=len(fixed), df=ns.df, peak=tuple(ns.PEAK))
    mJ = Add83(mk).fit(np.column_stack([M, idx, r83]), y); mJ.m_.ae_override_ = raw_v
    return mJ, np.column_stack([Mv, np.zeros(len(Mv)), r83v]), M, Mv


# %% ① AE 이동 민감도
out = []; P = lambda s="": (out.append(s), print(s))
models = {"J": build(ns.FIXED), "J′": build([c for c in ns.FIXED if c not in DROP])}
mm0 = models["J"][0].m_
aez_tr = mm0.Z_[:, 2]                                   # train AE z (안쪽 OOF, 전체 학습 모델)
fv0 = models["J"][0].fires(models["J"][1]); aez_v = fv0["ae"]
med_v, med_all = float(np.median(aez_v)), float(np.median(aez_tr))
SHIFT = {"그대로 (0)": 0.0, "-0.5": 0.5, "-1.0": 1.0, f"val 중앙→train 전체 중앙 (-{med_v - med_all:.2f})": med_v - med_all, f"val 중앙→train 정상 중앙 (-{med_v:.2f})": med_v}
P(f"AE z 중앙: train 정상 {np.median(aez_tr[nor]):.2f} · train 전체 {med_all:.2f} · train 고장 {np.median(aez_tr[~nor]):.2f} · val {med_v:.2f} (val ≥ 1.0: {np.mean(aez_v >= 1) * 100:.0f} %, train 정상 ≥ 1.0: {np.mean(aez_tr[nor] >= 1) * 100:.0f} %)")
tab = pd.DataFrame(dict(파일=nv))
cnt = {}
for nm, (mJ, Fv, M, Mv) in models.items():
    mm = mJ.m_; f = mJ.fires(Fv); c_ae = float(mm.c_.coef_[0][2])
    rule = f["E2"] | f["R83"]
    tab[f"{nm} ET"] = f["a"].round(2) if nm == "J" else tab.get("J ET")
    if nm == "J":
        tab["AE z"], tab["박자 z"] = f["ae"].round(2), f["z"].round(1)
        tab["규칙"] = np.where(f["R83"], "8.3k", np.where(f["E2"], "E2", ""))
    marg = (f["score"] - mm.tc_) / c_ae                 # 결합만으로 고장인 파일이 정상으로 바뀌려면 AE z를 얼마나 내려야 하나 (음수 = 이미 정상)
    tab[f"{nm} 여유(AE z)"] = np.where(rule, np.nan, marg).round(2)
    P(f"{nm}: 결합 계수 [ET, 박자 z, AE z] = {np.round(mm.c_.coef_[0], 3).tolist()} · 문턱 {mm.tc_:.2f}")
    for k, d in SHIFT.items():
        s = mm._comb_score(f["a"], f["z"], f["ae"] - d); on = (s >= mm.tc_) | rule
        tab[f"{nm} {k}"] = np.where(on, "고장", "")
        cnt[(nm, k)] = sorted(np.array(nv)[on].tolist())
t_et = float(best_f2_threshold(y, mm0.Z_[:, 0])); et_on = fv0["a"] > t_et
P(f"ET 단독 (ET ≥ {t_et:.2f}): 고장 {int(et_on.sum())} — {' '.join(np.array(nv)[et_on])}")
P("")
P("| 보정 | J 고장 수 | J 고장 파일 | J′ 고장 수 | J′ 고장 파일 |"); P("|---|---|---|---|---|")
base = {m: set(cnt[(m, "그대로 (0)")]) for m in models}
for k in SHIFT:
    cell = lambda m: " ".join(x if x in base[m] else f"**+{x}**" for x in cnt[(m, k)]) + ("".join(f" ~~{x}~~" for x in sorted(base[m] - set(cnt[(m, k)]))))
    P(f"| {k} | {len(cnt[('J', k)])} | {cell('J')} | {len(cnt[('J′', k)])} | {cell('J′')} |")
P("(굵게 = 보정 없을 때보다 새로 고장, 취소선 = 보정 후 정상으로 바뀜)")

# 그림: 결합만으로 판정되는 val 파일의 '여유'(AE z 단위) — 보정량 세로선과 비교
fig, ax = plt.subplots(figsize=(10, 7.5))
mg = tab["J 여유(AE z)"].values; ok = ~np.isnan(mg); order = np.argsort(mg[ok]); lab = np.array(nv)[ok][order]; v = mg[ok][order]
show = np.abs(v) < 3.5
ax.barh(np.arange(show.sum()), v[show], color=np.where(v[show] >= 0, ABN, NOR), height=.6)
ax.set_yticks(np.arange(show.sum())); ax.set_yticklabels(lab[show], fontsize=8)
for i_, (k, d) in enumerate(list(SHIFT.items())[1:]):
    ax.axvline(d, color="#555", lw=1, ls=":"); ax.text(d + .04, -.9 - .55 * i_, k.split(" (")[0] + f" ({d:.2f})", fontsize=8, va="center", ha="left", color="#333")
ax.set_ylim(-1.2 - .55 * (len(SHIFT) - 1), show.sum() - .4)
ax.axvline(0, color="#222", lw=1.2)
ax.set_xlabel("J 결합 판정 여유 (AE z 단위). 0 위 = 고장. 막대가 점선보다 짧으면 그만큼 보정했을 때 정상으로 바뀜")
ax.set_title(f"val 판정이 AE 보정에 얼마나 민감한가 (규칙으로 켜진 파일 제외, |여유| < 3.5만 표시)", fontsize=10); ax.grid(axis="x", alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(FIG, "val_AE보정_여유.png"), dpi=130); plt.close(fig)

# %% ② ET 이동의 출처 (피처별)
mJ, Fv, M, Mv = models["J"]; H_tr = mm0._parts(M)[0]; H_v = mm0._parts(Mv)[0]; cols = list(ns.FIXED) + list(ns.PEAK)
a_tr = mm0.Z_[:, 0]; a_v = fv0["a"]
P(""); P(f"ET logit 중앙: train 정상 OOF {np.median(a_tr[nor]):.2f} · train 전체 OOF {np.median(a_tr):.2f} · val {np.median(a_v):.2f}")
fvP = models["J′"][0].fires(models["J′"][1]); aP_tr = models["J′"][0].m_.Z_[:, 0]
P(f"J′ ET logit 중앙: train 정상 OOF {np.median(aP_tr[nor]):.2f} · train 전체 OOF {np.median(aP_tr):.2f} · val {np.median(fvP['a']):.2f}")


def qmap(v_src, v_ref):
    """v_src의 순위를 v_ref의 분위로 옮김 (분포만 맞추고 파일 순서는 유지)"""
    r = (pd.Series(v_src).rank(method="average").values - .5) / len(v_src)
    return np.quantile(v_ref, r)


imp = mm0.m_.feature_importances_
rows = []
for j, c in enumerate(cols):
    pv = np.array([np.mean(H_tr[nor, j] <= x) * 100 for x in H_v[:, j]])
    hi = np.mean(H_v[:, j] > np.percentile(H_tr[nor, j], 95)) * 100
    Hs = H_v.copy(); Hs[:, j] = qmap(H_v[:, j], H_tr[:, j])          # val의 이 열만 train 전체 분포로 맞춤
    d_all = float(np.median(a_v) - np.median(lg(mm0.m_.predict_proba(Hs)[:, 1])))
    Hs[:, j] = qmap(H_v[:, j], H_tr[nor, j]); d_nor = float(np.median(a_v) - np.median(lg(mm0.m_.predict_proba(Hs)[:, 1])))
    rows.append(dict(피처=c, ET중요도=round(float(imp[j]), 3), val중앙_백분위=round(float(np.median(pv))), val_정상95위=round(hi), 고장중앙_백분위=round(float(np.median([np.mean(H_tr[nor, j] <= x) * 100 for x in H_tr[~nor, j]]))),
                     ET하락_전체맞춤=round(d_all, 2), ET하락_정상맞춤=round(d_nor, 2)))
R = pd.DataFrame(rows).sort_values("ET하락_정상맞춤", ascending=False)
Hs = H_v.copy()
for j in range(len(cols)): Hs[:, j] = qmap(H_v[:, j], H_tr[nor, j])
P(f"모든 열을 train 정상 분포로 맞추면 val ET 중앙 {np.median(lg(mm0.m_.predict_proba(Hs)[:, 1])):.2f} (원래 {np.median(a_v):.2f})")
P(""); P("val중앙_백분위 = val 중앙값이 train 정상 분포에서 몇 %인가(50이면 같음). val_정상95위 = val 중 train 정상 95 % 분위를 넘는 비율(정상이면 5 %). 고장중앙_백분위 = train 고장의 같은 값(참고)")
P("ET하락 = val의 그 열 하나만 train 분포로 맞췄을 때 val ET 중앙이 얼마나 내려가나(클수록 그 열이 이동의 원인). 전체맞춤 = 고장 20 % 섞인 train 전체 분포, 정상맞춤 = train 정상 분포"); P("")
P(md_table(R))

# ②-b "val에 고장이 많아서인가": 고장이 섞인 것만으로 이동이 생겼다면, 피처마다 같은 고장 비율 p로 설명돼야 한다
# p_j = (val 중 정상 95 % 분위 초과 비율 − 0.05) / (train 고장 중 초과 비율 − 0.05). 위쪽 꼬리가 고장 방향인 피처만(고장 초과 ≥ 30 %)
P(""); P("②-b 혼합 비율 추정: val 이동이 '고장이 섞여서'만이라면 피처마다 같은 고장 비율 p가 나와야 함 (위 꼬리 = 고장 방향, train 고장 초과 ≥ 30 %인 피처)")
P("| 피처 | train 고장 중 정상95위 초과 | val 중 초과 | 추정 고장 비율 p |"); P("|---|---|---|---|")
Hall_tr = np.column_stack([H_tr, ns.T[["L1_S"]].values]); Hall_v = np.column_stack([H_v, ns.Tv[["L1_S"]].values])
for j, c in enumerate(cols + ["L1_S"]):
    q = np.percentile(Hall_tr[nor, j], 95); fa = np.mean(Hall_tr[~nor, j] > q); fv_ = np.mean(Hall_v[:, j] > q)
    if fa >= .3: P(f"| {c} | {fa * 100:.0f} % | {fv_ * 100:.0f} % | {max(0, (fv_ - .05) / (fa - .05)) * 100:.0f} % |")
open(os.path.join(HERE, "figs", "val이동_민감도_log.txt"), "w", encoding="utf-8").write("\n".join(out))
tab.to_csv(os.path.join(HERE, "figs", "val이동_파일별.csv"), index=False, encoding="utf-8-sig")
print("done")
