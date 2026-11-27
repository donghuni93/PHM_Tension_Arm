# %% [markdown]
# 4-5 후반. 경계 plot: AE z × ET(D+) z — train OOF 점수 + val, 두 점수의 상관 · val 이동 확인
# 축 = 전문가 점수 (AE: 정상 스펙트럼 학습, ET: D+ 81열). 점수는 train 5겹 OOF(seed 0). val은 train 150개 전체로 학습한 전문가로 점수(라벨 없음).
# 구성 = EXP-039의 "AE+ET(D+ 81열) 규칙 없음"과 같다. z = (점수 − train 정상 OOF 평균) / 표준편차. val 눈금은 학습 방식이 달라 조금 다를 수 있다(EXP-039와 같은 한계).
# 실행: python "3_단계/4_피처/4-5_1120_D플러스_val_tSNE/run_경계plot.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.evaluate import best_f2_threshold
from src.dplus import DPlus, AEET
from src.final import _fit_ae, _ae_score
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")

# %% 데이터 · OOF 점수 · val 점수
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X); g = np.arange(len(y))
dp = DPlus().fit(X, E); D = dp.D_
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val")); Dv, Tv = dp.transform(Xv)
F_ = np.c_[g, D]
S = np.zeros((len(y), 2)); sc = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F_, y):
    m = AEET(rules=False).fit(F_[tr], y[tr]); S[te], _ = m.expert_scores(F_[te]); sc[te] = m.st_.decision_function(m.sc_.transform(S[te]))
thr = best_f2_threshold(y, sc); fin = AEET(rules=False).fit(F_, y)
ae_final = _fit_ae(dp.T_["LP"][y == 0]); ae_v = _ae_score(ae_final, Tv["LP"])
Sv = np.c_[ae_v, fin._et(fin.X_, y, Dv)]; sv = fin.st_.decision_function(fin.sc_.transform(Sv)); pv = sv >= thr
mu, sd = S[y == 0].mean(0), S[y == 0].std(0); Z = (S - mu) / sd; Zv = (Sv - mu) / sd
p = sc >= thr; err = np.where(p != (y == 1))[0]

# %% 숫자 (라벨 안 쓰는 val 비교 포함)
lines = []
def L(s): print(s); lines.append(s)
L(f"OOF 임계값 {thr:.2f} | 재현율 {int((p & (y == 1)).sum())}/30 · 오경보 {int((p & (y == 0)).sum())}/120 | val 고장 판정 {int(pv.sum())}/40")
L(f"축별 AUC (OOF): AE {roc_auc_score(y, S[:, 0]):.3f} · ET {roc_auc_score(y, S[:, 1]):.3f}")
for nm, msk in (("전체", np.ones(len(y), bool)), ("정상만", y == 0), ("고장만", y == 1)):
    L(f"AE z와 ET z의 상관 ρ ({nm}): {np.corrcoef(Z[msk, 0], Z[msk, 1])[0, 1]:.2f}")
for j, ax_ in enumerate(("AE", "ET")):
    q95 = np.percentile(Z[y == 0, j], 95)
    L(f"{ax_} z — train 정상 중앙 {np.median(Z[y == 0, j]):.2f} · 95% {q95:.2f} | train 고장 중앙 {np.median(Z[y == 1, j]):.2f} | val 중앙 {np.median(Zv[:, j]):.2f} · val 중 정상 95% 초과 {int((Zv[:, j] > q95).sum())}/40 (같은 분포면 약 2 + 고장 몫)")
L("OOF 오류: " + ", ".join(f"{names[i]}({'놓침' if y[i] else '오경보'})" for i in err))
L("val 고장 판정: " + " ".join(f"val_{nv[i][-3:]}" for i in np.where(pv)[0]))
open(os.path.join(HERE, "경계plot_수치.txt"), "w", encoding="utf-8").write("\n".join(lines))
pd.DataFrame({"file": nv, "AE_z": Zv[:, 0].round(2), "ET_z": Zv[:, 1].round(2), "점수": sv.round(2), "판정": pv.astype(int)}).to_csv(os.path.join(HERE, "val_경계plot점수.csv"), index=False, encoding="utf-8-sig")

# %% 그림: 왼쪽 경계 plot, 오른쪽 축별 분포 (train 정상 · 고장 · val)
NOR, ABN, PUR, DARK = "#2a78d6", "#eb6834", "#b57cd1", "#4A148C"
w = fin.st_.coef_[0] / fin.sc_.scale_; b = fin.st_.intercept_[0] - (fin.st_.coef_[0] * fin.sc_.mean_ / fin.sc_.scale_).sum()
gx = np.linspace(-3, 6, 100); GX, GY = np.meshgrid(gx, gx); Zg = w[0] * (GX * sd[0] + mu[0]) + w[1] * (GY * sd[1] + mu[1]) + b
fig = plt.figure(figsize=(17, 7.4)); a = fig.add_axes([0.04, 0.1, 0.5, 0.82])
a.contourf(GX, GY, Zg, levels=[thr, 1e9], colors=["#FFE0B2"], alpha=0.5); a.contour(GX, GY, Zg, levels=[thr], colors=["k"], linewidths=1.7); a.contour(GX, GY, Zg, levels=[thr - 1, thr + 1], colors=["k"], linewidths=.7, linestyles=":")
a.scatter(Z[y == 0, 0], Z[y == 0, 1], s=22, color=NOR, alpha=.55, label="train 정상 (OOF)"); a.scatter(Z[y == 1, 0], Z[y == 1, 1], s=38, color=ABN, alpha=.9, label="train 고장 (OOF)")
a.scatter(Z[err, 0], Z[err, 1], marker="x", s=90, color="k", lw=1.4, label="× OOF 오류")
for i in err: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Z[i], fontsize=8, xytext=(4, 4), textcoords="offset points")
a.scatter(Zv[~pv, 0], Zv[~pv, 1], c=PUR, s=85, edgecolor=DARK, lw=1.2, zorder=5, label="val: 정상 판정"); a.scatter(Zv[pv, 0], Zv[pv, 1], c=DARK, s=85, edgecolor="white", lw=.8, zorder=6, label="val: 고장 판정")
for i in np.where(pv)[0]: a.annotate("v" + nv[i][-3:], Zv[i], fontsize=8, color=DARK, weight="bold", xytext=(5, 5), textcoords="offset points")
a.set_xlim(-3, 6); a.set_ylim(-3, 6); a.set_xlabel("AE z (스펙트럼 모양 이상)"); a.set_ylabel("ET(D+) z (충격·물리 피처 이상)"); a.legend(fontsize=8, loc="lower right")
a.set_title(f"경계 plot — 굵은 선 = 임계값 {thr:.2f}, 점선 ±1 (OOF 재현율 {int((p & (y == 1)).sum())}/30 · 오경보 {int((p & (y == 0)).sum())}/120 · val 고장 판정 {int(pv.sum())}/40)", fontsize=10)
for k, (j, nm) in enumerate(((0, "AE z"), (1, "ET(D+) z"))):
    h = fig.add_axes([0.6, 0.57 - 0.47 * k, 0.37, 0.33]); bins = np.linspace(-3, 8, 45)
    h.hist(Z[y == 0, j], bins=bins, color=NOR, alpha=.55, density=True, label="train 정상"); h.hist(Z[y == 1, j], bins=bins, color=ABN, alpha=.6, density=True, label="train 고장"); h.hist(Zv[:, j], bins=bins, histtype="step", color=DARK, lw=2, density=True, label="val 40")
    h.axvline(np.percentile(Z[y == 0, j], 95), color="k", ls=":", lw=1); h.set_title(f"{nm} 분포 (점선 = train 정상 95%)", fontsize=10); h.set_xlabel(nm); h.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "경계plot_AE×ET_D플러스.png"), dpi=115, bbox_inches="tight"); plt.close(fig); print("완료")
