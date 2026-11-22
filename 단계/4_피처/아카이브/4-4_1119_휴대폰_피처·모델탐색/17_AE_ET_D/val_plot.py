# %% [markdown]
# 17-2. AE + ET(D) 스택의 경계 plot + val (보라). 구성 P3 = AE + ET(D 76열) → 로지스틱 + 규칙(8.3k · 유격, EXP-025와 같은 규칙) / P1 = 규칙 없음
# train 점은 5겹 OOF, val은 train 150개 전체로 학습한 최종 모델로 채점. 규칙 판정(val)은 EXP-025 최종 모델의 근거표 열(rule83 · knock)을 그대로 쓴다.
import os, sys, warnings, copy
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read().split("g = np.arange(len(y))")[0]
G = {"__file__": os.path.join(HERE, "run.py")}; exec(compile(src, "run17", "exec"), G)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.model_selection import StratifiedKFold
from src.evaluate import best_f2_threshold
from src.final import signal_tables, _fit_ae, _ae_score
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
AEET, D, y, names, E = G["AEET"], G["D"], G["y"], G["names"], G["E"]
g16 = G["g16"]; Dv, nv, Xtr, Xv, mark = g16["Dv"], g16["nv"], g16["X"], g16["Xv"], g16["mark"]
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
Fall = np.c_[np.arange(len(y)), D]
def experts(m, F):
    F = np.asarray(F); te = F[:, 0].astype(int); return np.c_[G["ae_mean"](m.idx_, te)[:, 0], m._et(m.X_, m.y_, F[:, 1:])], te
# %% train OOF (5겹, seed 0)
S = np.zeros((len(y), 2)); sc = {False: np.zeros(len(y)), True: np.zeros(len(y))}; rule = np.zeros(len(y), bool)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Fall, y):
    for rl in (False, True):
        m = AEET(rules=rl).fit(Fall[tr], y[tr]); sc[rl][te] = m.decision_function(Fall[te])
        if rl: rule[te] = np.any([r(te) for r in m.rules_], axis=0)
    S[te], _ = experts(m, Fall[te])
thr = {rl: best_f2_threshold(y, sc[rl]) for rl in (False, True)}
for rl in (False, True):
    p = sc[rl] >= thr[rl]; print(f"train OOF (규칙 {'O' if rl else 'X'}): 임계값 {thr[rl]:.2f} 재현율 {int((p & (y == 1)).sum())}/30 오경보 {int((p & (y == 0)).sum())}/120 | 놓친 {[names[i][-3:] for i in np.where(~p & (y == 1))[0]]} | 오경보 {[names[i][-3:] for i in np.where(p & (y == 0))[0]]}")
# %% 최종 모델 → val
fin = {rl: AEET(rules=rl).fit(Fall, y) for rl in (False, True)}
Tt, Tv = signal_tables(Xtr)[0], signal_tables(Xv, signal_tables(Xtr)[1])[0]
ae = _fit_ae(Tt["LP"][y == 0]); ae_v = _ae_score(ae, Tv["LP"])
et_v = fin[False]._et(D, y, Dv); Sv = np.c_[ae_v, et_v]
sv = {rl: fin[rl].st_.decision_function(fin[rl].sc_.transform(Sv)) for rl in (False, True)}
t25 = pd.read_csv("/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad/val_근거표_025.csv").set_index("file")
rule_v = (t25.loc[nv, "rule83"].values | t25.loc[nv, "knock"].values).astype(bool); sv[True] = sv[True] + 100.0 * rule_v
pred = {rl: sv[rl] >= thr[rl] for rl in (False, True)}
mu, sd = S[y == 0].mean(0), S[y == 0].std(0); Z, Zv = (S - mu) / sd, (Sv - mu) / sd
# 경계 (규칙 제외 로지스틱 결정함수 = thr) 평면
m0 = fin[False]; w = m0.st_.coef_[0] / m0.sc_.scale_; b = m0.st_.intercept_[0] - (m0.st_.coef_[0] * m0.sc_.mean_ / m0.sc_.scale_).sum()
gx = np.linspace(-3, 5, 90); gy = np.linspace(-3, 5, 90); GX, GY = np.meshgrid(gx, gy); Zg = w[0] * (GX * sd[0] + mu[0]) + w[1] * (GY * sd[1] + mu[1]) + b
PUR, LIGHT, DARK = "#7B1FA2", "#CE93D8", "#4A148C"
fig, ax = plt.subplots(1, 2, figsize=(18, 7.2)); a = ax[0]
a.contourf(GX, GY, Zg, levels=[thr[False], 1e9], colors=[LIGHT], alpha=.2); a.contour(GX, GY, Zg, levels=[thr[False]], colors=[DARK], linewidths=1.8)
a.scatter(Z[y == 0, 0], Z[y == 0, 1], s=24, color="#9E9E9E", alpha=.65, label="train 정상 (OOF)"); a.scatter(Z[y == 1, 0], Z[y == 1, 1], s=40, color="#EF6C00", alpha=.9, label="train 고장 (OOF)")
a.scatter(Z[rule, 0], Z[rule, 1], s=150, facecolor="none", edgecolor="#EF6C00", lw=1.2, label="train 규칙 켜짐")
a.scatter(Zv[:, 0], Zv[:, 1], c=np.where(pred[True], DARK, LIGHT), s=95, edgecolor=DARK, lw=1.4, zorder=5, label="val: 고장 판정 (진한 보라, 규칙 포함)"); a.scatter([], [], c=LIGHT, s=95, edgecolor=DARK, label="val: 정상 판정 (연보라)")
a.scatter(Zv[rule_v, 0], Zv[rule_v, 1], s=240, facecolor="none", edgecolor=PUR, lw=2, zorder=4, label="val 규칙 켜짐")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Z[i], fontsize=8, xytext=(4, 4), textcoords="offset points")
for i in np.where(pred[True])[0]: a.annotate(nv[i][-3:], Zv[i], fontsize=8, color=DARK, weight="bold", xytext=(5, 5), textcoords="offset points")
a.set_xlim(-3, 5); a.set_ylim(-3, 5); a.set_xlabel("AE z (스펙트럼 모양 이상)"); a.set_ylabel("ET(D) z (D 76열 충격·울림 이상)"); a.legend(fontsize=8, loc="lower right")
a.set_title(f"AE + ET(D) 스택 경계 (선 = 규칙 제외 임계값 {thr[False]:.2f})\nval 고장 판정 {int(pred[True].sum())}/40 (규칙 포함) · train OOF 규칙 포함 재현율/오경보 {int(((sc[True] >= thr[True]) & (y == 1)).sum())}/30 · {int(((sc[True] >= thr[True]) & (y == 0)).sum())}/120", fontsize=10)
a = ax[1]; bins = np.linspace(-9, 10, 50); s_ = np.minimum(sc[True], 9.5)
a.hist(s_[y == 0], bins=bins, color="#9E9E9E", alpha=.65, label="train 정상 (OOF)"); a.hist(s_[y == 1], bins=bins, color="#EF6C00", alpha=.75, label="train 고장 (OOF), 규칙 켜진 파일은 오른쪽 끝에 모음")
a.axvline(thr[True], color=DARK, ls="--", lw=1.6, label=f"임계값 {thr[True]:.2f}"); sv_ = np.minimum(sv[True], 9.5)
a.scatter(sv_, -2.5 + np.random.default_rng(0).uniform(-.9, .9, 40), c=np.where(pred[True], DARK, LIGHT), s=65, edgecolor=DARK, zorder=5, label="val 40"); a.set_xlabel("결합 점수 (규칙 켜진 파일은 9.5로 표시)"); a.set_ylabel("파일 수"); a.legend(fontsize=8); a.set_title("결합 점수 분포: train OOF 대비 val")
fig.savefig(os.path.join(FIG, "boundary_val.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
f25 = set(t25.index[t25.pred == 1]); fP3 = set(np.array(nv)[pred[True]]); fP1 = set(np.array(nv)[pred[False]])
V = pd.DataFrame({"file": nv, "ae_z": Zv[:, 0], "et_D_z": Zv[:, 1], "P1 점수": sv[False], "P3 점수": sv[True], "P1 판정": pred[False].astype(int), "P3 판정": pred[True].astype(int), "025 판정": [int(f in f25) for f in nv]}).sort_values("P3 점수", ascending=False)
V.round(3).to_csv(os.path.join(HERE, "val_scores.csv"), index=False, encoding="utf-8-sig")
print("val 고장 판정 P3:", sorted(f[-3:] for f in fP3), "| P1(규칙 없음):", sorted(f[-3:] for f in fP1)); print("025 모델:", sorted(f[-3:] for f in f25))
print("P3 ∩ 025:", len(fP3 & f25), "| P3만:", sorted(f[-3:] for f in fP3 - f25), "| 025만:", sorted(f[-3:] for f in f25 - fP3))
print("val z 평균 (AE/ET(D)):", Zv.mean(0).round(2), "| train 정상 0, 고장 평균:", Z[y == 1].mean(0).round(2), "| 경계 ±0.5:", [nv[i][-3:] for i in np.where(np.abs(sv[False] - thr[False]) < 0.5)[0]])
