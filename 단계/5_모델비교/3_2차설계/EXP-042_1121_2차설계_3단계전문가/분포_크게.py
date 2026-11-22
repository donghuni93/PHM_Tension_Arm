# EXP-042 해석용 보조 그림 (run.py는 고치지 않음): ① 새 모델 평면 크게(글씨 안 겹치게) ② ET 점수 하나만의 분포
# 실행: conda run -n qe-phm python "단계/5_모델비교/3_2차설계/EXP-042_1121_2차설계_3단계전문가/분포_크게.py"
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score, average_precision_score
from src.exp2util import *
from src.evaluate import best_f2_threshold
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs")
ns = NS(); y, names = ns.y, ns.names
W = json.load(open(os.path.join(find_exp("EXP-041"), "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]
mk = ns.mk(W, ("E1", "E2"))
res = json.load(open(os.path.join(HERE, "result.json"), encoding="utf-8"))
row = res["configs"]["중첩: D2 | D1 + E2 (하중 대비 덜컥)"]
fo = oof_fires(ns, mk, seed=0)
a, z = fo["a"], fo["z"]; w = wrong_of(ns, row); out = []

# ---- ① 평면 크게: y를 symlog로 펴서 z≈0 줄에 몰린 점을 보이게, 라벨은 위아래로 엇갈려 배치
fig, ax = plt.subplots(figsize=(16, 11))
m = mk().fit(ns.M, y)
xx, yy = np.meshgrid(np.linspace(-7, 7, 400), np.linspace(-3, 20, 400)); D = m._comb_score(xx.ravel(), yy.ravel()).reshape(xx.shape)
t = m.tc_ if hasattr(m, "tc_") else 0
ax.contourf(xx, yy, D, levels=[-1e9, t, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy, D, levels=[t], colors="k", linewidths=1.8)
ax.set_yscale("symlog", linthresh=1.0)
xc, zc = np.clip(a, -7, 7), np.clip(z, -3, 20); rng = np.random.RandomState(0); zj = zc + 0  # z=0 줄은 그대로(축 의미 유지)
nor = ns.nor
ax.scatter(xc[nor], zj[nor], color=NOR, s=40, marker="s", lw=0, alpha=.75, label="정상 120")
ax.scatter(xc[~nor], zj[~nor], color=ABN, s=55, lw=0, alpha=.95, label="고장 30")
for e, mkr, col in (("E1", "D", "#2ca02c"), ("E2", "P", "#7a3db8")):
    if fo[e].any(): ax.scatter(xc[fo[e]], zj[fo[e]], s=190, marker=mkr, facecolor="none", edgecolor=col, lw=1.6, label=f"{e}가 켠 파일")
# 라벨: 고장은 모두, 정상은 자주 틀린 것만. x순으로 정렬 후 위/아래 번갈아 오프셋
lab = sorted(set(list(w) + [i for i in np.flatnonzero(~nor)]), key=lambda i: xc[i])
ys_off = [(0, 14), (0, -20), (14, 24), (-14, -30), (16, 34), (-16, 42)]
for k, i in enumerate(lab):
    s = short(names[i]) + (f" {'미탐' if y[i] else '오경보'}{w[i]}/5" if i in w else "")
    dx, dy = ys_off[k % len(ys_off)]
    ax.annotate(s, (xc[i], zj[i]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=ABN if y[i] else NOR, weight="bold" if i in w else None,
                arrowprops=dict(arrowstyle="-", color="#888", lw=.5), ha="center")
ax.scatter(xc[list(w)], zj[list(w)], s=100, marker="x", color="k", lw=1.8, label="자주 틀림 (2/5+)")
ax.set_xlabel("ET 점수 (logit, 폴드 밖 OOF)"); ax.set_ylabel("박자 z (symlog: |z|≤1 선형, 그 위 로그)"); ax.grid(alpha=.3); ax.legend(fontsize=11, loc="upper left")
ax.set_title(f"EXP-042 D2 (ET + 박자 z → 로지스틱 + 전문가 E1·E2) — 중첩 F2 {row['f2']:.3f} ± {row['f2_sd']:.3f} · 재현율 {row['recall']:.3f} · 오경보율 {row['fpr']:.3f}", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "분포_평면_크게.png"), dpi=130); plt.close(fig)

# ---- ② ET 하나만: OOF logit 분포 (seed 0) + 3 seed 평균 AUC · PR-AUC · 최적 문턱 F2(낙관적)
aucs, aps = [], []
for s in range(5):
    f = oof_fires(ns, mk, seed=s)["a"]; aucs.append(roc_auc_score(y, f)); aps.append(average_precision_score(y, f))
fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
bins = np.linspace(a.min() - .2, a.max() + .2, 45)
axs[0].hist(a[nor], bins=bins, color=NOR, alpha=.7, label="정상 120"); axs[0].hist(a[~nor], bins=bins, color=ABN, alpha=.8, label="고장 30")
axs[0].set_ylabel("파일 수"); axs[0].legend(fontsize=11); axs[0].grid(alpha=.3)
thr = best_f2_threshold(y, a); axs[0].axvline(thr, color="k", ls="--"); axs[0].text(thr, axs[0].get_ylim()[1] * .92, f" F2 최대 문턱 {thr:.2f}\n (사후 선택, 낙관적)", fontsize=9)
axs[0].set_title(f"ET 점수 하나만 (OOF logit, seed 0) — 5 seed 평균 ROC-AUC {np.mean(aucs):.3f} · PR-AUC {np.mean(aps):.3f}", fontsize=12)
axs[1].scatter(a[nor], rng.uniform(-.35, .35, nor.sum()) + 0, color=NOR, s=30, marker="s", alpha=.7); axs[1].scatter(a[~nor], rng.uniform(-.35, .35, (~nor).sum()) + 1, color=ABN, s=40, alpha=.9)
axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(["정상", "고장"]); axs[1].grid(alpha=.3); axs[1].set_xlabel("ET 점수 (logit, 폴드 밖 OOF)")
sk = sorted([i for i in np.flatnonzero(~nor)], key=lambda i: a[i])
for k, i in enumerate(sk):
    if a[i] < thr + .0: axs[1].annotate(short(names[i]), (a[i], 1 + rng.uniform(-.35, .35)), xytext=(0, 9 + 9 * (k % 3)), textcoords="offset points", fontsize=8.5, color=ABN, ha="center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "분포_ET만.png"), dpi=130); plt.close(fig)

pred = a > thr
out.append(f"ET만 OOF seed0: 문턱 {thr:.3f} → 재현율 {pred[~nor].mean():.3f} 오경보율 {pred[nor].mean():.3f}")
out.append(f"ROC-AUC 5seed {np.round(aucs,3).tolist()} 평균 {np.mean(aucs):.3f} / PR-AUC {np.round(aps,3).tolist()} 평균 {np.mean(aps):.3f}")
below = [(short(names[i]), round(float(a[i]), 2)) for i in np.flatnonzero(~nor & ~pred)]
out.append(f"그 문턱에서 놓친 고장: {below}")
srt = np.sort(a[nor]); out.append(f"정상 ET 점수 분위 50/90/95/99: {np.percentile(a[nor],[50,90,95,99]).round(2).tolist()} · 고장 중앙 {np.median(a[~nor]):.2f} · 최소 {a[~nor].min():.2f}")
out.append(f"정상 중 문턱 위: {[(short(names[i]), round(float(a[i]),2)) for i in np.flatnonzero(nor & pred)]}")
open(os.path.join(HERE, "figs", "분포_log.txt"), "w", encoding="utf-8").write("\n".join(out))
