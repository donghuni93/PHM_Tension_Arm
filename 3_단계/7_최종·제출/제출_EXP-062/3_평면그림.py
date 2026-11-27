# %% [markdown]
# 최종 모델(EXP-062 구성 3) 결합 평면 위에 학습 150 · val 40 · test 40을 패널 3개로 나눠 찍는다 (11-27)
# - 저장된 모델(모델/final_model_062.joblib)만 읽는다. 다시 학습하지 않고 제출 CSV도 만들지 않는다
# - 축 = EXP-062 logistic_plane과 같음: 가로 ET logit + (c박자/cET)·박자 z, 세로 AE z, 배경 = 결합 판단, 세로 점선 = ET 단독 판단 경계
# - 점 색 = 최종 판정(결합 OR E2 OR 8.3k), 이름표는 서로 겹치지 않게 자리를 고른다
#     conda run -n qe-phm python "3_단계/7_최종·제출/제출_EXP-062/3_평면그림.py"
# %%
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.data import load_folder
from src.evaluate import best_f2_threshold
from src.exp2util import NS, NOR, ABN
from src.exp import outlier_idx
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"

m = joblib.load(os.path.join(HERE, "모델", "final_model_062.joblib")); core = m.core_
ns = NS(); y = ns.y; names = np.array(ns.names)
Xv, nv = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
Xt, nt = load_folder(os.path.join(ROOT, "2_data", "TeamProject_test", "test"))
Dv, Dt = m.predict_table(Xv, nv), m.predict_table(Xt, nt)
Pt = core.parts(m._tables(ns.X))
c, b, tc = core.c_.coef_[0], core.c_.intercept_[0], core.tc_
Z = core.Z_; t_et = float(best_f2_threshold(y, Z[:, 0])); k = c[1] / c[0]
comb_tr = core.c_.decision_function(Z) >= tc
fin_tr = comb_tr | Pt.E2.values | Pt.rule83.values
short = lambda s: str(s).replace("nor_", "N").replace("abn_", "A").replace("val_", "V").replace("test_", "T").replace(".csv", "")

# 세 묶음: (제목, 가로, 세로, 최종 판정, 이름, 표시 마스크)
tr_show = ~((y == 1) & ~comb_tr & fin_tr)                          # 규칙만으로 잡힌 학습 고장은 평면에서 뺌 (결정.md 기본 그림)
G = [("학습 150 (안쪽 OOF 입력)", Z[:, 0] + k * Z[:, 1], Z[:, 2], fin_tr, names, tr_show),
     ("val 40", Dv.et_logit.values + k * Dv.beat_z.values, Dv.ae_z.values, Dv.pred.values.astype(bool), np.array(nv), np.ones(len(Dv), bool)),
     ("test 40", Dt.et_logit.values + k * Dt.beat_z.values, Dt.ae_z.values, Dt.pred.values.astype(bool), np.array(nt), np.ones(len(Dt), bool))]
allx = np.concatenate([g[1][g[5]] for g in G] + [[t_et]]); ally = np.concatenate([g[2][g[5]] for g in G])
x0, x1, y0, y1 = allx.min() - .6, allx.max() + .6, ally.min() - .4, ally.max() + .4
gx, gy = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
bg = (c[0] * gx + c[2] * gy + b >= tc).astype(float)


def place_labels(ax, fig, xs, ys, txt, all_x, all_y):
    """이름표를 점 주변 후보 자리 중 이미 놓인 이름표 · 패널의 모든 점과 겹치지 않는 첫 자리에 놓는다.
    가까운 자리가 다 막히면 더 먼 자리에 가는 선으로 잇는다"""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.c_[all_x, all_y]); taken = []
    ring = [(6, 4), (6, -12), (-6, 4), (-6, -12), (0, 9), (0, -18)]
    far = [(a * s, bb) for s in (18, 30, 45) for a in (1, -1) for bb in (s, -s, 0)]
    for i in np.argsort(ys)[::-1]:
        chosen = None
        for dx, dy in ring + far:
            kw = dict(arrowprops=dict(arrowstyle="-", lw=.5, color="#777")) if (dx, dy) in far else {}
            t = ax.annotate(txt[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, zorder=4,
                            ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center", **kw)
            bb = t.get_window_extent(r).expanded(1.08, 1.15)
            if not any(bb.overlaps(q) for q in taken) and not any(bb.contains(px, py) for px, py in pts):
                chosen = t; break
            t.remove()
        if chosen is None:
            chosen = ax.annotate(txt[i], (xs[i], ys[i]), xytext=far[-1], textcoords="offset points", fontsize=7.5, va="center",
                                 arrowprops=dict(arrowstyle="-", lw=.5, color="#777"))
        taken.append(chosen.get_window_extent(r).expanded(1.08, 1.15))


fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
o = np.zeros(len(y), bool); o[outlier_idx(names)] = True
for ax, (tt, x, y3, fin, nm, show) in zip(axs, G):
    ax.contourf(gx, gy, bg, levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    col = np.where(fin, ABN, NOR)
    if tt.startswith("학습"):
        for msk, mk, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mk, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        lab = np.flatnonzero(show & (fin != (y == 1)))          # 학습은 틀린 파일만 이름
        sub = f"틀린 파일 {len(lab)}개 이름 표시 · 규칙만으로 잡힌 고장 {int((~show).sum())}개는 뺌"
    else:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        lab = np.flatnonzero(fin | (np.abs(Dt.margin.values if tt.startswith("test") else Dv.margin.values) < 1.0))
        sub = f"고장 판정 {int(fin.sum())}/{len(fin)} · 이름 = 고장 판정 + 여유 |점수-문턱| < 1"
        if tt.startswith("val"): sub += " (val도 같은 기준)"
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    if len(lab): place_labels(ax, fig, x[lab], y3[lab], [short(nm[i]) for i in lab], x[show], y3[show])
    ax.set_xlabel(f"ET logit + ({k:.2f})·박자 z", color=MUTED); ax.set_title(f"{tt}\n{sub}", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper left")
axs[0].set_ylabel("AE z", color=MUTED)
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상"), Line2D([], [], marker="^", ls="", color="#999", label="▲ 라벨 없음(val · test)")]
fig.legend(handles=leg, ncol=6, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("EXP-062 구성 3 최종 모델 — 같은 결합 평면 위의 학습 · val · test (train 150 전체 학습, 다시 학습 안 함). 배경 = 결합 판단, 점선 = ET 단독 판단",
             x=.01, ha="left", fontsize=11, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .94))
OUT = os.path.join(HERE, "결과"); os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "평면_학습_val_test.png"), dpi=130)

# %% ET 단독 1차원 그림 (11-27 사용자): 가로 = ET logit만(박자 · AE 없음), 세로 = 겹치지 않게 쌓은 자리(의미 없음)
#    세로 점선 = ET 단독 판단 경계(학습 OOF ET logit의 F2 최적 문턱). 점 색 = 최종 판정(결합 OR 규칙), 테두리 굵음 = ET 단독 판정과 최종 판정이 다름
def swarm(xs, d):
    """가로 x는 그대로 두고, 이미 놓인 점과 거리 d 안에 들지 않는 가장 가운데 세로 자리를 고른다 (단위 = 화면 비율로 맞춘 값)"""
    ys = np.full(len(xs), np.nan); placed = []
    for i in np.argsort(xs):
        for k_ in range(200):
            cy = (k_ + 1) // 2 * d * (1 if k_ % 2 else -1)
            if all((xs[i] - px) ** 2 + (cy - py) ** 2 >= d * d for px, py in placed if abs(xs[i] - px) < d):
                ys[i] = cy; placed.append((xs[i], cy)); break
    return ys


E = [("학습 150 (안쪽 OOF ET logit)", Z[:, 0], fin_tr, names, tr_show),
     ("val 40", Dv.et_logit.values, Dv.pred.values.astype(bool), np.array(nv), np.ones(len(Dv), bool)),
     ("test 40", Dt.et_logit.values, Dt.pred.values.astype(bool), np.array(nt), np.ones(len(Dt), bool))]
ex_all = np.concatenate([e[1][e[4]] for e in E]); ex0, ex1 = ex_all.min() - .4, ex_all.max() + .4
D_SW = (ex1 - ex0) / 95                                            # 점 지름 ≈ 가로 폭의 1/95
fig, axs = plt.subplots(3, 1, figsize=(20, 13), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1, 1]))
for ax, (tt, et, fin, nm, show) in zip(axs, E):
    x = et[show]; f = fin[show]; nmm = nm[show]; sw = swarm(x, D_SW)
    et_only = x >= t_et; col = np.where(f, ABN, NOR); diff = f != et_only
    ax.axvspan(ex0, t_et, color="#d6e6fa", zorder=0); ax.axvspan(t_et, ex1, color="#fbe0d0", zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    if tt.startswith("학습"):
        yy = y[show]; oo = o[show]
        for msk, mk, s_ in (((yy == 0) & ~oo, "s", 45), (yy == 1, "o", 55), (oo, "D", 70)):
            ax.scatter(x[msk], sw[msk], marker=mk, s=s_, c=col[msk], edgecolor="k", lw=np.where(diff[msk], 2.0, .5), zorder=3)
        lab = np.flatnonzero((f != (yy == 1)) | diff)
        sub = f"ET 단독 고장 {int(et_only.sum())} · 최종 고장 {int(f.sum())} · 이름 = 라벨과 최종 판정이 다르거나 ET 단독과 최종이 다른 파일 · 규칙만으로 잡힌 고장 {int((~show).sum())}개는 뺌"
    else:
        ax.scatter(x, sw, marker="^", s=75, c=col, edgecolor="k", lw=np.where(diff, 2.0, .6), zorder=3)
        lab = np.arange(len(x)) if tt.startswith("test") else np.flatnonzero(f | diff | (np.abs(x - t_et) < 1))
        sub = f"ET 단독 고장 {int(et_only.sum())}/{len(x)} · 최종 고장 {int(f.sum())}/{len(x)} · 이름 = " + ("모든 파일" if tt.startswith("test") else "최종 고장 + 경계 ±1 안")
    lim = np.nanmax(np.abs(sw)) + 3 * D_SW
    ax.set_xlim(ex0, ex1); ax.set_ylim(-lim, lim); ax.set_yticks([]); ax.grid(axis="x", alpha=.25)
    if len(lab): place_labels(ax, fig, x[lab], sw[lab], [short(nmm[i]) for i in lab], x, sw)
    ax.set_title(f"{tt}  —  {sub}", fontsize=10, loc="left"); ax.legend(fontsize=8, loc="upper left")
axs[-1].set_xlabel("ET logit (ET 단독 점수, 박자 · AE 없음)", color=MUTED)
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="최종 고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="최종 정상 판정"),
       Line2D([], [], marker="o", ls="", mfc="#ddd", mec="k", mew=2, label="굵은 테두리 = ET 단독 판정과 다름"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상"), Line2D([], [], marker="^", ls="", color="#999", label="▲ 라벨 없음")]
fig.legend(handles=leg, ncol=7, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("EXP-062 구성 3 — ET 단독 점수 위의 학습 · val · test. 세로 위치는 겹치지 않게 쌓은 것(의미 없음). 배경 = ET 단독 판단",
             x=.01, ha="left", fontsize=11, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(OUT, "ET단독_학습_val_test.png"), dpi=130)

# %% test 요약 (제출 파일 아님)
S = [f"test {len(Dt)}개 · 고장 판정 {int(Dt.pred.sum())}개 · 규칙: 8.3k {int(Dt.rule83.sum())} · E2 {int(Dt.E2.sum())} · 결합만 {int((Dt.pred.astype(bool) & ~Dt.rule83 & ~Dt.E2).sum())}",
     "", "file | 판정 | 가로(ET+박자) | AE z | 여유 | 규칙"]
for r, xx in sorted(zip(Dt.itertuples(), G[2][1]), key=lambda p: -p[0].margin):
    S.append(f"{r.file} | {'고장' if r.pred else '정상'} | {xx:+.2f} | {r.ae_z:+.2f} | {r.margin:+.2f} | {('E2 ' if r.E2 else '') + ('8.3k' if r.rule83 else '') or '—'}")
open(os.path.join(OUT, "test_평면요약.txt"), "w", encoding="utf-8").write("\n".join(S) + "\n")
print("done")
