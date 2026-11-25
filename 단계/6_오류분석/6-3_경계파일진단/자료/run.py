# %% [markdown]
# 6-4 (11-22) J 경계 파일 진단 — 충격 수 × 낮은 울림 비율
# - 가로 = 충격 수(H1_nimp = log1p(n)에서 되돌림), 세로 = 4 kHz 아래로 울린 충격 비율(= H7 1–2k + H8 2–4k, ET가 보는 값 그대로)
# - train 정상 □ · 고장 ○ · val △(라벨 없음). 경계 파일 13개(EXP-044 J, 그림6 파일별 여유범위에서 0을 가로지르는 파일)는 이름 표시
# - 충격 0개 파일은 울림 비율이 정의되지 않아(−1) 뺀다
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp2util import NS
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575", "#d9dde3"
BOUND = ["abn_009", "nor_065", "nor_037", "nor_095", "nor_110", "nor_106", "nor_073",
         "val_025", "val_018", "val_021", "val_012", "val_036", "val_037"]

ns = NS(); y = ns.y; names = np.array(ns.names); nv = np.array([f"val_{n[-3:]}" for n in ns.nv])
def parts(T):
    n = np.rint(np.expm1(T["H1_nimp"].values)).astype(int); low = (T["H7_ring_1-2k"] + T["H8_ring_2-4k"]).values
    return n, low
n, low = parts(ns.T); nv_, lowv = parts(ns.Tv)

# %% 그림
rng = np.random.default_rng(0)
jx = lambda k: k * np.exp(rng.uniform(-.07, .07, len(k))); jy = lambda v: np.clip(v + rng.uniform(-.018, .018, len(v)), -.03, 1.03)
fig, ax = plt.subplots(figsize=(11.5, 6.6))
ax.axvspan(.75, 3.5, color="#eef0f3", zorder=0)
m = n > 0; mv = nv_ > 0
few = lambda msk, k: int((msk & (k >= 1) & (k <= 3)).sum())
ax.text(1.6, 1.085, f"충격 1–3개: 정상 {few(y == 0, n)} · 고장 {few(y == 1, n)} · val {few(np.ones(len(nv_), bool), nv_)}", ha="center", fontsize=9.5, color=MUTED)
X, Y = jx(n.astype(float)), jy(low); Xv, Yv = jx(nv_.astype(float)), jy(lowv)
ax.scatter(X[m & (y == 0)], Y[m & (y == 0)], marker="s", s=34, color=NOR, edgecolor="white", lw=.8, alpha=.85, label=f"train 정상 ({int((m & (y == 0)).sum())})", zorder=3)
ax.scatter(X[m & (y == 1)], Y[m & (y == 1)], marker="o", s=44, color=ABN, edgecolor="white", lw=.8, alpha=.9, label=f"train 고장 ({int((m & (y == 1)).sum())})", zorder=3)
ax.scatter(Xv[mv], Yv[mv], marker="^", s=62, color=VAL, edgecolor="white", lw=.8, alpha=.9, label=f"val ({int(mv.sum())}, 라벨 없음)", zorder=4)
# 경계 파일: 검은 테두리 + 이름
for nm in BOUND:
    if nm.startswith("val"):
        i = list(nv).index(nm); x, yy, mk, c = Xv[i], Yv[i], "^", VAL
    else:
        i = list(names).index(nm); x, yy, mk, c = X[i], Y[i], ("o" if y[i] else "s"), (ABN if y[i] else NOR)
    ax.scatter([x], [yy], marker=mk, s=120, facecolor=c, edgecolor=INK, lw=1.6, zorder=5)
    lab = nm.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
    off = {"N037": (8, 8), "N110": (8, -14), "N073": (-36, 10), "N065": (8, -14), "V021": (8, 6)}.get(lab, (7, 6))
    ax.annotate(lab, (x, yy), xytext=off, textcoords="offset points", fontsize=9, color=INK, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.75))
# 구간별 "4 kHz 아래 울림이 절반 이상"인 파일 비율 (정상 · 고장 · val)
txt = []
for lo, hi in ((1, 3), (4, 6), (7, 999)):
    def fr(k, v, msk):
        g = v[msk & (k >= lo) & (k <= hi)]; return f"{int((g >= .5).sum())}/{len(g)}"
    txt.append(f"충격 {f'{lo}–{hi}개' if hi < 999 else f'{lo}개 이상'}:  정상 {fr(n, low, y == 0)} · 고장 {fr(n, low, y == 1)} · val {fr(nv_, lowv, np.ones(len(nv_), bool))}")
ax.text(.99, .80, "4 kHz 아래 울림이 절반 이상인 파일\n" + "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=INK,
        bbox=dict(boxstyle="round,pad=.45", fc="white", ec=GRID))
ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50"])
ax.set_xlim(.75, 75); ax.set_ylim(-.06, 1.13)
ax.set_xlabel("충격 수 (새 충격 정의, 로그 눈금)", color=MUTED); ax.set_ylabel("4 kHz 아래로 울린 충격 비율 (H7 + H8)", color=MUTED)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
ax.tick_params(colors=MUTED); ax.grid(color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h_, l_ = ax.get_legend_handles_labels()
h_.append(Line2D([], [], marker="o", ls="", ms=10, mfc="white", mec=INK, mew=1.6)); l_.append("J 경계 파일 13개 (굵은 테두리)")
ax.legend(h_, l_, loc="upper right", bbox_to_anchor=(1, 1.0), fontsize=9, frameon=False)
ax.set_title("충격 1–3개에서 4 kHz 아래로 절반 이상 울린 파일: train 정상 21 % · val 53 %. 고장은 충격이 거의 항상 4개 이상",
             loc="left", fontsize=11.5, color=INK, pad=12)
fig.text(.01, .005, "점은 겹침을 피하려 살짝 흔듦(가로 ±7 %, 세로 ±0.02). 충격 0개 파일(정상 17 · val 4)은 비율이 없어 뺌. 경계 파일 = EXP-044 J에서 폴드 경계에 따라 판정이 바뀌는 파일",
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, .02, 1, 1)); fig.savefig(os.path.join(FIG, "충격수_낮은울림.png"), dpi=150); plt.close(fig)
print("done")
