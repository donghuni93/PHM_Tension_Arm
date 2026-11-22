# EXP-050 팀원 공유용 그림 (11-22). 실험 결과(result.json · figs/)와 학습 피처(ns.T)만 읽는다. 모델은 다시 학습하지 않는다
# 실행: conda run -n qe-phm python 그림_공유.py
import os, sys, json, shutil
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from src.exp2util import NS
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575", "#d9dde3"
A = 3.0
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.grid(color=GRID, lw=.6, alpha=.7); ax.set_axisbelow(True)

# ---------- 그림 1: 문제 (6-4 그림 그대로) ----------
shutil.copy(os.path.join(_d, "단계", "6_오류분석", "6-4_1122_경계파일진단", "figs", "충격수_낮은울림.png"), os.path.join(HERE, "그림1_문제_충격수_낮은울림.png"))

# ---------- 그림 2: 축소 원리 ----------
ns = NS(); y = ns.y
def parts(T):
    n = np.rint(np.expm1(T["H1_nimp"].values)).astype(int); low = (T["H7_ring_1-2k"] + T["H8_ring_2-4k"]).values; return n, low
n, low = parts(ns.T); nv, lowv = parts(ns.Tv)
R = ["H7_ring_1-2k", "H8_ring_2-4k"]
m0 = (y == 0) & (n > 0); p0 = float((ns.T.loc[m0, R].sum(1).values * n[m0]).sum() / n[m0].sum())
shr = lambda k, r: (r * k + A * p0) / (k + A)
rng = np.random.default_rng(0); jx = lambda k: k * np.exp(rng.uniform(-.07, .07, len(k))); jy = lambda v: np.clip(v + rng.uniform(-.018, .018, len(v)), -.03, 1.03)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25, 1.25]))
ax = axs[0]; k = np.arange(1, 61)
ax.plot(k, k / (k + A), color=INK, lw=2)
for kk in (1, 3, 10, 30):
    w = kk / (kk + A); ax.scatter([kk], [w], color=INK, s=30, zorder=3)
    ax.annotate(f"충격 {kk}개 → 자기 값 {w*100:.0f} %", (kk, w), xytext=(8, -14 if kk > 1 else 6), textcoords="offset points", fontsize=9, color=INK)
ax.set_xscale("log"); ax.set_xticks([1, 3, 10, 30]); ax.set_xticklabels(["1", "3", "10", "30"]); ax.set_ylim(0, 1.02)
ax.set_xlabel("충격 수 n (로그 눈금)", color=MUTED); ax.set_ylabel("축소 뒤 값에서 파일 자기 비율의 몫  n / (n + α)", color=MUTED)
ax.set_title("(a) α = 3: 충격이 적을수록 정상 기준 p0 쪽으로", loc="left", fontsize=11, color=INK); style(ax)
for j, (lab, L, Lv) in enumerate((("(b) 축소 전 (J)", low, lowv), ("(c) 축소 뒤 (J-축소)", np.where(n > 0, shr(n, low), low), np.where(nv > 0, shr(nv, lowv), lowv)))):
    ax = axs[j + 1]; m, mv = n > 0, nv > 0; X, Y, Xv, Yv = jx(n.astype(float)), jy(L), jx(nv.astype(float)), jy(Lv)
    ax.axhline(p0, color=MUTED, ls="--", lw=1); ax.text(55, p0 + .02, f"p0 = {p0:.2f}", ha="right", fontsize=9, color=MUTED)
    ax.scatter(X[m & (y == 0)], Y[m & (y == 0)], marker="s", s=26, color=NOR, edgecolor="white", lw=.6, alpha=.85, label="train 정상")
    ax.scatter(X[m & (y == 1)], Y[m & (y == 1)], marker="o", s=34, color=ABN, edgecolor="white", lw=.6, alpha=.9, label="train 고장")
    ax.scatter(Xv[mv], Yv[mv], marker="^", s=44, color=VAL, edgecolor="white", lw=.6, alpha=.9, label="val (라벨 없음)")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20, 50]); ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50"]); ax.set_xlim(.75, 75); ax.set_ylim(-.05, 1.08)
    ax.set_xlabel("충격 수 (로그 눈금)", color=MUTED); ax.set_ylabel("4 kHz 아래로 울린 충격 비율 (H7 + H8)", color=MUTED)
    ax.set_title(lab, loc="left", fontsize=11, color=INK); style(ax)
    if j == 0: ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.text(.01, .01, f"p0 = train 정상 전체 충격에서 4 kHz 아래로 운 비율({p0:.2f}). 실험에서는 대역 4개(H7–H10)를 각각 축소하고, p0는 폴드마다 그 폴드 학습 정상으로 다시 잰다. 충격 0개 파일은 뺐다(값 -1 그대로). 점은 겹침을 피하려 살짝 흔듦",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(os.path.join(HERE, "그림2_축소원리.png"), dpi=150); plt.close(fig)

# ---------- 그림 3: 결합 평면 J vs J-축소 (실험 그림의 왼쪽 열) ----------
im = Image.open(os.path.join(EXP, "figs", "logistic_plane.png")); W, H = im.size
im.crop((0, int(H * .055), W // 2, H)).save(os.path.join(HERE, "그림3_결합평면_J_vs_J축소.png"))

# ---------- 그림 4: 점수와 흔들림 ----------
res = json.load(open(os.path.join(EXP, "result.json"), encoding="utf-8")); C = res["configs"]; N = res["notes"]["흔들림 (a)(b) · 결합 계수 · p0"]
order = ["J", "J-축소", "J−H10", "J−H10+축소"]; lab = {"J": "J (기준)", "J-축소": "J-축소 (채택 방향)", "J−H10": "J - H10", "J−H10+축소": "J - H10 + 축소"}
col = {"J": INK, "J-축소": NOR, "J−H10": "#9aa3b0", "J−H10+축소": "#9aa3b0"}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axs[0]
for i, c in enumerate(order):
    f, s = C[c]["f2"], C[c]["f2_sd"]; ax.errorbar([f], [i], xerr=[s], fmt="o", color=col[c], ms=8, capsize=4, lw=2)
    ax.text(f + s + .002, i, f"{f:.3f} ± {s:.3f}", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([lab[c] for c in order]); ax.invert_yaxis(); ax.set_xlim(.94, 1.0)
ax.set_xlabel("중첩 CV F2 (seed 5 평균 ± 표준편차)", color=MUTED); ax.set_title("(a) 점수: 축소는 J와 같고, H10을 빼면 떨어진다", loc="left", fontsize=11, color=INK); style(ax)
ax = axs[1]
for i, c in enumerate(order):
    lo, hi = N[c]["결합만_val고장_5_95"]; ax.plot([lo, hi], [i, i], color=col[c], lw=7, solid_capstyle="round")
    ax.text(hi + .4, i, f"{lo:.0f} – {hi:.0f}개", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([lab[c] for c in order]); ax.invert_yaxis(); ax.set_xlim(5, 25)
ax.set_xlabel("val 40개 중 결합 판정 고장 수, 부트스트랩 300회 5–95 % 범위", color=MUTED)
ax.set_title("(b) 흔들림: 축소하면 val 판정 범위가 좁아진다", loc="left", fontsize=11, color=INK); style(ax)
fig.text(.01, .01, "(b) 전체 학습 모델의 결합 입력(ET · 박자 · AE)을 300번 다시 뽑아 로지스틱과 문턱을 다시 맞췄을 때 val 고장 판정 수가 움직이는 범위(규칙으로 켜진 val 포함). 실제 val 판정은 J · J-축소 모두 10개",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(os.path.join(HERE, "그림4_점수_흔들림.png"), dpi=150); plt.close(fig)
print("p0", p0)
