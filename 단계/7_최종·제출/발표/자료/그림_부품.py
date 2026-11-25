# %% 발표용 부품 그림 (발표_목차.md 5–12장): 기존 EDA · 실험 그림을 발표 화면용으로 다시 그림
# 다시 학습하지 않는다. 판정 값은 저장된 제출 모델과 표에서 읽는다
#   outputs/final/final_model.joblib (FinalJ, 학습 150 전체) · outputs/final/학습파일_재대입표.csv
#   단계/7_최종·제출/출력/근거표_submission_val_9조.csv · 단계/6_오류분석/6-7_1122_val확실도/표_val확실도.md
# 학습 평면의 결합 입력 = 모델의 안쪽 4등분 OOF(core_.Z_). EXP-050 logistic_plane과 같은 정의
#   배경 = 결합(로지스틱) 판단, 세로 점선 = ET 단독 판단(ET OOF logit에서 F2 최대 문턱), 점 색 = 최종 판정
import os, sys, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch, spectrogram
from scipy.ndimage import median_filter, uniform_filter1d
from src.exp import load_train
from src.impacts import detect_impacts, highpass
from src.final import impact_rings, P83_MIN, W83_MIN
from src.evaluate import best_f2_threshold
import src.viz  # noqa: F401  (맑은 고딕 · 마이너스 기호)

FS = 25000
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)
NAVY, INK, MUTED, GRID = "#0B2A4F", "#1F2937", "#6B7280", "#E5E7EB"
NOR, ABN = "#2a78d6", "#eb6834"
BG_N, BG_A = "#d6e6fa", "#fbe0d0"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
log = []


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.append(f"저장: figs/{name}.png")


def short(n):
    return n.split("_")[0][0].upper() + n.split("_")[1]          # nor_065 → N065, abn_002 → A002, val_025 → V025


X, y, names = load_train()
names = np.array(names)
ix = {n: i for i, n in enumerate(names)}
Xc = X - X.mean(1, keepdims=True)

# %% 5a. 큰 타격의 반복 — 정상의 큰 충격은 한 번, 고장은 비슷한 크기가 여러 번 (2-5)
def strikes(i):
    d = detect_impacts(Xc[i])
    xh = highpass(Xc[i])
    sig = 1.4826 * np.median(np.abs(xh))                       # 바탕 크기 (충격에 덜 끌려가는 강건 σ)
    return xh / sig, d


r21 = {}
for i in np.flatnonzero(y == 0):
    d = detect_impacts(Xc[i])
    h = np.sort(d["height_g"])[::-1]
    if len(h) >= 2 and h[0] > 0:
        r21[names[i]] = (h[1] / h[0], h[0])
nor_pick = min((k for k in r21 if r21[k][1] >= np.median([v[1] for v in r21.values()])), key=lambda k: r21[k][0])
fig, axs = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
for ax, n, c in zip(axs, [nor_pick, "abn_005"], [NOR, ABN]):
    s, d = strikes(ix[n])
    t = np.arange(len(s)) / FS
    ax.plot(t, s, color=c, lw=0.5)
    M = np.abs(s).max(); ax.set_ylim(-M * 1.1, M * 1.35)               # 음수 쪽 충격도 표시가 축 안에 들어오게
    h = d["height_g"]; o = np.argsort(h)[::-1][:2]
    ratio = h[o[1]] / h[o[0]] if len(o) > 1 else 0
    for k, j in enumerate(o):
        w = slice(max(0, d["idx"][j] - 75), d["idx"][j] + 75)            # 포락선 봉우리 ±3 ms 안의 실제 최댓값 위에 표시
        top = np.abs(s[w]).max()
        ax.annotate(f"{k + 1}위", (d["t"][j], top), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=13, fontweight="bold", color=INK)
    lab = "정상" if n.startswith("nor") else "고장"
    ax.set_title(f"{lab} {n[-3:]} — 2위 충격 / 1위 충격 = {ratio:.2f}", loc="left", fontsize=15, color=NAVY)
    ax.set_ylabel("배경 대비 배수")
    ax.grid(axis="y", color=GRID)
axs[1].set_xlabel("시간 (s) · 1 kHz 고역통과 신호")
fig.tight_layout()
save(fig, "5a_큰타격반복")
log.append(f"5a 정상 예시 = {nor_pick} (2위/1위 {r21[nor_pick][0]:.2f})")

# %% 5b. 충격이 울리는 주파수 — 정상은 4–8 kHz 한 곳, 고장은 2–4 kHz와 8 kHz 위 (2-7)
R = impact_rings(X)
fi, f = R[:, 0].astype(int), R[:, 1]
bins = np.arange(1000, 12251, 250)
fig, ax = plt.subplots(figsize=(13, 4.8))
for lab, c, nm in [(0, NOR, "정상"), (1, ABN, "고장")]:
    files = np.flatnonzero(y == lab)
    files = [i for i in files if (fi == i).any()]
    H = np.mean([np.histogram(f[fi == i], bins)[0] / (fi == i).sum() for i in files], axis=0)   # 파일마다 같은 무게
    ax.step(bins[:-1] / 1000, H, where="post", color=c, lw=2.2, label=f"{nm} ({len(files)}파일)")
    ax.fill_between(bins[:-1] / 1000, H, step="post", color=c, alpha=0.18)
for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 12.5)]:
    ax.axvline(lo, color=GRID, lw=1)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.97, f"{lo}–{hi} kHz", ha="center", va="top", fontsize=12, color=MUTED)
ax.set_xlim(1, 12.25)
ax.set_xlabel("충격의 울림 주파수 (kHz)")
ax.set_ylabel("파일 평균 비율")
ax.legend(frameon=False, loc="center left")
ax.grid(axis="y", color=GRID)
save(fig, "5b_울림주파수")

# %% 6. 박자 — 022는 느린 하중 흔들림 주기(T)에 맞춰 충격이 난다 (2-6)
i22 = ix["abn_022"]
lp = sosfiltfilt(butter(4, 30, btype="lowpass", fs=FS, output="sos"), Xc[i22])
dec = 25
v = lp[::dec] - lp[::dec].mean()
ac = np.correlate(v, v, "full")[len(v) - 1:]; ac /= ac[0]
lag = np.arange(len(ac)) * dec / FS
win = (lag >= 0.3) & (lag <= 1.2)
T = lag[win][ac[win].argmax()]
d = detect_impacts(Xc[i22])
tt = d["t"]
lock = np.array([np.any(np.abs(np.abs(tt - a) - T) <= 0.03 * T) for a in tt])
t = np.arange(X.shape[1]) / FS
fig, axs = plt.subplots(2, 1, figsize=(13, 5.8), sharex=True)
axs[0].plot(t, Xc[i22], color="#9CA3AF", lw=0.4, label="원신호")
axs[0].plot(t, lp, color=NAVY, lw=2.2, label="30 Hz 아래 느린 흔들림 (하중)")
axs[0].set_title(f"고장 022 — 느린 흔들림이 T = {T:.2f} s 마다 같은 모양으로 반복", loc="left", fontsize=15, color=NAVY)
axs[0].set_ylabel("g"); axs[0].legend(frameon=False, loc="upper right", fontsize=12, ncol=2)
hp = highpass(Xc[i22])
axs[1].plot(t, hp, color="#9CA3AF", lw=0.4)
for a, l in zip(tt, lock):
    axs[1].axvline(a, color=ABN if l else MUTED, lw=1.6 if l else 0.8)
axs[1].set_title(f"충격 {len(tt)}개 중 {lock.sum()}개가 T 떨어진 곳에 짝이 있다 (주황 선)", loc="left", fontsize=15, color=NAVY)
axs[1].set_ylabel("g (1 kHz 고역)"); axs[1].set_xlabel("시간 (s)")
axs[1].set_xlim(0, 2)
fig.tight_layout()
save(fig, "6_박자_022")

# %% 모델 · 표 읽기
m = joblib.load(os.path.join(ROOT, "outputs", "final", "final_model.joblib"))
core = m.core_
Dtr = pd.read_csv(os.path.join(ROOT, "outputs", "final", "학습파일_재대입표.csv"), encoding="utf-8-sig")
assert list(Dtr["file"]) == list(names)
Dv = pd.read_csv(os.path.join(ROOT, "단계", "7_최종·제출", "아카이브_J축소제출", "출력", "근거표_submission_val_9조.csv"), encoding="utf-8-sig")

# %% 7a. L2 위 꼬리 규칙(E2) — 하중 대비 고주파 L2_hfl(옛 H18), 문턱 = 학습 정상 99 % 분위
h18 = Dtr["L2_hfl"].values
e2 = Dtr["E2"].values.astype(bool)
fig, ax = plt.subplots(figsize=(13, 3.6))
rng = np.random.default_rng(0)
for lab, c, yy in [(0, NOR, 1), (1, ABN, 0)]:
    mk = y == lab
    ax.scatter(h18[mk], yy + rng.uniform(-0.18, 0.18, mk.sum()), s=34, color=c, edgecolor="white", lw=0.5, zorder=3)
ax.axvline(core.t18_, color=ABN, ls="--", lw=1.8)
ax.text(core.t18_, 1.45, f" 문턱 = 정상의 99 % 분위 ({core.t18_:.2f})", color=ABN, fontsize=13, va="center")
for k, i in enumerate(sorted(np.flatnonzero(e2), key=lambda i: h18[i])):
    ax.annotate(short(names[i]), (h18[i], 0), xytext=(0, -24 - 18 * (k % 2)), textcoords="offset points", ha="center", fontsize=12, color=ABN)
ax.set_yticks([0, 1]); ax.set_yticklabels(["고장 30", "정상 120"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("L2_hfl 하중 대비 고주파 = 1 kHz 위 덜컥거림 ÷ 20 Hz 아래 흔들림 (log)")
ax.grid(axis="x", color=GRID)
save(fig, "7a_E2_하중대비고주파")
log.append(f"7a E2 켜진 학습 파일: {[short(n) for n in names[e2]]} (정상 {int((e2 & (y == 0)).sum())})")

# %% 7b. 규칙 8.3k — 8.20–8.45 kHz 봉우리가 높고 넓게 솟는다 (2-2)
fq, P8 = welch(Xc, fs=FS, nperseg=8192)
L = np.log10(P8); dfq = fq[1] - fq[0]
Sm = uniform_filter1d(L, max(1, int(round(55 / dfq))), axis=1)
PR = Sm - median_filter(Sm, size=(1, int(round(1200 / dfq))), mode="nearest")
sel = (fq >= 7000) & (fq <= 10000)
fig, ax = plt.subplots(figsize=(13, 4.8))
q5, q50, q95 = np.percentile(PR[y == 0][:, sel], [5, 50, 95], axis=0)
ax.fill_between(fq[sel] / 1000, q5, q95, color=NOR, alpha=0.18, label="정상 120개 5–95 %")
ax.plot(fq[sel] / 1000, q50, color=NOR, lw=1.6, label="정상 중앙값")
for n, c in [("abn_002", ABN), ("abn_029", "#B4441B")]:
    ax.plot(fq[sel] / 1000, PR[ix[n], sel], color=c, lw=2.2, label=f"고장 {n[-3:]}")
ax.axvspan(8.20, 8.45, color="#FDE7DA", zorder=0)
ax.axhline(P83_MIN, color=ABN, ls="--", lw=1.4)
ax.text(9.95, P83_MIN + 0.04, f"규칙 문턱 {P83_MIN} dec (주변의 약 5.6배) · 폭 ≥ {W83_MIN:.0f} Hz · 충격성 K > 3", ha="right", fontsize=12, color=ABN)
ax.text(8.325, ax.get_ylim()[0] + 0.05, "8.20–8.45 kHz", ha="center", fontsize=12, color=MUTED)
ax.set_xlabel("주파수 (kHz)"); ax.set_ylabel("주변 대비 돌출 (dec)")
ax.legend(frameon=False, loc="upper left", fontsize=12, ncol=2)
ax.grid(axis="y", color=GRID)
save(fig, "7b_8.3k모드")

# %% 8 · 11b. 결합 평면 (EXP-050 logistic_plane과 같은 정의)
c, b, tc = core.c_.coef_[0], core.c_.intercept_[0], core.tc_
Z = core.Z_
t_et = float(best_f2_threshold(y, Z[:, 0]))
comb_tr = core.c_.decision_function(Z) >= tc
rule_tr = Dtr["E2"].values.astype(bool) | Dtr["rule83"].values.astype(bool)
fin_tr = comb_tr | rule_tr
fp_tr = (y == 0) & fin_tr
log.append(f"8 학습 OOF 평면: 결합 고장 {int((comb_tr & (y == 1)).sum())}/30 · 규칙만 {int(((y == 1) & ~comb_tr & rule_tr).sum())} · 오경보 {[short(n) for n in names[fp_tr]]} · ET 단독 문턱 {t_et:.2f}")


def plane(ax, et, zb, ae, fin, lab, marker, label_set, title):
    x = et + c[1] / c[0] * zb
    x0, x1 = min(x.min(), t_et) - .6, max(x.max(), t_et) + .6
    y0, y1 = ae.min() - .6, ae.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
    ax.contourf(gx, gy, (c[0] * gx + c[2] * gy + b >= tc).astype(float), levels=[-.5, .5, 1.5], colors=[BG_N, BG_A], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.4, ls="--")
    col = np.where(fin, ABN, NOR)
    if lab is None:
        ax.scatter(x, ae, marker=marker, s=90, c=col, edgecolor="k", lw=.6, zorder=3)
    else:
        for L_, mk in [(0, "s"), (1, "o")]:
            k = lab == L_
            ax.scatter(x[k], ae[k], marker=mk, s=60 if L_ == 0 else 75, c=col[k], edgecolor="k", lw=.5, zorder=3)
    for k, (i, n) in enumerate(sorted(label_set, key=lambda p: -ae[p[0]])):
        ax.annotate(n, (x[i], ae[i]), xytext=(8, 6 if k % 2 == 0 else -16), textcoords="offset points", fontsize=12, fontweight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel(f"ET 점수 + 박자 몫  (ET logit + {c[1] / c[0]:.2f} × 박자 z)")
    ax.set_ylabel("AE z (정상 스펙트럼에서 벗어난 정도)")
    ax.set_title(title, loc="left", fontsize=15, color=NAVY)


# 학습: 규칙만으로 잡힌 고장은 뺀다 (결정.md 기본 그림 규칙)
show = ~((y == 1) & ~comb_tr & rule_tr)
xs = Z[:, 0] + c[1] / c[0] * Z[:, 1]
rescued = (y == 0) & (Z[:, 0] >= t_et) & ~comb_tr                  # ET만이면 오경보, 결합에서 정상
labs = [(k, short(names[i])) for k, i in enumerate(np.flatnonzero(show)) if rescued[i] or fp_tr[i]]
fig, ax = plt.subplots(figsize=(12, 7))
plane(ax, Z[show, 0], Z[show, 1], Z[show, 2], fin_tr[show], y[show], None, labs,
      f"점선 오른쪽 = ET만 쓰면 고장 · 파란 배경 = AE를 더한 결합에서 정상")
from matplotlib.lines import Line2D
leg = [Line2D([], [], marker="s", ls="", mfc=NOR, mec="k", ms=9, label="정상 파일 → 정상 판정"),
       Line2D([], [], marker="s", ls="", mfc=ABN, mec="k", ms=9, label="정상 파일 → 오경보"),
       Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", ms=10, label="고장 파일 → 검출"),
       Line2D([], [], color=INK, ls="--", label=f"ET 단독 경계")]
ax.legend(handles=leg, frameon=True, framealpha=.9, loc="upper left", fontsize=12)
save(fig, "8_AE_평면")
log.append(f"8 ET만이면 오경보인데 결합에서 정상: {[short(n) for n in names[rescued]]}")

# %% 9. 울림 비율 축소 — 충격이 적은 파일의 비율을 정상 쪽으로 당김 (EXP-050)
p0 = core.p0_
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
n = np.arange(1, 21)
ax = axs[0]
ax.plot(n, np.ones_like(n, float), color=MUTED, lw=2, ls="--", label="축소 전: k / n = 1")
ax.plot(n, (n + 3 * p0[1]) / (n + 3), color=NAVY, lw=2.6, marker="o", ms=6, label=f"축소 후: (k + 3·p0) / (n + 3)")
ax.axhline(p0[1], color=NOR, lw=1.2)
ax.text(20, p0[1] + 0.03, f"정상 기준 p0 = {p0[1]:.3f}", ha="right", fontsize=12, color=NOR)
ax.set_xlabel("충격 수 n (모두 2–4 kHz로 울렸을 때, k = n)")
ax.set_ylabel("2–4 kHz 울림 비율")
ax.set_title("충격 1–3개뿐이면 비율을 그만큼 덜 믿는다", loc="left", fontsize=15, color=NAVY)
ax.legend(frameon=False, loc="center right", fontsize=12)
ax.set_ylim(0, 1.1); ax.grid(axis="y", color=GRID)
ax = axs[1]
# 11-23: 001이 8.3k 규칙(K83 조건 추가)에서 빠져 모든 부트스트랩 판정이 정확히 1개씩 줄었다. 001은 규칙으로만 고장이었고
#        결합 점수는 25/25 정상(6-7)이라 재계산 없이 8–21 → 7–20, 8–14 → 7–13 (규칙은 재표집과 무관)
rows = [("J (축소 전)", 7, 20), ("J-축소 (제출)", 7, 13)]
for k, (nm, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [k, k], color=ABN if k else MUTED, lw=10, solid_capstyle="round")
    ax.text(hi + 0.5, k, f"{lo}–{hi}개", va="center", fontsize=14, fontweight="bold" if k else "normal")
ax.axvline(9, color=INK, lw=1.2, ls=":")
ax.text(9, 1.45, "제출 판정 9개", ha="center", fontsize=12, color=INK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(-0.6, 1.7); ax.set_xlim(4, 23)
ax.set_xlabel("val 고장 판정 수 (결합기 부트스트랩 300회, 5–95 %)")
ax.set_title("F2는 0.977 그대로, val 판정 흔들림은 줄었다", loc="left", fontsize=15, color=NAVY)
ax.grid(axis="x", color=GRID)
fig.tight_layout()
save(fig, "9_울림비율축소")

# %% 10. 정직한 점수 — 낙관 요인을 따로 보여 준다
rows = [  # 이름, 값들, sd, 메모
    ("25년 3조 방식 (우리 데이터)", [0.538], 0, "EXP-003"),
    ("J에서 8.3k 규칙 뺌 (사후 문턱 없음)", [0.950], 0.009, "EXP-044 L"),
    ("J-축소 (제출)", [0.977], 0.010, "EXP-050 · 8.3k 문턱은 사후값"),
    ("리허설 3회 (학습 40개를 val로 가정)", [0.976, 0.930, 0.976], 0, "고장 8개 모두 검출 · 오경보 1 · 3 · 1"),
]
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.axvspan(0.950, 0.977, color="#FDE7DA", zorder=0)
ax.text(0.9635, len(rows) - 0.35, "새 데이터 기대 범위", ha="center", fontsize=12, color=ABN)
for k, (nm, vals, sd, note) in enumerate(rows[::-1]):
    col = ABN if "제출" in nm else NAVY
    if sd:
        ax.errorbar(vals[0], k, xerr=sd, color=INK, capsize=5, lw=1.2, zorder=2)
    ax.scatter(vals, [k] * len(vals), s=110, color=col, edgecolor="white", lw=1, zorder=3)
    txt = " · ".join(f"{v:.3f}" for v in vals) + (f" ± {sd:.3f}" if sd else "")
    ax.text(1.005, k, f"{txt}   {note}", va="center", fontsize=12, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xlabel("F2 (중첩 교차검증 · 리허설은 한 번씩)")
ax.grid(axis="x", color=GRID)
save(fig, "10_정직한점수")

# %% 11a. val 확실도 — 학습 분할만 바꾼 모델 25개의 판정 (6-7)
md = open(os.path.join(ROOT, "단계", "6_오류분석", "6-4_val확실도", "자료", "표_val확실도.md"), encoding="utf-8").read()
V = [r for r in (re.match(r"\| (\d{3}) \| (고장|정상) \| ([-\d.]+) \| (\d+) \|", l) for l in md.splitlines()) if r]
V = pd.DataFrame([(r[1], r[2], float(r[3]), int(r[4])) for r in V], columns=["val", "판정", "여유", "표"])
assert len(V) == 40, len(V)
# 11-23: 6-7 표는 8.3k 규칙 K83 조건(2-16) 이전 값. 001은 규칙으로만 고장이었고(결합 25/25 정상) 지금은 정상 → 모델 25개 모두 정상(0표)
_i001 = V["val"] == "001"
assert (V.loc[_i001, "표"] == 25).all(), "6-7 표의 001 행이 예상과 다르다"
V.loc[_i001, ["판정", "표"]] = ["정상", 0]
V = V.sort_values(["표", "여유"], ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 4.6))
ax.bar(range(40), V["표"], color=np.where(V["판정"] == "고장", ABN, NOR), width=0.75, zorder=2)
ax.bar(range(40), np.where(V["표"] == 0, 0.35, 0), color=NOR, width=0.75, zorder=2)        # 0표도 보이게
for k, r in V.iterrows():
    if r["val"] in ("001", "025", "018", "027"):
        ax.text(k, r["표"] + 0.6, r["val"], ha="center", fontsize=12, fontweight="bold")
unan = int(((V["표"] == 25) & (V["판정"] == "고장")).sum() + ((V["표"] == 0) & (V["판정"] == "정상")).sum())
n_pos = int((V["판정"] == "고장").sum())
ax.set_xticks([]); ax.set_ylim(0, 28)
ax.set_ylabel("모델 25개 중 고장 판정 수")
ax.set_title(f"val 40개 중 {unan}개는 모델 25개가 만장일치", loc="left", fontsize=15, color=NAVY)
ax.grid(axis="y", color=GRID)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ABN, label=f"제출 판정 고장 ({n_pos})"),
                   plt.Rectangle((0, 0), 1, 1, color=NOR, label=f"제출 판정 정상 ({40 - n_pos})")], frameon=False, loc="upper right", fontsize=12)
save(fig, "11a_val확실도")
log.append(f"11a 25개 같은 판정 {unan} · 제출 고장 {int((V['판정'] == '고장').sum())}")

# %% 11b. val 평면 (학습 150 전체 모델)
fin_v = Dv["pred"].values.astype(bool)
labs_v = [(i, short(n)) for i, n in enumerate(Dv["file"]) if n in ("val_001", "val_025", "val_018", "val_027")]
fig, ax = plt.subplots(figsize=(12, 7))
plane(ax, Dv["et_logit"].values, Dv["beat_z"].values, Dv["ae_z"].values, fin_v, None, "^", labs_v,
      "val 40개 — 학습과 같은 결합 경계 (▲ 주황 = 고장 판정)")
save(fig, "11b_val평면")

# %% 12. 상시 오경보 정상 — 2초 내내 이어지는 좁은 고주파 지속음 (2-9)
import json
from src.exp import find_exp
FPC = json.load(open(os.path.join(find_exp("EXP-050"), "result.json"), encoding="utf-8"))["configs"]["J-축소"]["fp_counts"]   # 중첩 CV seed 5회 중 오경보 횟수
show_f = ["nor_116", "nor_065", "nor_059", "nor_001"]
hp200 = butter(4, 200, btype="highpass", fs=FS, output="sos")
fig, axs = plt.subplots(len(show_f), 1, figsize=(13, 9), sharex=True)
for ax, n in zip(axs, show_f):
    fr, tt_, Sxx = spectrogram(sosfiltfilt(hp200, Xc[ix[n]]), fs=FS, nperseg=512, noverlap=384)
    kk = fr >= 6000
    Lg = np.log10(Sxx[kk] + 1e-20)
    ax.pcolormesh(tt_, fr[kk] / 1000, Lg, shading="auto", cmap="magma",
                  vmin=np.percentile(Lg, 5), vmax=np.percentile(Lg, 99.5), rasterized=True)
    med = np.median(Lg, axis=1); band = (fr[kk] >= 9000) & (fr[kk] <= 11500)
    pk = fr[kk][band][med[band].argmax()] / 1000
    fpn = FPC.get(n, 0)
    tag = f"중첩 CV 오경보 5회 중 {fpn}회 · {pk:.1f} kHz 지속음" if fpn else "오경보 0회 (비교용 보통 정상)"
    ax.set_title(f"정상 {n[-3:]} — {tag}", loc="left", fontsize=14, color=NAVY)
    ax.set_ylabel("kHz")
axs[-1].set_xlabel("시간 (s)")
fig.tight_layout()
save(fig, "12_고주파지속음")

open(os.path.join(FIG, "_log_부품.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
