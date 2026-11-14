# %% [markdown]
# 2-11 경계 파일의 특징 — 로지스틱 구분선 위의 정상, 선 근처의 고장(002·029 등)은 무엇이 다른가
# - 출발: EXP-018 boundary_lr (AE 기본 + ET 기본, 로지스틱, OOF seed 0 분할)
# - 목표 (사용자 11-14): ① 선 위 정상을 아래로 내릴 특징 ② 002·029를 고장 본진 쪽으로 옮길 특징 ③ 017·022는 배제
# - 그룹 (실행 전 고정): 결정값 d = 로지스틱 결정값 (OOF)
#   정상 경계 = 정상 & d > 0 · 정상 아래 = 나머지 정상 · 고장 경계 = 고장 & d < 1 (017·022 제외) · 고장 본진 = 나머지 고장
# - 라벨은 그룹 나누기·색칠·설명에만. 여기서 본 차이를 피처로 쓰려면 라벨 없는 정의로 다시 만들어 CV로 확인 (원칙 3)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-11_경계파일특징_8.25khz/run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import *
from src.experts import ScoreStack, AEScore, ETScore, LP, F_PSD
from src.features import feature_table_impact
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y))
ED = cached_table("ext", feature_table_ext, X); FN = list(ED.columns); E = ED.values
IMP = cached_table("impact", feature_table_impact, X)
LOG = []
def log(*a): LOG.append(" ".join(str(x) for x in a)); print(*a, flush=True)
def save(fig, name): fig.savefig(os.path.join(FIG, name + ".png"), dpi=80); plt.close(fig)
C = {"정상 아래": "#9cc3ec", "정상 경계": "#1f5fae", "고장 경계": "#f2a37d", "고장 본진": "#d4501e", "배제": "#9aa5b1"}
EXCL = ["abn_017", "abn_022"]


# %% 그룹: EXP-018 boundary_lr과 같은 OOF z·결정값
AE0, ET0 = AEScore(), ETScore()
Z, D = np.empty((len(y), 2)), np.empty(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = ScoreStack((AE0, ET0), combine="lr").fit(g[tr][:, None], y[tr])
    Z[te] = m.zscores(g[te][:, None]); D[te] = m.decision_function(g[te][:, None])
grp = np.empty(len(y), dtype=object)
for i, n in enumerate(names):
    if n in EXCL: grp[i] = "배제"
    elif y[i] == 0: grp[i] = "정상 경계" if D[i] > 0 else "정상 아래"
    else: grp[i] = "고장 경계" if D[i] < 1 else "고장 본진"
G = {k: [names[i] for i in np.flatnonzero(grp == k)] for k in C}
for k, v in G.items():
    log(f"{k} ({len(v)}): " + " ".join(n.split('_')[1] for n in v))
nor_ref = grp == "정상 아래"                       # 기준 정상
core = grp == "고장 본진"
BN = G["정상 경계"]; BF = G["고장 경계"]
pct = lambda v, ref: (ref[None, :] < v[:, None]).mean(1) * 100   # 기준 분포에서 몇 % 위치


# %% 그림 1: 그룹 지도 (boundary_lr과 같은 평면)
fig, ax = plt.subplots(figsize=(9.5, 8))
zc = np.clip(Z, -3, 4)
for k, c in C.items():
    s = grp == k
    ax.scatter(*zc[s].T, s=40 if k != "정상 아래" else 18, color=c, label=f"{k} ({s.sum()})", lw=0)
for n in BN + BF + EXCL:
    i = names.index(n); ax.annotate(n.replace("abn_", "A").replace("nor_", "N"), zc[i], xytext=(4, 3),
                                    textcoords="offset points", fontsize=8, color=C[grp[i]])
ax.axvline(1, color="k", lw=.5, ls=":"); ax.axhline(1, color="k", lw=.5, ls=":")
ax.set_xlim(-3, 4); ax.set_ylim(-3, 4); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax.set_xlabel("AE z (1 = 정상 90%선)"); ax.set_ylabel("ET z (1 = 정상 90%선)")
ax.set_title("2-11 그룹 (EXP-018 OOF seed 0). 정상 경계 = 로지스틱 선 위 정상, 고장 경계 = 결정값 < 1 고장", fontsize=10)
fig.tight_layout(); save(fig, "1_groups")


# %% ① ET 쪽: 선 위 정상은 어떤 ET 피처에서 고장처럼 보이나 (ET 전체 학습 중요도 상위 20, 설명용)
et = make_et()().fit(E, y)
imp = et.feature_importances_
top = np.argsort(imp)[::-1][:20]
P = np.array([pct(E[:, j], E[nor_ref, j]) for j in top]).T          # (파일, 20) 기준 정상 중 %
rows = BN + BF
M = np.vstack([P[[names.index(n) for n in rows]], np.median(P[core], 0), np.median(P[nor_ref], 0)])
labels = [n.replace("abn_", "고장 ").replace("nor_", "정상 ") for n in rows] + ["고장 본진 중앙", "정상 아래 중앙"]
fig, ax = plt.subplots(figsize=(15, 0.42 * len(labels) + 2))
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="w" if abs(M[i, j] - 50) > 35 else "k")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xticks(range(len(top))); ax.set_xticklabels([FN[j] for j in top], rotation=60, ha="right", fontsize=7)
ax.axhline(len(BN) - .5, color="k", lw=1); ax.axhline(len(rows) - .5, color="k", lw=1)
ax.set_title("ET 중요도 상위 20 피처에서 기준 정상(선 아래 정상) 중 몇 % 위치인가. 빨강 = 고장 쪽(높음)", fontsize=10)
fig.colorbar(im, ax=ax, shrink=.5, label="기준 정상 중 %"); fig.tight_layout(); save(fig, "2_et_features")
hi = (P >= 90)
log(f"\n[①] ET 상위 20 피처 중 기준 정상 90% 이상인 개수 (고장 본진 중앙 = {np.median(hi[core].sum(1)):.0f})")
for n in rows:
    i = names.index(n)
    fam = [FN[top[j]] for j in np.flatnonzero(hi[i])]
    log(f"  {n}: {hi[i].sum():2d}개 — {' '.join(fam[:8])}{' …' if len(fam) > 8 else ''}")
log("  ET 상위 20: " + " ".join(f"{FN[j]}({imp[j]:.3f})" for j in top))


# %% ② AE 쪽: 복원 오차가 어느 대역에서 나나 (OOF, 기준 정상 중앙값 대비 초과 오차의 대역별 몫)
A = np.empty((len(y), LP.shape[-1]))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    A[te] = np.log10(AE0.bins(tuple(tr.tolist()), tuple(te.tolist())) + 1e-12)
EXC = np.clip(A - np.median(A[nor_ref], 0), 0, None)
BANDS = [(200, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 8800), (8800, 10500), (10500, 12500)]
SH = np.column_stack([EXC[:, (F_PSD >= lo) & (F_PSD < hi)].sum(1) for lo, hi in BANDS]) / (EXC.sum(1, keepdims=True) + 1e-12) * 100
WID = np.array([((F_PSD >= lo) & (F_PSD < hi)).mean() * 100 for lo, hi in BANDS])
M = np.vstack([SH[[names.index(n) for n in rows]], np.median(SH[core], 0), np.median(SH[nor_ref], 0), WID])
labels2 = labels + ["(대역 폭 비율 = 균등할 때)"]
fig, ax = plt.subplots(figsize=(12, 0.42 * len(labels2) + 2))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=50, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="w" if M[i, j] > 35 else "k")
ax.set_yticks(range(len(labels2))); ax.set_yticklabels(labels2, fontsize=8)
ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([f"{lo/1000:g}–{hi/1000:g} kHz" for lo, hi in BANDS], fontsize=8)
ax.axhline(len(BN) - .5, color="k", lw=1); ax.axhline(len(rows) - .5, color="k", lw=1)
ax.set_title("AE 초과 복원 오차의 대역별 몫 (%). 균등하면 맨 아래 줄과 같다 — 한 대역에 몰리면 그 대역이 원인", fontsize=10)
fig.colorbar(im, ax=ax, shrink=.5); fig.tight_layout(); save(fig, "3_ae_bands")
log("\n[②] AE 초과 오차 몫이 가장 큰 대역 (몫 %, 대역 폭 대비 배수) · AE z")
for n in rows + ["abn_017", "abn_022"]:
    i = names.index(n); j = int(np.argmax(SH[i] / WID))
    log(f"  {n}: {BANDS[j][0]/1000:g}–{BANDS[j][1]/1000:g} kHz {SH[i, j]:.0f}% (×{SH[i, j] / WID[j]:.1f}) · AE z {Z[i, 0]:+.2f} · ET z {Z[i, 1]:+.2f}")


# %% ③·④ 충격 구조와 시간 구조: 단발 큰 사건인가, 2초 내내 반복인가
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=25000, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
WR = np.sqrt((H[:, :50000].reshape(len(y), 20, 2500) ** 2).mean(2))     # 0.1초 창 20개 RMS
WRn = WR / np.median(WR, 1, keepdims=True)
ACT = (WRn > 2).sum(1)                                              # 중앙 창의 2배 넘는 창 수
TOPW = np.sort(WRn, 1)[:, ::-1]
SECOND = TOPW[:, 1] / TOPW[:, 0]                                    # 두 번째로 큰 창 / 가장 큰 창
fig, axs = plt.subplots(1, 3, figsize=(19, 6))
for k, c in C.items():
    s = grp == k
    axs[0].scatter(IMP["E4_big_height_log"][s], IMP["E2_top2_ratio"][s], color=c, s=40 if k != "정상 아래" else 16, lw=0, label=k)
    axs[1].scatter(TOPW[s, 0], ACT[s] + np.random.default_rng(0).uniform(-.2, .2, s.sum()), color=c, s=40 if k != "정상 아래" else 16, lw=0)
    axs[2].scatter(IMP["E1_n_impact_log"][s], IMP["E3_big_ringing_log"][s], color=c, s=40 if k != "정상 아래" else 16, lw=0)
for n in rows + EXCL:
    i = names.index(n); t = n.replace("abn_", "A").replace("nor_", "N")
    axs[0].annotate(t, (IMP["E4_big_height_log"][i], IMP["E2_top2_ratio"][i]), fontsize=7, color=C[grp[i]])
    axs[1].annotate(t, (TOPW[i, 0], ACT[i]), fontsize=7, color=C[grp[i]])
    axs[2].annotate(t, (IMP["E1_n_impact_log"][i], IMP["E3_big_ringing_log"][i]), fontsize=7, color=C[grp[i]])
axs[0].set_xlabel("가장 큰 사건 높이 log10(σ배)"); axs[0].set_ylabel("두 번째 / 첫 번째 사건 높이"); axs[0].set_title("단발인가: 오른쪽 아래 = 큰 사건 하나")
axs[1].set_xlabel("가장 큰 0.1초 창 RMS / 창 중앙값"); axs[1].set_ylabel("중앙값 2배 넘는 창 수 (20개 중)"); axs[1].set_title("2초 중 얼마나 자주 들뜨나")
axs[2].set_xlabel("사건 수 log(1+n)"); axs[2].set_ylabel("큰 사건 잔향 비 log10"); axs[2].set_title("사건 수와 울림")
axs[0].legend(fontsize=8)
for a in axs: a.grid(alpha=.3)
fig.suptitle("충격·시간 구조 (200 Hz 고역통과, 사건 = |h| > 8σ). 라벨 = 경계 파일", fontsize=11)
fig.tight_layout(); save(fig, "4_impact_time")

order = BN + BF + G["고장 본진"][:6]
fig, ax = plt.subplots(figsize=(12, 0.32 * len(order) + 2))
im = ax.imshow(np.log2(WRn[[names.index(n) for n in order]]), cmap="magma", vmin=0, vmax=4, aspect="auto")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7.5)
ax.set_xticks(range(0, 20, 2)); ax.set_xticklabels([f"{t/10:.1f}" for t in range(0, 20, 2)]); ax.set_xlabel("초 (0.1초 창)")
ax.axhline(len(BN) - .5, color="w", lw=1); ax.axhline(len(BN) + len(BF) - .5, color="w", lw=1)
ax.set_title("0.1초 창 RMS / 파일 중앙값 (log2). 위: 정상 경계 · 가운데: 고장 경계 · 아래: 고장 본진 6개", fontsize=10)
fig.colorbar(im, ax=ax, shrink=.6, label="log2 배"); fig.tight_layout(); save(fig, "5_window_rms")
log("\n[③④] 충격·시간 구조 중앙값 (그룹): 큰 사건 높이 log · 두 번째/첫 번째 사건 · 사건 수 log · 잔향 log · 최대 창 배수 · 2배 창 수 · 2위/1위 창")
for k in C:
    s = grp == k
    log(f"  {k:6s}: {IMP['E4_big_height_log'][s].median():.2f} · {IMP['E2_top2_ratio'][s].median():.2f} · "
        f"{IMP['E1_n_impact_log'][s].median():.2f} · {IMP['E3_big_ringing_log'][s].median():.2f} · "
        f"{np.median(TOPW[s, 0]):.1f} · {np.median(ACT[s]):.0f} · {np.median(SECOND[s]):.2f}")
for n in rows:
    i = names.index(n)
    log(f"  {n}: 높이 {IMP['E4_big_height_log'][i]:.2f} · 2위/1위 사건 {IMP['E2_top2_ratio'][i]:.2f} · 사건 {np.expm1(IMP['E1_n_impact_log'][i]):.0f}개 · "
        f"잔향 {IMP['E3_big_ringing_log'][i]:.2f} · 최대 창 ×{TOPW[i, 0]:.1f} · 2배 창 {ACT[i]} · 2위/1위 창 {SECOND[i]:.2f}")


# %% ⑤ 002·029 vs 고장 본진: 무엇을 공유하고 무엇이 빠졌나 (피처 계열별, 기준 정상 중 %)
fams = {}
for j, n in enumerate(FN):
    key = n.rsplit("_", 1)[0] if n[:3] in ("P02", "E01", "S01", "B04") else n[:1] + "계열(" + {"T": "시간", "W": "창", "F": "주파수", "B": "대역"}.get(n[0], n[0]) + ")"
    fams.setdefault(key, []).append(j)
PF = np.column_stack([pct(E[:, j], E[nor_ref, j]) for j in range(E.shape[1])])
famkeys = list(fams)
FM = {k: np.median(PF[:, fams[k]], 1) for k in famkeys}           # 파일별 계열 중앙 %
show = ["abn_002", "abn_029"] + [n for n in BF if n not in ("abn_002", "abn_029")]
fig, ax = plt.subplots(figsize=(14, 5.5))
xs = np.arange(len(famkeys)); w = .8 / (len(show) + 1)
ax.bar(xs - .4 + w / 2, [np.median(FM[k][core]) for k in famkeys], w, color=C["고장 본진"], label="고장 본진 중앙")
for t, n in enumerate(show):
    ax.bar(xs - .4 + w * (t + 1.5), [FM[k][names.index(n)] for k in famkeys], w, label=n,
           color=["#7a3fbf", "#3fa34d", "#f2a37d", "#c9a227", "#888"][t % 5])
ax.axhline(50, color="k", lw=.6); ax.axhline(90, color="k", lw=.6, ls=":")
ax.set_xticks(xs); ax.set_xticklabels(famkeys, rotation=35, ha="right", fontsize=8); ax.set_ylabel("계열 피처들의 기준 정상 중 % (중앙)")
ax.set_title("피처 계열별 위치: 고장 본진은 어디서 높고, 경계 고장은 그중 무엇이 빠졌나", fontsize=10)
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3, axis="y"); fig.tight_layout(); save(fig, "6_family_profile")

# 대역별 프로파일: P02 첨도·피크, B04 비율 (1.25 kHz 대역 10개 / 625 Hz 20개)
fig, axs = plt.subplots(1, 3, figsize=(19, 5))
for a, pre in zip(axs, ["P02_kurtosis", "P02_peak", "B04_ratio"]):
    js = fams[pre]
    a.plot(range(len(js)), np.median(PF[core][:, js], 0), color=C["고장 본진"], lw=2.5, label="고장 본진 중앙")
    a.fill_between(range(len(js)), np.percentile(PF[core][:, js], 25, 0), np.percentile(PF[core][:, js], 75, 0), color=C["고장 본진"], alpha=.15)
    for t, n in enumerate(show):
        a.plot(range(len(js)), PF[names.index(n), js], lw=1.2, marker="o", ms=3, label=n, color=["#7a3fbf", "#3fa34d", "#f2a37d", "#c9a227", "#888"][t % 5])
    for n in ["nor_059", "nor_062"]:
        a.plot(range(len(js)), PF[names.index(n), js], lw=.8, ls="--", label=n, color=C["정상 경계"])
    a.set_xticks(range(len(js))); a.set_xticklabels([FN[j].rsplit("_", 1)[1] for j in js], rotation=90, fontsize=7)
    a.axhline(90, color="k", lw=.5, ls=":"); a.set_title(pre + " (대역, 단위 100 Hz)", fontsize=10); a.set_ylabel("기준 정상 중 %"); a.grid(alpha=.3)
axs[0].legend(fontsize=7)
fig.tight_layout(); save(fig, "7_band_profile")

# PSD: 002·029 vs 고장 본진 중앙 vs 정상 띠, 059·062 대비
f, L = viz.log_psd(X)
fig, axs = plt.subplots(2, 1, figsize=(14, 9))
for a, (lo, hi) in zip(axs, ((0, 12500), (7000, 10000))):
    m = (f >= lo) & (f <= hi)
    a.fill_between(f[m], np.percentile(L[nor_ref], 5, 0)[m], np.percentile(L[nor_ref], 95, 0)[m], color=C["정상 아래"], alpha=.4, label="기준 정상 5–95%")
    a.plot(f[m], np.median(L[core], 0)[m], color=C["고장 본진"], lw=2, label="고장 본진 중앙")
    for n, c, ls in (("abn_002", "#7a3fbf", "-"), ("abn_029", "#3fa34d", "-"), ("nor_059", C["정상 경계"], "--"), ("nor_062", "#5fa8e8", "--")):
        a.plot(f[m], L[names.index(n)][m], color=c, lw=.9, ls=ls, label=n)
    a.set_xlim(lo, hi); a.grid(alpha=.3); a.set_ylabel("log10 PSD")
axs[0].legend(fontsize=8, ncol=3); axs[1].set_xlabel("Hz")
axs[0].set_title("002·029의 PSD: 고장 본진처럼 넓은 대역이 오르나, 한 곳(8.25 kHz)만 오르나. 아래 = 7–10 kHz 확대", fontsize=10)
fig.tight_layout(); save(fig, "8_psd_002_029")

log("\n[⑤] 피처 계열별 기준 정상 중 % (중앙): 고장 본진 / " + " / ".join(show) + " / 059 / 062")
for k in famkeys:
    log(f"  {k:22s} {np.median(FM[k][core]):5.0f} / " + " / ".join(f"{FM[k][names.index(n)]:5.0f}" for n in show)
        + f" / {FM[k][names.index('nor_059')]:5.0f} / {FM[k][names.index('nor_062')]:5.0f}")

open(os.path.join(FIG, "numbers.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
