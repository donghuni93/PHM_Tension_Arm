# %% [markdown]
# 2-0-2 EXP-027 가족 단위 검정 통과 피처 5개 — 고장·정상 산포도 (고장 색 = 2-0 도메인 유형)
# - 피처: K1_second_snr · P02_peak_025 · P02_peak_075 · P02_kurtosis_012 · S01_cv_012 (EXP-027 README)
# - 값: outputs/cache/feat_ext.csv(P02·S01) · feat_knock.csv(K1) — 실험들이 쓴 것과 같은 캐시. 행 순서 = load_train()
# - 정상 분위(5·95%)는 정상 120개 기준. 라벨은 색·집계에만
# - 그림은 한 장에 하나씩 (11-16 사용자 요청: 묶지 말 것) → figs/2-0-2/
# 실행: conda run -n qe-phm python "단계/2_EDA/유의미/2-0_고장유형분류/2-0_2_가족단위피처산포도.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.exp import load_train, cached_table
from src.features import feature_table_ext, feature_table_knock, feature_table_phys
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(FIG, "2-0-2"); os.makedirs(OUT, exist_ok=True)
for old in ("6_fam5_strips.png", "7_fam5_matrix.png", "8_spec_modes.png"):    # 예전 묶음 그림 정리
    p = os.path.join(FIG, old)
    if os.path.exists(p):
        os.remove(p)

X, y, names = load_train(); y = np.asarray(y); nor = y == 0
E = cached_table("ext", feature_table_ext, X); K = cached_table("knock", feature_table_knock, X); PH = cached_table("phys", feature_table_phys, X)
T = pd.read_csv(os.path.join(FIG, "types.csv"), index_col=0, encoding="utf-8-sig").loc[names]
assert (T.y.values == y).all()
from src.experts import peak83_table, Rule83
_P83, _W83, _P86 = peak83_table()                                    # EXP-021 규칙과 같은 정의 (대역 경계는 사후, 결정 '대역 예외')
RULE_ON = Rule83(0.75, 150)(np.arange(len(y))).astype(bool)          # 8.3k 규칙: 돌출 ≥ 0.75 dec & 폭 ≥ 150 Hz

# 표시용 값: 크기·첨도·변동계수는 log10 (K1은 이미 log10)
FE = [  # 키, 표시 이름, 값
    ("K1", "K1 두 번째 사건 높이 (log10 σ배)", K["K1_second_snr"].values),
    ("pk025", "P02 Peak 2.5–3.75 kHz (log10 g)", np.log10(E["P02_peak_025"].values)),
    ("pk075", "P02 Peak 7.5–8.75 kHz (log10 g)", np.log10(E["P02_peak_075"].values)),
    ("ku012", "P02 첨도 1.25–2.5 kHz (log10, Pearson)", np.log10(E["P02_kurtosis_012"].values)),
    ("cv012", "S01 시간 변동 1.25–2.5 kHz (log10 CV)", np.log10(E["S01_cv_012"].values)),
    # H3 스펙트럼 모양 (11-16 추가): 250 Hz 칸 평균 로그 PSD − 파일 전체 평균 (게인 무관). EXP-027 군집 43 = 8.25–9.75 kHz 6칸,
    # 대표 = 9.5–9.75 kHz 칸. 8–9 kHz는 칸 평균을 쓰지 않는다: 같은 대역 안에 고장 모드(8.20–8.45)와 정상 모드(8.45–8.75)가
    # 섞여 있어서(2-11 README2, EXP-020·021) → 규칙과 같은 정의의 두 모드 축(src.experts.peak83_table)을 따로 쓴다
    ("h3_95", "H3 모양 9.5–9.75 kHz (log10, 파일 평균 대비)", PH["H3_shape_09500"].values),
    ("p83", "8.20–8.45 kHz 최대 돌출 (dec) = 고장 모드", _P83),
    ("p86", "8.45–8.75 kHz 최대 돌출 (dec) = 정상 모드", _P86),
    ("dmode", "두 모드 차 (8.3 - 8.6 돌출, dec)", _P83 - _P86),
    ("w83", "8.3 kHz 봉우리 폭 (Hz)", _W83),
]
FAM = ["K1", "pk025", "pk075", "ku012", "cv012"]                    # EXP-027 가족 단위 통과 5개
V = pd.DataFrame({k: v for k, _, v in FE}, index=names)
LAB = {k: l for k, l, _ in FE}
TYPES = [("L4", "L4 저주파 공진", "#eda100", "D"), ("L3", "L3 금속 타격", "#1baf7a", "s"),
         ("L2", "L2 출렁임+충격", "#eb6834", "^"), ("L1", "L1 국소·고주파", "#2a78d6", "o"), ("U", "미분류", "#6b6a64", "X")]
NOR_C = "#c9c8c2"
P5 = {k: np.percentile(V.loc[nor, k], 5) for k in V}; P95 = {k: np.percentile(V.loc[nor, k], 95) for k in V}
short = lambda n: n.split("_")[1]
fpn = T.index[(T.y == 0) & (T.fp >= 3)]                              # EXP-025 오경보 3/5+ 정상 (이름 표시)
FP_H = plt.Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec="#2a78d6", mew=1.3)
RULE_H = plt.Line2D([], [], marker="D", ls="", ms=10, mfc="none", mec="#3d3d3a")


def style(ax):
    ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=130); plt.close(fig)


# %% 그림 1: 피처별 띠 그림 (피처마다 한 장) — 정상 5–95% 띠, 고장은 유형 색 + 이름
jit = np.random.default_rng(0).uniform(-.22, .22, len(y))
for n_, k in enumerate(V, 1):
    v = V[k].values
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.axhspan(P5[k], P95[k], color="#2a78d6", alpha=.08, lw=0, label="정상 5–95%")
    ax.axhline(P95[k], color="#6b6a64", ls="--", lw=1, label="정상 95%")
    ax.scatter(0 + jit[nor], v[nor], s=22, c=NOR_C, lw=0, label="정상")
    for i in np.flatnonzero(nor):
        if names[i] in fpn or v[i] > np.sort(v[nor])[-4]:            # 오경보 정상 + 정상 상위 3개 이름
            ax.annotate(short(names[i]), (jit[i], v[i]), fontsize=9, color="#2a78d6", xytext=(5, 0), textcoords="offset points", va="center")
    for t, tl, col, mk in TYPES:
        m = (~nor) & (T.type.values == t)
        ax.scatter(1 + jit[m], v[m], s=60, c=col, marker=mk, lw=.7, edgecolors="white", label=tl, zorder=3)
        for i in np.flatnonzero(m):
            ax.annotate(short(names[i]), (1 + jit[i], v[i]), fontsize=9, color="#3d3d3a", xytext=(5, 0), textcoords="offset points", va="center")
    n_hi = int(((~nor) & (v > P95[k])).sum())
    ax.set_title(f"{LAB[k]}\n정상 95% 위: 고장 {n_hi}/30 · 정상 {int((nor & (v > P95[k])).sum())}/120", fontsize=12, loc="left")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"], fontsize=11); ax.set_xlim(-.6, 1.75); style(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, f"1-{n_:02d}_strip_{k}.png")

# %% 그림 2: 가족 5개 두 피처 평면 (쌍마다 한 장)
PAIRS = [(a, b) for i, a in enumerate(FAM) for b in FAM[i + 1:]]     # 10쌍, x = 앞 피처
for n_, (kx, ky) in enumerate(PAIRS, 1):
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.scatter(V.loc[nor, kx], V.loc[nor, ky], s=22, c=NOR_C, lw=0, label="정상")
    for t, tl, col, mk in TYPES:
        m = (~nor) & (T.type.values == t)
        ax.scatter(V.loc[m, kx], V.loc[m, ky], s=60, c=col, marker=mk, lw=.7, edgecolors="white", label=tl, zorder=3)
    for n in fpn:                                                   # 오경보 정상은 파란 테두리 + 이름
        ax.scatter(V.loc[n, kx], V.loc[n, ky], s=90, facecolor="none", edgecolor="#2a78d6", lw=1.3, zorder=4)
        ax.annotate(short(n), (V.loc[n, kx], V.loc[n, ky]), fontsize=9, color="#2a78d6", xytext=(5, 3), textcoords="offset points")
    for n in ("abn_002", "abn_022", "abn_029"):                    # 늘 어려운 고장 이름
        ax.annotate(short(n), (V.loc[n, kx], V.loc[n, ky]), fontsize=9, xytext=(5, 3), textcoords="offset points")
    ax.axvline(P95[kx], color="#6b6a64", ls="--", lw=.9); ax.axhline(P95[ky], color="#6b6a64", ls="--", lw=.9)
    ax.set_xlabel(LAB[kx], fontsize=11); ax.set_ylabel(LAB[ky], fontsize=11)
    ax.set_title(f"{LAB[kx].split(' (')[0]}  ×  {LAB[ky].split(' (')[0]}\n점선 = 정상 95%, 이름 = 002·022·029 + EXP-025 오경보 정상(파랑)", fontsize=11, loc="left")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [FP_H], l + ["EXP-025 오경보 정상"], fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style(ax); save(fig, f"2-{n_:02d}_pair_{kx}_{ky}.png")

# %% 그림 3: 스펙트럼 모양 (평면마다 한 장) — 8.3 kHz(고장 모드)와 8.6 kHz(정상 모드)를 따로 (EXP-021 규칙과 같은 정의)
PANELS = [("p86", "p83", "① 두 모드 비교 — 대각선 위 = 8.3 kHz 봉우리가 더 큼"),
          ("w83", "p83", "② 8.3 kHz 규칙 — 돌출 ≥ 0.75 dec & 폭 ≥ 150 Hz (오른쪽 위 상자)"),
          ("dmode", "h3_95", "③ 두 모드 차 vs H3 9.5–9.75 kHz 모양 (EXP-027 대표 칸)")]
for n_, (kx, ky, ttl) in enumerate(PANELS, 1):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(V.loc[nor, kx], V.loc[nor, ky], s=22, c=NOR_C, lw=0, label="정상")
    for t, tl, col, mk in TYPES:
        m = (~nor) & (T.type.values == t)
        ax.scatter(V.loc[m, kx], V.loc[m, ky], s=60, c=col, marker=mk, lw=.7, edgecolors="white", label=tl, zorder=3)
    for n in fpn:
        ax.scatter(V.loc[n, kx], V.loc[n, ky], s=90, facecolor="none", edgecolor="#2a78d6", lw=1.3, zorder=4)
    on = np.flatnonzero(RULE_ON)
    ax.scatter(V[kx].values[on], V[ky].values[on], s=190, marker="D", facecolor="none", edgecolor="#3d3d3a", lw=1, zorder=2)
    show = set(on) | {names.index(n) for n in ("abn_002", "abn_029", "abn_022", "nor_019", "nor_004")} | {names.index(n) for n in fpn}
    show |= set(np.argsort(-V[ky].values)[:6])
    for i in show:
        ax.annotate(short(names[i]), (V[kx].iloc[i], V[ky].iloc[i]), fontsize=9, xytext=(5, 3), textcoords="offset points",
                    color="#2a78d6" if nor[i] else "#3d3d3a", zorder=5)
    if kx == "p86":
        lim = [min(V.p86.min(), V.p83.min()), max(V.p86.max(), V.p83.max())]
        ax.plot(lim, lim, color="#6b6a64", lw=1, ls="--")
    if kx == "w83":
        ax.axhline(0.75, color="#6b6a64", ls="--", lw=1); ax.axvline(150, color="#6b6a64", ls="--", lw=1)
    if kx == "dmode":
        ax.axvline(0, color="#6b6a64", ls="--", lw=1); ax.axhline(P95["h3_95"], color="#6b6a64", ls=":", lw=1)
    ax.set_xlabel(LAB[kx], fontsize=11); ax.set_ylabel(LAB[ky], fontsize=11); ax.set_title(ttl, fontsize=12, loc="left")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [FP_H, RULE_H], l + ["EXP-025 오경보 정상", "8.3k 규칙 켜짐"], fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    style(ax); save(fig, f"3-{n_}_spec_{kx}_{ky}.png")

# %% 요약 (UTF-8)
L = []
for k in V:
    v = V[k]
    hiF = [short(n) for n in V.index[(~nor) & (v > P95[k])]]
    hiN = [short(n) for n in V.index[nor & (v > P95[k])]]
    loF = [short(n) for n in V.index[(~nor) & (v <= np.percentile(V.loc[nor, k], 50))]]
    L.append(f"[{LAB[k]}] 정상 5/50/95% {P5[k]:.3f}/{np.percentile(V.loc[nor, k], 50):.3f}/{P95[k]:.3f}")
    L.append(f"  정상 95% 위 고장 {len(hiF)}: {' '.join(hiF)}")
    L.append(f"  정상 95% 위 정상 {len(hiN)}: {' '.join(hiN)}")
    L.append(f"  정상 중앙 이하 고장 {len(loF)}: {' '.join(loF)}")
    for t, tl, *_ in TYPES:
        m = (~nor) & (T.type.values == t)
        L.append(f"    {tl}: 95% 위 {int((m & (v > P95[k]).values).sum())}/{m.sum()}")
anyhi = (V[FAM] > pd.Series(P95)[FAM]).any(axis=1)
L += ["", f"가족 5개 중 하나라도 정상 95% 위: 고장 {int((anyhi & ~nor).sum())}/30 · 정상 {int((anyhi & nor).sum())}/120",
      "  고장 중 하나도 안 넘는 것: " + " ".join(short(n) for n in V.index[(~anyhi) & (~nor)]),
      "", "오경보 정상 값:", V.loc[fpn].round(3).to_string(),
      "", "어려운 고장 값 (정상 백분위):",
      V.loc[["abn_002", "abn_022", "abn_029", "abn_015", "abn_017"]].apply(lambda r: pd.Series({k: 100 * (V.loc[nor, k] < r[k]).mean() for k in V}), axis=1).round(0).to_string()]
V.round(4).assign(y=y, type=T.type.values).to_csv(os.path.join(FIG, "fam5_values.csv"), encoding="utf-8-sig")
open(os.path.join(FIG, "fam5_summary.txt"), "w", encoding="utf-8").write("\n".join(L))
print("done")
