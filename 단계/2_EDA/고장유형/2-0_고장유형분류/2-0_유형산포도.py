# %% [markdown]
# 2-0 고장 도메인 유형 분류 — EDA 뷰어 지표로 고장·정상을 1-3의 4유형(L1–L4)에 나눈다
# - 지표 값: 2-4 EDA 뷰어(build_viewer.py)가 남긴 figs/features_raw.csv(평균 제거)·features_hp.csv(200 Hz 고역통과)
#   → 뷰어 "산점도" 화면과 같은 값
# - 유형 규칙: 1-3 고장유형.md의 "신호 징후"를 지표로 옮긴 것. 문턱은 정상 분포 분위(정상 95%·중앙)와 뷰어 태그(0–20 Hz ≥ 60%)만
#   사용 → 라벨은 색·집계에만 쓴다. 단, 규칙을 쓰기 전에 고장 30개의 값을 본 뒤라 완전한 사전 고정은 아니다 (md 주의 참고)
# - 같은 규칙을 정상 120개에도 적용 → "고장과 비슷한 정상" 무리
# 실행: conda run -n qe-phm python "단계/2_EDA/고장유형/2-0_고장유형분류/2-0_유형산포도.py"

# %% 설정
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False; plt.rcParams["mathtext.fontset"] = "dejavusans"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
VIEW = os.path.join(ROOT, "단계", "2_EDA", "2-4_EDA뷰어", "figs")
R = pd.read_csv(os.path.join(VIEW, "features_raw.csv"), index_col=0, encoding="utf-8-sig")
H = pd.read_csv(os.path.join(VIEW, "features_hp.csv"), index_col=0, encoding="utf-8-sig")
y = R["y"].values; nor = y == 0; names = list(R.index)

# 도메인 축 (뷰어 지표 이름 그대로)
F = pd.DataFrame({
    "e20_200": R["e20_200"],      # 20–200 Hz 에너지 (log g²) — 저주파 대진폭 진동 (L4)
    "r20_200": R["s200"] - R["s20"],  # 20–200 Hz 몫 (전체 분산 중) — 에너지가 저주파에 몰렸나 (1-3 L4 "99% 집중")
    "e200": R["e200"],            # 200 Hz 이상 에너지 (log g²) — 광대역 타격 에너지 (L3)
    "max_hp": H["maxabs"],        # 고역통과 Peak (g) — 강한 금속 타격 (L3)
    "n_imp_hp": H["n_imp"],       # 고역통과 충격 수 — 연발 타격 (L3 연발)
    "s20": R["s20"],              # 0–20 Hz 분산 몫 — 느린 출렁임 (L2)
    "kurt_hp": H["kurt"],         # 고역통과 첨도 — 출렁임에 가려진 충격 (L2)
    "hi_ratio": H["hi_ratio"],    # 8 kHz 이상 / 전체 (log) — 고주파 링잉 (L1)
    "wkm_hp": H["win_kurt_max"],  # 50 ms 창 첨도 최대 — 짧은 국소 충격 (L1)
}, index=names)
Q = lambda k, q: float(np.percentile(F.loc[nor, k], q))
TH = dict(e20_200=Q("e20_200", 95), r20_200=Q("r20_200", 95),e200=Q("e200", 95), max_hp=Q("max_hp", 95), n_imp_hp=Q("n_imp_hp", 95),
          s20=0.6, kurt_hp=Q("kurt_hp", 50), hi_ratio=Q("hi_ratio", 95), wkm_hp=Q("wkm_hp", 95))

# %% 유형 규칙 (위에서부터 처음 맞는 것)
TYPES = [  # key, 이름, 색(dataviz 기본 범주 순서), 마커
    ("L4", "L4 저주파 대진폭 공진", "#eda100", "D"),
    ("L3", "L3 금속 타격·광대역", "#1baf7a", "s"),
    ("L2", "L2 출렁임 + 가려진 충격", "#eb6834", "^"),
    ("L1", "L1 국소·고주파 (에너지 정상 수준)", "#2a78d6", "o"),
    ("U", "미분류 (모든 축에서 정상 안)", "#8a8a85", "x"),
]
TNAME = {k: n for k, n, *_ in TYPES}


def classify(r):
    # 11-16: 몫 조건 추가 — 크기만 보면 에너지가 12 kHz에 몰린 평균 −1 g 파일(097·101·120)이 걸림
    if r.e20_200 > TH["e20_200"] and r.r20_200 > TH["r20_200"] and r.n_imp_hp <= 2:
        return "L4", "충격 없이 20–200 Hz 에너지·몫 > 정상 95%"
    if r.e200 > TH["e200"] or r.max_hp > TH["max_hp"]:
        sub = "연발" if r.n_imp_hp > TH["n_imp_hp"] else ("강타" if r.max_hp > TH["max_hp"] else "광대역")
        return "L3", sub
    if r.s20 >= TH["s20"] and r.kurt_hp > TH["kurt_hp"]:
        return "L2", "0–20 Hz ≥ 60% & 고역통과 첨도 > 정상 중앙"
    if r.hi_ratio > TH["hi_ratio"] or r.wkm_hp > TH["wkm_hp"]:
        return "L1", "고역 비 > 정상 95%" if r.hi_ratio > TH["hi_ratio"] else "창 첨도 최대 > 정상 95%"
    return "U", ""


C = F.apply(lambda r: pd.Series(classify(r), index=["type", "why"]), axis=1)
C.insert(0, "y", y)

# 최종 모델 EXP-025 결과 (seed 5번 중 놓침·오경보 횟수) — 유형별로 모델이 어디서 틀리는지
_r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", "base", "EXP-025_1114_유격반복규칙", "result.json"), encoding="utf-8"))
_c = _r["configs"][_r.get("main") or next(iter(_r["configs"]))]
C["miss"] = [int(_c["miss_counts"].get(n, 0)) for n in names]
C["fp"] = [int(_c["fp_counts"].get(n, 0)) for n in names]
pd.concat([C, F.round(4)], axis=1).to_csv(os.path.join(FIG, "types.csv"), encoding="utf-8-sig")

# %% 그림: 도메인 축 4장 × (고장 | 정상)
PLOTS = [  # 파일명, 제목, x, y, x로그, y로그, (x선, y선), 어떤 유형을 보는 축인가
    ("1_L4_low", "저주파 크기 vs 저주파 몫 — L4는 오른쪽 위 (크고 몰려 있음)", "e20_200", "r20_200", False, False, ("e20_200", "r20_200"), "20–200 Hz 에너지 (log g²)", "20–200 Hz 몫"),
    ("2_L3_energy_peak", "고역 에너지 vs 고역 Peak — L3는 오른쪽 또는 위", "e200", "max_hp", False, True, ("e200", "max_hp"), "200 Hz 이상 에너지 (log g²)", "고역통과 Peak (g)"),
    ("3_L3_hits", "타격 횟수 vs 세기 — L3 연발은 오른쪽, 강타는 위", "n_imp_hp", "max_hp", "symlog", True, ("n_imp_hp", "max_hp"), "고역통과 충격 수", "고역통과 Peak (g)"),
    ("4_L2_sway", "출렁임 몫 vs 가려진 충격 — L2는 오른쪽 위", "s20", "kurt_hp", False, "symlog", ("s20", "kurt_hp"), "0–20 Hz 분산 몫", "고역통과 첨도"),
    ("5_L1_local", "고주파 몫 vs 국소 충격 — L1은 오른쪽 또는 위", "hi_ratio", "wkm_hp", False, "symlog", ("hi_ratio", "wkm_hp"), "고역 비 8 kHz+ (log)", "50 ms 창 첨도 최대"),
]
TH_LABEL = dict(e20_200="정상 95%", r20_200="정상 95%",e200="정상 95%", max_hp="정상 95%", n_imp_hp="정상 95%", s20="60%", kurt_hp="정상 중앙", hi_ratio="정상 95%", wkm_hp="정상 95%")


def scale(ax, which, mode):
    if mode == "symlog":
        (ax.set_xscale if which == "x" else ax.set_yscale)("symlog", linthresh=1)
    elif mode:
        (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    if mode:                                                   # 눈금은 평범한 숫자로 (10^n 표기 대신)
        (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


for fn, title, kx, ky, lx, ly, (tx, ty), xl, yl in PLOTS:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for ax, grp, lab in ((axs[0], ~nor, "고장 30개 — 유형별 색"), (axs[1], nor, "정상 120개 — 같은 규칙으로 고장과 비슷한 정상")):
        ax.scatter(F.loc[~grp, kx], F.loc[~grp, ky], s=14, c="#c9c8c2", lw=0, zorder=1,
                   label="고장 (회색)" if grp is nor else "정상 (회색)")
        for k, n, col, mk in TYPES:
            m = grp & (C["type"] == k).values
            if not m.any():
                continue
            ax.scatter(F.loc[m, kx], F.loc[m, ky], s=46, c=col, marker=mk, lw=(1.6 if mk == "x" else .8),
                       **({} if mk == "x" else dict(edgecolors="white")), zorder=3, label=f"{n} ({m.sum()})")
            for i in np.flatnonzero(m):
                if grp is nor and k == "U":
                    continue                                    # 정상 미분류는 이름 생략 (너무 많음)
                ax.annotate(names[i].split("_")[1], (F[kx].iloc[i], F[ky].iloc[i]), fontsize=7, color="#3d3d3a",
                            xytext=(3, 3), textcoords="offset points", zorder=4)
        ax.axvline(TH[tx], color="#6b6a64", lw=1, ls="--", zorder=2); ax.axhline(TH[ty], color="#6b6a64", lw=1, ls="--", zorder=2)
        ax.text(TH[tx], 1.0, f" {TH_LABEL[tx]}", transform=ax.get_xaxis_transform(), fontsize=7, color="#6b6a64", va="top")
        ax.text(1.0, TH[ty], f"{TH_LABEL[ty]} ", transform=ax.get_yaxis_transform(), fontsize=7, color="#6b6a64", ha="right", va="bottom")
        scale(ax, "x", lx); scale(ax, "y", ly)
        ax.set_title(lab, fontsize=10, loc="left"); ax.set_xlabel(xl); ax.grid(alpha=.25, lw=.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=7.5, loc="best", frameon=False)
    axs[0].set_ylabel(yl)
    fig.suptitle(title + "   (점선 = 정상 분포 기준 문턱, 색 = 규칙으로 정한 유형)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fn + ".png"), dpi=130); plt.close(fig)

# %% 요약 (UTF-8)
L = ["문턱 (정상 분포 기준): " + ", ".join(f"{k} {v:.3g}" for k, v in TH.items()), ""]
for k, n, *_ in TYPES:
    for g, gl in ((~nor, "고장"), (nor, "정상")):
        m = g & (C["type"] == k).values
        sub = C[m]
        extra = ""
        if gl == "고장":
            extra = f" | EXP-025 놓침(5번 중) " + ", ".join(f"{i.split('_')[1]}:{v}" for i, v in sub["miss"].items() if v)
        else:
            extra = f" | EXP-025 오경보(5번 중) " + ", ".join(f"{i.split('_')[1]}:{v}" for i, v in sub["fp"].items() if v)
        lst = ", ".join(f"{i.split('_')[1]}" + (f"({w})" if k == "L3" else "") for i, w in sub["why"].items()) if (gl == "고장" or k != "U") else f"{m.sum()}개"
        L.append(f"[{n}] {gl} {m.sum()}: {lst}{extra}")
    L.append("")
open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8").write("\n".join(L))
print("done")
