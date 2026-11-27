# %% 발표용 새 그림 4개 (발표_목차.md 1 · 2 · 3 · 4장)
# 1 kpi       : 결론 KPI 카드 (EXP-050 result.json)
# 2 rms_chart : RMS 관리도 (과제 PDF p13과 같은 정의: 원신호 RMS, 정상 평균 + 3σ)
# 3 structure : J-축소 구조도 (파이프라인_설명 §2)
# 4 ladder    : 부품을 하나씩 뺀 F2 사다리 (파이프라인_설명 §4, 실험로그)
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from src.exp import load_train, find_exp
import src.viz  # noqa: F401  (맑은 고딕 · 마이너스 기호 설정)

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

NAVY, INK, MUTED, GRID = "#0B2A4F", "#1F2937", "#6B7280", "#E5E7EB"
NOR, ABN = "#2a78d6", "#eb6834"      # 실험 그림과 같은 색 (정상 파랑 · 고장 주황)
LIGHT = "#EEF2F7"
plt.rcParams.update({"font.size": 14, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})

R = json.load(open(os.path.join(find_exp("EXP-050"), "result.json"), encoding="utf-8"))
C = R["configs"]["J-축소"]
log = []


def save(fig, name):
    p = os.path.join(FIG, f"{name}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.append(f"저장: figs/{name}.png")


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=INK, fs=14, bold=False, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(ax, p, q, color=MUTED, lw=1.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0))


# %% 1. KPI 카드
import pandas as pd
val_pos = int((pd.read_csv(os.path.join(ROOT, "3_단계", "7_최종·제출", "아카이브_J축소제출", "출력", "submission_val_9조.csv"))["label"] == 1).sum())   # 제출 파일의 고장 판정 수 (11-23 K83 규칙 후 9)
cards = [
    ("중첩 CV F2", f"{C['f2']:.3f}", f"± {C['f2_sd']:.3f} (5회 반복)"),
    ("재현율", f"{C['recall']:.3f}", "고장 30개 모두 검출"),
    ("오경보율", f"{C['fpr']:.3f}", "정상 120개 중 약 3.6개"),
    ("val 고장 판정", f"{val_pos} / 40", "1차 제출 (11/26)"),
]
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.set_xlim(0, 4); ax.set_ylim(0, 1); ax.axis("off")
for i, (t, v, s) in enumerate(cards):
    x = i + 0.05
    ax.add_patch(FancyBboxPatch((x, 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=LIGHT, ec="none"))
    ax.add_patch(plt.Rectangle((x, 0.05), 0.025, 0.9, fc=NAVY if i else ABN, ec="none"))
    ax.text(x + 0.08, 0.78, t, fontsize=16, color=MUTED, va="center")
    ax.text(x + 0.08, 0.47, v, fontsize=36, color=NAVY, fontweight="bold", va="center")
    ax.text(x + 0.08, 0.18, s, fontsize=13, color=MUTED, va="center")
save(fig, "1_kpi")

# %% 2. RMS 관리도
X, y, names = load_train()
rms = np.sqrt((X ** 2).mean(axis=1))          # 원신호 RMS (평균 제거 없음, 과제 PDF와 같음)
rn = rms[y == 0]
cl, ucl = rn.mean(), rn.mean() + 3 * rn.std(ddof=1)
fp, tp = int((rn > ucl).sum()), int((rms[y == 1] > ucl).sum())
fn = int((y == 1).sum()) - tp
f2 = 5 * tp / (5 * tp + 4 * fn + fp)
log.append(f"RMS 관리도: CL {cl:.3f} UCL {ucl:.3f} · 정상 초과 {fp} · 고장 초과 {tp} · F2 {f2:.3f}")

fig, ax = plt.subplots(figsize=(13, 5))
idx = np.arange(len(y))
ax.scatter(idx[y == 0], rms[y == 0], s=22, color=NOR, label=f"정상 120개 (초과 {fp}개 = 오경보)", zorder=3)
ax.scatter(idx[y == 1], rms[y == 1], s=30, color=ABN, label=f"고장 30개 (초과 {tp}개 = 검출)", zorder=3)
ax.axhline(ucl, color=ABN, ls="--", lw=1.6)
ax.axhline(cl, color=INK, lw=1.2)
ax.text(len(y) + 1, ucl, f"UCL = 평균 + 3σ = {ucl:.2f}", va="center", fontsize=13, color=ABN)
ax.text(len(y) + 1, cl, f"CL = {cl:.2f}", va="center", fontsize=13)
ax.axvline((y == 0).sum() - 0.5, color=GRID, lw=1.2)
ax.set_xlim(-2, len(y) + 1); ax.set_xticks([])
ax.set_ylabel("파일 RMS (g)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=13)
ax.set_title(f"고장 30개 중 {fn}개가 정상 범위 안 → F2 {f2:.3f}", loc="left", fontsize=16, color=NAVY)
save(fig, "2_rms_chart")

# %% 3. 구조도
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(-0.4, 6.9); ax.axis("off")
H = 0.95                      # 상자 높이
yc = 4.0 + H / 2              # 결합 · OR · 판정의 가운데 높이 (= 박자 상자 가운데)
box(ax, 0.1, 2.9, 1.6, 1.2, "진동 신호\n2초 · 25 kHz", fc="white", fs=13)
experts = [
    ("ET (충격 · 울림 피처 15열)", "아는 고장과 닮았나"),
    ("박자 z", "하중 주기에 맞춰 덜컥거리나"),
    ("AE z (정상 스펙트럼만 학습)", "정상에서 얼마나 벗어났나"),
]
for (t, q), yy in zip(experts, [5.2, 4.0, 2.8]):
    box(ax, 2.5, yy, 4.0, H, f"{t}\n“{q}”", fs=13)
    arrow(ax, (1.7, 3.5), (2.5, yy + H / 2))
    arrow(ax, (6.5, yy + H / 2), (7.3, yc))
ax.text(4.5, 6.45, "통계 모델 (데이터에서 학습)", ha="center", fontsize=14, color=NAVY, fontweight="bold")
box(ax, 7.3, yc - 0.65, 2.3, 1.3, "로지스틱 결합\n점수 ≥ 문턱", fc=NAVY, ec=NAVY, tc="white", fs=14, bold=True)

rules = [("규칙 E2 · 하중 대비 고주파", "유격: 004 011 017 019"),
         ("규칙 8.3k · 8.3 kHz 모드", "공진: 002 003 006 010 029 030")]
mx, my = 10.7, 1.15           # 규칙 화살표가 모이는 점 → 위로 OR
for (t, s), yy in zip(rules, [1.25, 0.05]):
    box(ax, 2.5, yy, 4.0, H, f"{t}\n{s}", fc="#FDF0EA", ec=ABN, fs=13)
    arrow(ax, (1.7, 3.1), (2.5, yy + H / 2))
    ax.plot([6.5, mx], [yy + H / 2, my], color=ABN, lw=1.8)
arrow(ax, (mx, my), (mx, yc - 0.65), color=ABN)
ax.text(12.5, 1.5, "물리 규칙\n하나라도 켜지면 고장", ha="center", fontsize=14, color=ABN, fontweight="bold")

box(ax, 10.2, yc - 0.65, 1.0, 1.3, "OR", fc="white", ec=INK, fs=18, bold=True)
arrow(ax, (9.6, yc), (10.2, yc))
box(ax, 11.8, yc - 0.65, 2.1, 1.3, "판정\n정상 0 / 고장 1", fc="white", ec=NAVY, fs=14, bold=True)
arrow(ax, (11.2, yc), (11.8, yc))
save(fig, "3_structure")

# %% 4. 사다리 (아래 → 위로 부품을 더함, 발표는 위에서 아래로 하나씩 빼며 읽음)
rows = [  # 이름, F2, sd, 맡은 일, 종류
    ("RMS 관리도 (평균 + 3σ)", f2, 0, "고장 28개 놓침", "base"),
    ("전부 고장으로 판정", 0.556, 0, "재현율 1 · 오경보율 1", "base"),
    ("ET(충격 · 울림) + 박자 z", 0.907, 0.009, "022는 박자로 검출 · 002 · 017 놓침", "model"),
    ("+ 규칙 E2", 0.933, 0.014, "+017 검출", "model"),
    ("+ 규칙 8.3k", 0.959, 0.014, "+002 검출 → 재현율 0.993", "model"),
    ("+ AE z  (J)", 0.977, 0.010, "오경보율 0.047 → 0.030", "model"),
    ("+ 울림 비율 축소  (J-축소, 제출)", C["f2"], C["f2_sd"], "점수 같음 · val 흔들림 7–20 → 7–13", "final"),
]
fig, ax = plt.subplots(figsize=(13, 5.6))
yy = np.arange(len(rows))
col = {"base": "#C7CDD6", "model": NAVY, "final": ABN}
for i, (n, v, s, note, k) in enumerate(rows):
    ax.barh(i, v, height=0.62, color=col[k], zorder=2)
    if s:
        ax.errorbar(v, i, xerr=s, color=INK, capsize=4, lw=1.2, zorder=3)
    ax.text(v + 0.02 + s, i, f"{v:.3f}" + (f" ± {s:.3f}" if s else ""), va="center", fontsize=13,
            fontweight="bold" if k == "final" else "normal")
    ax.text(1.17, i, note, va="center", fontsize=12, color=MUTED)
ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
ax.set_xlim(0, 1.17); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("F2 (중첩 교차검증, 5회 반복 평균 ± sd)")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(1.17, len(rows) - 0.35, "더한 부품이 맡은 일", fontsize=12, color=MUTED, fontweight="bold")
save(fig, "4_ladder")

open(os.path.join(FIG, "_log.txt"), "w", encoding="utf-8").write("\n".join(log) + "\n")
