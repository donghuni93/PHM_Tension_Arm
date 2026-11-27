# %% [markdown]
# 6-12 (11-23) V025는 고장인가 정상인가 — 학습 150개에서 '닮은 파일'의 라벨로 본다 (모델 없음)
# V025의 물리 특징 하나씩: ① 울림이 1.6–2.2k(1.9k 공유 모드)에 몰림 ② 쌍 충격(12 ms 안 두 번) ③ 사건 간격이 규칙적 ④ 충격이 작음 ⑤ 충격 수가 적음
# 각 특징을 가진 train 파일의 고장 비율(기저 고장 비율 30/150 = 20 %) + 특징 5개를 합친 거리로 가장 가까운 train 파일 10개
# 충격 · 울림 주파수는 2-14 캐시(impacts.csv: 1 kHz 고역 포락 충격, 감쇠 사인파 50 Hz 사전). 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp import OUTLIER_NORMALS
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
NOR, ABN, VAL, INK, MUTED = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575"
D = pd.read_csv(os.path.join(_d, "3_단계", "2_EDA", "2-14_1123_울림모드경계", "figs", "impacts.csv"))
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")

# %% 파일별 물리 특징 (모든 190개, 같은 정의)
def desc(d):
    t = np.sort(d.t.values); f = d.f50.values; h = d.hg.values
    ev = [[t[0]]]
    for v in t[1:]: (ev[-1].append(v) if v - ev[-1][-1] < .020 else ev.append([v]))
    st = np.array([e[0] for e in ev]); iv = np.diff(st)
    return dict(충격=len(t), 사건=len(ev), 쌍비율=np.mean([len(e) >= 2 for e in ev]),
                간격CV=iv.std() / iv.mean() if len(iv) >= 2 else np.nan,
                공유모드비율=np.mean((f >= 1600) & (f < 2200)), 연결부비율=np.mean((f >= 2200) & (f < 4500)),
                본체비율=np.mean((f >= 5500) & (f < 8000)), 최대세기g=h.max())
P = pd.DataFrame({fn: desc(d) for fn, d in D.groupby("file")}).T
P["grp"] = np.where(P.index.str.startswith("val"), "val", np.where(P.index.str.startswith("abn"), "고장", "정상"))
TR = P[P.grp != "val"].copy(); TR["y"] = (TR.grp == "고장").astype(int)
v = P.loc["val_025"]; say("V025 특징:", v.drop("grp").round(3).to_dict())
say(f"train 충격 있는 파일 {len(TR)}개 (고장 {TR.y.sum()}), 충격 없는 train 파일은 제외")

# %% ① ~ ⑤ 특징별: V025와 같은 쪽에 있는 train 파일의 고장 비율
CONDS = {
    "① 울림 대부분이 1.6–2.2k (공유 모드 비율 ≥ 0.5)": TR.공유모드비율 >= .5,
    "① ′ 2.2–4.5k 연결부 울림 없음": TR.연결부비율 == 0,
    "② 쌍 충격 사건 비율 ≥ 0.5": TR.쌍비율 >= .5,
    "③ 사건 3개 이상 + 간격 CV < 0.1": (TR.사건 >= 3) & (TR.간격CV < .1),
    "④ 최대 세기 ≤ 0.3 g": TR.최대세기g <= .3,
    "⑤ 충격 3–8개": TR.충격.between(3, 8),
    "① + ④ + ⑤": (TR.공유모드비율 >= .5) & (TR.최대세기g <= .3) & TR.충격.between(3, 8),
    "② + ③": (TR.쌍비율 >= .5) & (TR.사건 >= 3) & (TR.간격CV < .1),
}
R = []
for k, m in CONDS.items():
    s = TR[m]; R.append(dict(조건=k, 파일=len(s), 고장=int(s.y.sum()), 고장비율=f"{s.y.mean() * 100:.0f} %" if len(s) else "-",
                            해당=" ".join(short(i) for i in s.index[:12]) + (" …" if len(s) > 12 else "")))
R = pd.DataFrame(R); say("\n특징별 (기저 고장 비율 20 %)"); say(R.to_string(index=False))

# %% 합친 거리: 특징 5개를 train 정상의 순위(백분위)로 맞춘 뒤 유클리드
FE = ["공유모드비율", "연결부비율", "본체비율", "쌍비율", "간격CV", "최대세기g", "충격"]
Q = P[FE].astype(float).copy(); Q["간격CV"] = Q.간격CV.fillna(1.0); Q["충격"] = np.log1p(Q.충격); Q["최대세기g"] = np.log10(Q.최대세기g)
N0 = Q.loc[TR.index[TR.y == 0]]
Z = (Q - N0.median()) / (N0.quantile(.75) - N0.quantile(.25)).replace(0, 1)
dd = np.sqrt(((Z.loc[TR.index] - Z.loc["val_025"]) ** 2).sum(1)).sort_values()
nn = dd.head(10); say("\n특징 7개 거리로 가까운 train 10개: " + " · ".join(f"{short(i)}({d:.2f})" for i, d in nn.items()) + f" → 고장 {int(TR.y[nn.index].sum())}/10")
say("  가까운 5개: 고장 %d/5" % int(TR.y[nn.index[:5]].sum()))
say(TR.loc[nn.index, FE + ["grp"]].round(3).to_string())

# %% 그림: 공유 모드 비율 × 연결부 비율 (점 크기 = 충격 수), 쌍 충격 · 규칙 간격 표시
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
ax = axs[0]
for g, c in (("정상", NOR), ("고장", ABN)):
    s = TR[TR.grp == g]; ax.scatter(s.공유모드비율 + np.random.default_rng(0).normal(0, .01, len(s)), s.연결부비율, s=12 + 4 * s.충격.clip(0, 40), color=c, alpha=.55, edgecolor="white", lw=.4, label=f"train {g}")
ax.scatter(v.공유모드비율, v.연결부비율, s=260, marker="*", color=VAL, edgecolor=INK, lw=1, label="V025", zorder=5)
for i in nn.index[:5]: ax.annotate(short(i), (TR.공유모드비율[i], TR.연결부비율[i]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("1.6–2.2k (1.9k 공유 모드) 울림 비율", color=MUTED); ax.set_ylabel("2.2–4.5k (연결부) 울림 비율", color=MUTED); ax.grid(alpha=.2); ax.legend(fontsize=9)
ax.set_title("어느 모드로 울렸나 — V025는 공유 모드만, 연결부 0 (점 크기 = 충격 수)", loc="left", fontsize=10.5)
ax = axs[1]
for g, c in (("정상", NOR), ("고장", ABN)):
    s = TR[TR.grp == g]; ax.scatter(s.간격CV.fillna(1.2), s.쌍비율 + np.random.default_rng(1).normal(0, .015, len(s)), s=18, color=c, alpha=.55, edgecolor="white", lw=.4, label=f"train {g}")
ax.scatter(v.간격CV, v.쌍비율, s=260, marker="*", color=VAL, edgecolor=INK, lw=1, label="V025", zorder=5)
for i in nn.index[:5]: ax.annotate(short(i), (TR.간격CV.fillna(1.2)[i], TR.쌍비율[i]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.set_xscale("log"); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"));ax.set_xlabel("사건 간격 CV (작을수록 규칙적, 사건 2개 이하 = 1.2)", color=MUTED); ax.set_ylabel("쌍 충격 사건 비율", color=MUTED); ax.grid(alpha=.2); ax.legend(fontsize=9)
ax.set_title("언제 · 어떻게 쳤나 — 규칙적 쌍 충격", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "V025_닮은파일.png"), dpi=140); plt.close(fig)
OUT.close()
