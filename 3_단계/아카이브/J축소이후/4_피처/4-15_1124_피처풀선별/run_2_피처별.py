# %% [markdown]
# 4-15 ② 정상 증거 · 고장 증거 지도 + 후보 피처별 EDA (run.py 결과를 읽음, 모델 없음)
# 1 꼬리 지도: 모든 풀 피처 + 지금 ET 입력(J-축소 + RM)의 고장 꼬리 × 정상 꼬리
# 2 후보마다 폴더 `피처_<이름>/figs/`: 분포(정상 · 고장 · val, 정상 95 % · 고장 5 % 선) · H1(충격 수)와의 관계 · 헷갈리는 파일 값
# 실행: conda run -n qe-phm python run_2_피처별.py

# %% 설정
import os, sys, warnings, json
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.exp import OUTLIER_NORMALS
from src.exp2util import NS
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
NOR, ABN, VAL, INK, MUTED = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575"
ns = NS(); y = ns.y; names = list(ns.names); nor = y == 0; n = len(y)
S = pd.read_csv(os.path.join(FIG, "풀_전체지표.csv")); SEL = json.load(open(os.path.join(FIG, "선별.json"), encoding="utf-8"))
HARD = SEL["hard"]; KEEP = SEL["keep"]; OK = SEL["ok"]
# 관문 (run.py 단계 2와 같은 규칙. 풀_전체지표.csv는 관문 전에 저장됨)
g1 = (S.분리도 >= .45) | (S.고장꼬리 >= .30) | (S.정상꼬리 >= .30); g2 = S.기존최대ρ < .8; g3 = (S.헷갈림분리도 >= .30) & S.헷갈림방향같음
S["관문"] = np.select([~g1, ~g2, ~g3], ["① 약함", "② 기존과 중복", "③ 헷갈리는 파일 못 가름"], "통과")
S.loc[(S.관문 == "통과") & ~S.피처.isin(KEEP), "관문"] = "④ 후보끼리 중복"
S.round(3).to_csv(os.path.join(FIG, "풀_전체지표.csv"), index=False, encoding="utf-8-sig")
CT = pd.read_csv(os.path.join(FIG, "후보_train.csv")); CV = pd.read_csv(os.path.join(FIG, "후보_val.csv"))
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")

# 지금 ET 입력의 꼬리 (같은 정의)
RM = ["RM_n_joint", "RM_e_joint", "RM_e_high"]
from src.exp import cached_table
from src.impacts import ring_mode_table
T = pd.concat([ns.T.reset_index(drop=True), cached_table("ringmode_train", ring_mode_table, ns.X)], axis=1)
def tails(v):
    a = roc_auc_score(y, v); w = v if a >= .5 else -v
    return abs(a - .5) * 2, (w[y == 1] > np.quantile(w[nor], .95)).mean(), (w[nor] < np.quantile(w[y == 1], .05)).mean()
B = pd.DataFrame([dict(피처=c, **dict(zip(("분리도", "고장꼬리", "정상꼬리"), tails(T[c].values.astype(float))))) for c in list(ns.FIXED) + RM])
B.round(3).to_csv(os.path.join(FIG, "지금입력_꼬리.csv"), index=False, encoding="utf-8-sig")

# %% 그림 1: 꼬리 지도
fig, ax = plt.subplots(figsize=(11, 8.5))
GC = {"① 약함": "#d0d5dc", "② 기존과 중복": "#9aa3b0", "③ 헷갈리는 파일 못 가름": "#c9b6e8", "④ 후보끼리 중복": "#f2c14e", "통과": "#e8443a"}
for g, c in GC.items():
    s = S[S.관문 == g]; ax.scatter(s.고장꼬리, s.정상꼬리, s=18 + 60 * s.분리도, color=c, alpha=.8, edgecolor="white", lw=.4, label=f"풀: {g} ({len(s)})")
ax.scatter(B.고장꼬리, B.정상꼬리, s=90, marker="s", facecolor="none", edgecolor=INK, lw=1.4, label="지금 ET 입력 (J-축소 + RM)")
for r in B.itertuples(): ax.annotate(r.피처.split("_")[0], (r.고장꼬리, r.정상꼬리), fontsize=7.5, xytext=(4, -9), textcoords="offset points", color=INK)
for r in S[S.관문 == "통과"].itertuples(): ax.annotate(r.피처, (r.고장꼬리, r.정상꼬리), fontsize=8.5, xytext=(5, 4), textcoords="offset points", color="#c0392b", weight="bold")
ax.plot([0, .7], [0, .7], color=MUTED, ls=":", lw=1)
ax.set_xlabel("고장 꼬리 = 고장 중 '정상 95 % 선 너머' 비율 → 고장을 뜻하는 피처", color=INK)
ax.set_ylabel("정상 꼬리 = 정상 중 '고장 5 % 선 너머' 비율 → 정상을 뜻하는 피처", color=INK)
ax.set_title("정상 증거 × 고장 증거 — 점 크기 = 분리도. 오른쪽 = 고장을 확실히 짚음, 위 = 정상을 확실히 짚음", loc="left", fontsize=10.5)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_꼬리지도.png"), dpi=140); plt.close(fig)

# %% 후보별 폴더 그림
names_v = list(CV.file); OUT = set(OUTLIER_NORMALS)
H1 = T["H1_nimp"].values; H1v = ns.Tv["H1_nimp"].values
for c in KEEP:
    D = os.path.join(HERE, "피처_" + c.replace(":", "_")); os.makedirs(os.path.join(D, "figs"), exist_ok=True)
    v = CT[c].values.astype(float); vv = CV[c].values.astype(float); a = roc_auc_score(y, v); w = 1 if a >= .5 else -1
    q95 = np.quantile(v[nor], .95 if w > 0 else .05); q05 = np.quantile(v[y == 1], .05 if w > 0 else .95)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.2, 1, 1.1]))
    ax = axs[0]; rng = np.random.default_rng(0)
    for k, (g, vals, col) in enumerate((("정상", v[nor], NOR), ("고장", v[y == 1], ABN), ("val", vv, VAL))):
        ax.scatter(vals, k + rng.normal(0, .08, len(vals)), s=16, color=col, alpha=.6, edgecolor="white", lw=.3)
    o = [names.index(f) for f in OUT]; ax.scatter(v[o], np.zeros(len(o)), marker="D", s=50, facecolor="none", edgecolor=INK, label="이상치 정상 ◆")
    ax.axvline(q95, color=NOR, ls="--", lw=1.2, label="정상 95 % 선 (너머 = 고장 꼬리)"); ax.axvline(q05, color=ABN, ls="--", lw=1.2, label="고장 5 % 선 (너머 = 정상 꼬리)")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["정상 120", "고장 30", "val 40"]); ax.legend(fontsize=7.5, loc="best"); ax.grid(alpha=.2)
    r = S.set_index("피처").loc[c]
    ax.set_title(f"분포 — 분리도 {r.분리도:.2f} · 고장 꼬리 {r.고장꼬리:.2f} · 정상 꼬리 {r.정상꼬리:.2f}", loc="left", fontsize=10)
    ax = axs[1]
    for g, m, col in (("정상", nor, NOR), ("고장", y == 1, ABN)): ax.scatter(np.expm1(H1[m]), v[m], s=18, color=col, alpha=.6, edgecolor="white", lw=.3, label=g)
    ax.scatter(np.expm1(H1v), vv, s=22, marker="^", color=VAL, alpha=.5, label="val")
    ax.set_xscale("symlog"); ax.set_xlabel("충격 수 (H1)", color=MUTED); ax.set_ylabel(c, color=MUTED); ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax.set_title(f"충격 수와의 관계 — 기존 최대 |ρ| {r.기존최대ρ:.2f} ({r.닮은기존})", loc="left", fontsize=10)
    ax = axs[2]; hv = [(f, v[names.index(f)]) for f in HARD]
    pct = [(f, (v[nor] < x).mean() * 100 if w > 0 else (v[nor] > x).mean() * 100) for f, x in hv]
    pct.sort(key=lambda t: (t[0][:3], -t[1]))
    ax.barh([short(f) for f, _ in pct], [p for _, p in pct], color=[ABN if f.startswith("abn") else NOR for f, _ in pct])
    ax.axvline(95, color=INK, ls=":", lw=1); ax.set_xlim(0, 100); ax.invert_yaxis(); ax.tick_params(labelsize=8)
    ax.set_title("헷갈리는 파일의 정상 백분위 (고장 방향) — 주황 = 고장(높을수록 좋음), 파랑 = 정상(낮을수록 좋음)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(D, "figs", "1_분포·관계·헷갈림.png"), dpi=140); plt.close(fig)
    pd.DataFrame(pct, columns=["파일", "정상백분위_고장방향"]).round(1).to_csv(os.path.join(D, "figs", "헷갈리는파일.csv"), index=False, encoding="utf-8-sig")
print("done")
