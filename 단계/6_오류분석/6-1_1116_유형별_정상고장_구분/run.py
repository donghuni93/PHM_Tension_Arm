# %% [markdown]
# 6-1 유형별 정상·고장 구분: 파일 150개의 유형(2-0) · 전문가 점수(OOF) · 규칙 축 · 중첩 CV 오류 횟수를 한 표로 모은다
# - 새 피처·새 규칙을 만들지 않는다. 이미 있는 값을 모아 "어떤 유형이 왜 헷갈리는가"를 보기 위한 집계다 (실험 아님)
# - 입력: 2-0 figs/types.csv · 2-15 figs/events.csv · EXP-030 result.json · 피처 캐시 · 전문가 캐시(5-fold seed 0)
# - 출력: figs/table.csv (파일별) · figs/summary.txt (유형별 집계와 헷갈리는 파일 목록)
# 실행: conda run -n qe-phm python "단계/6_오류분석/6-1_1116_유형별_정상고장_구분/run.py"

# %%
import os, sys, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import ScoreStack, KnockRule, ae_mean, et_logit, peak83_table, _knock, _hits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]
T = pd.read_csv(os.path.join(_d, "단계/2_EDA/2-1_고장유형분류/figs/types.csv"), index_col=0).loc[names]
D = pd.DataFrame({"y": y, "type": T["type"].fillna("U").values, "peak_g": T["max_hp"].values, "n_imp": T["n_imp_hp"].values,
                  "s20": T["s20"].values, "kurt_hp": T["kurt_hp"].values, "e200": T["e200"].values, "hi_ratio": T["hi_ratio"].values}, index=names)
Z = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(experts=(ae_mean, et_logit), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
P83, W83, _ = peak83_table(); kn = KnockRule(2.0, 0.7).fit(g); H = _hits()
D["ae_z"], D["et_z"] = Z[:, 0], Z[:, 1]
D["P83"], D["W83"], D["r"], D["K3"] = P83, W83, kn.r(g), _knock()[:, 2]
D["rule83"] = (P83 >= 0.75) & (W83 >= 150); D["knock"] = kn(g)
D["n20"], D["S"] = H["HIT_n20"].values, H["LOCK_S"].values
ev_p = os.path.join(_d, "단계/2_EDA/아카이브/개관/2-15_1116_고역통과충격수오탐/figs/events.csv")
EV = pd.read_csv(ev_p, index_col=0)
ev_cols = list(EV.columns)
for c in EV.columns:
    if c.lower() in ("fc", "fc_khz", "decay", "decay_ms", "arm"):
        D["ev_" + c] = EV[c].reindex(names).values
res = json.load(open(os.path.join(_d, "단계/5_모델비교/아카이브/EXP-030_1116_최종후보고정_규칙감사/result.json"), encoding="utf-8"))["configs"]
for tag, key in (("AE", "중첩: L0 AE 단독"), ("ET", "중첩: L1 ET 단독"), ("결합", "중첩: L2 AE+ET 로지스틱 (EXP-014)"), ("최종", "중첩: A + 유격 반복 규칙 (EXP-025)")):
    c = res[key]; err = {**c["miss_counts"], **c["fp_counts"]}
    D["err_" + tag] = [err.get(n, 0) for n in names]
D.round(3).to_csv(os.path.join(HERE, "figs", "table.csv"), encoding="utf-8-sig")

# %% 요약
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")
nor = D[D.y == 0]; q = lambda c, p: np.percentile(nor[c], p)
L = ["events.csv 열: " + ", ".join(ev_cols), "",
     f"정상 분위 (중앙 · 90% · 95% · 최대): AE z {q('ae_z',50):.2f} · {q('ae_z',90):.2f} · {q('ae_z',95):.2f} · {nor.ae_z.max():.2f} | ET z {q('et_z',50):.2f} · {q('et_z',90):.2f} · {q('et_z',95):.2f} · {nor.et_z.max():.2f}"
     f" | Peak {q('peak_g',50):.2f} · {q('peak_g',95):.2f} g | 2위/1위 {q('K3',50):.2f} | r {q('r',95):.2f} | 큰 타격 수 중앙 {q('n20',50):.0f}", ""]
L.append("[유형 × 라벨] 개수 · AE z 중앙 · ET z 중앙 · Peak 중앙 · 큰 타격 수 중앙 · 2위/1위 중앙 | 최종 구성 오류 횟수 합(5번 중) | 결합만(규칙 없음) 오류 합")
for t in ("L4", "L3", "L2", "L1", "U"):
    for lab, nm in ((1, "고장"), (0, "정상")):
        d = D[(D.type == t) & (D.y == lab)]
        if len(d):
            L.append(f"  {t} {nm} {len(d):3d}개 · AE {d.ae_z.median():+.2f} · ET {d.et_z.median():+.2f} · Peak {d.peak_g.median():.2f} g · n20 {d.n20.median():.0f} · R {d.K3.median():.2f}"
                     f" | 최종 {int(d.err_최종.sum())} | 결합 {int(d.err_결합.sum())} | 8.3k {int(d.rule83.sum())} · 유격 {int(d.knock.sum())}")
cols = ["type", "ae_z", "et_z", "peak_g", "n20", "K3", "r", "P83", "W83", "S", "s20"] + [c for c in D.columns if c.startswith("ev_")] + ["rule83", "knock", "err_AE", "err_ET", "err_결합", "err_최종"]
def rows(title, d):
    L.append(""); L.append(title); L.append("  파일 | " + " | ".join(cols))
    for n, r in d.iterrows():
        L.append(f"  {s(n)} | " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(int(r[c]) if isinstance(r[c], (bool, np.bool_, np.integer)) else r[c]) for c in cols))
rows("[고장 30개] 최종 오류·결합 오류 많은 순", D[D.y == 1].sort_values(["err_최종", "err_결합", "ae_z"], ascending=[False, False, True]))
rows("[정상] 어느 구성에서든 오경보 2번 이상이거나 규칙 문턱 20% 이내", D[(D.y == 0) & ((D[["err_AE", "err_ET", "err_결합", "err_최종"]].max(1) >= 2)
     | (np.minimum(D.P83 / 0.75, D.W83 / 150) >= 0.8) | (np.minimum(D.r / 2, D.K3 / 0.7) >= 0.8))].sort_values(["err_최종", "err_결합", "err_ET", "err_AE"], ascending=False))
open(os.path.join(HERE, "figs", "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
