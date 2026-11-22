# %% [markdown]
# PHO-002 추가 (11-22 사용자 요청): 기준 W에 "중앙값 4등분 K_max" 1열만 추가한 quick_cv (5-fold × seed 3, PR-AUC)
# K_max4 = 커토그램 169대역 각각에서 2초를 4등분한 포락선 첨도의 중앙값 → 그 최댓값 (log10). 라벨 없음, run.py 캐시 사용
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from src.exp import load_train
from src.models2 import Two
from src.evaluate import quick_cv, separation
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(_d, "outputs", "cache")
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; g = np.arange(len(y)); nor = y == 0
z = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T, L, fL = z["T"].item(), z["L"], z["f"]
K4 = np.log10(np.maximum(np.load(os.path.join(CACHE, "pho002_kgram.npz"))["Km"].max(1), 1e-3))
H = np.column_stack([np.asarray(T[c], float) for c in FIXED]); S = np.asarray(T["L1_S"], float); df = float(fL[1] - fL[0]); nf = len(FIXED)
mk = lambda fx, k: (lambda: Two("lin", (), fixed=tuple(fx), nf=k, df=df, peak=tuple(PEAK)))
out = {}
for tag, F, m in (("기준 W", np.column_stack([H, S, L]), mk(FIXED, nf)), ("K4 +중앙값4 K_max", np.column_stack([H, K4, S, L]), mk(FIXED + ["K_max4"], nf + 1))):
    q = quick_cv(F, y, g, m); o = q["oof"]; sp = separation(o, y, names); ab = o[:, y == 1]
    fl = {k: float(np.mean([(ab[s] <= o[s, names.index(k)]).mean() for s in range(len(o))])) for k in ("nor_116", "nor_065", "nor_059", "nor_025")}
    rk = {k: q["miss_rank"][names.index(k)] for k in ("abn_002", "abn_014", "abn_017", "abn_022")}
    out[tag] = dict(pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"], fault_rank=rk, normal_faultlike=fl)
    print(f"{tag:18s} PR-AUC {out[tag]['pr_auc']:.3f} ± {out[tag]['pr_sd']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 고장 순위 "
          + " ".join(f"{k[-3:]} {v:.2f}" for k, v in rk.items()) + " | 정상의 고장다움 " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in fl.items()))
pct = lambda i: (K4[nor] < K4[i]).mean() * 100
print("K_max4 정상 중 백분위: " + " ".join(f"{k} {pct(names.index(k)):.0f}%" for k in ("abn_002", "abn_014", "abn_017", "abn_022", "nor_116", "nor_065", "nor_059", "nor_025")))
json.dump(out, open(os.path.join(HERE, "result_k4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
