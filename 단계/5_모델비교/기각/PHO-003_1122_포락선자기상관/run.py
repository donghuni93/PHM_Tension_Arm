# %% [markdown]
# PHO-003 (휴대폰 세션, 11-22) 포락선 자기상관 반복성 1열을 기준 W에 추가 → quick_cv (5-fold × seed 3, PR-AUC)
# ACF_rep (라벨 없음, 실행 전 고정): 평균 제거 → 1 kHz 고역 → |hilbert| → 1000 Hz 저역 → 1 kHz로 솎기 → 평균 제거 → 정규화 자기상관, 지연 50 ms–1.2 s의 최댓값
# 근거: 같은 세션 EDA에서 단독 AUC 0.61, 기존 피처(H1 · H3 · H5 · 박자 S · K_max4)와 |ρ| ≤ 0.25, 오경보 116 · 065를 정상 쪽(1 % · 14 %)으로 보냄
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from src.exp import load_train
from src.models2 import Two
from src.evaluate import quick_cv, separation
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(_d, "outputs", "cache"); FS = 25000
LOG = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; g = np.arange(len(y)); nor = y == 0
z = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T, L, fL = z["T"].item(), z["L"], z["f"]
hp = butter(4, 1000, btype="high", fs=FS, output="sos"); lp = butter(4, 1000, fs=FS, output="sos")
def acf_rep(x):
    e = sosfiltfilt(lp, np.abs(hilbert(sosfiltfilt(hp, x - x.mean()))))[::25]; e = e - e.mean(); e = e / (np.sqrt((e ** 2).sum()) + 1e-12)
    return float(np.correlate(e, e, "full")[len(e) - 1:][50:1201].max())
A = np.array(Parallel(n_jobs=-1)(delayed(acf_rep)(x) for x in X))
say(f"ACF_rep 단독 AUC {roc_auc_score(y, A):.2f}")
H = np.column_stack([np.asarray(T[c], float) for c in FIXED]); S = np.asarray(T["L1_S"], float); df = float(fL[1] - fL[0]); nf = len(FIXED)
mk = lambda fx, k: (lambda: Two("lin", (), fixed=tuple(fx), nf=k, df=df, peak=tuple(PEAK)))
out = {}
for tag, F, m in (("기준 W", np.column_stack([H, S, L]), mk(FIXED, nf)), ("+ACF_rep", np.column_stack([H, A, S, L]), mk(FIXED + ["ACF_rep"], nf + 1))):
    q = quick_cv(F, y, g, m); o = q["oof"]; sp = separation(o, y, names); ab = o[:, y == 1]
    fl = {k: float(np.mean([(ab[s] <= o[s, names.index(k)]).mean() for s in range(len(o))])) for k in ("nor_116", "nor_065", "nor_059", "nor_025")}
    rk = {k: q["miss_rank"][names.index(k)] for k in ("abn_002", "abn_014", "abn_017", "abn_022")}
    out[tag] = dict(pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"], fault_rank=rk, normal_faultlike=fl)
    say(f"{tag:10s} PR-AUC {out[tag]['pr_auc']:.3f} ± {out[tag]['pr_sd']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 고장 순위 "
        + " ".join(f"{k[-3:]} {v:.2f}" for k, v in rk.items()) + " | 정상의 고장다움 " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in fl.items()))
json.dump(dict(rule="quick_cv 5-fold × seed 3, PR-AUC", results=out), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
