# %% [markdown]
# PHO-004 (휴대폰 세션, 11-22) 100 ms 창 Top2/Top1 비율 1열을 기준 W에 추가 → quick_cv (5-fold × seed 3, PR-AUC)
# W2T1 (라벨 없음, 실행 전 고정): 평균 제거 → 8–12.4 kHz 대역통과(4차 양방향) → 2초를 100 ms 창 20개 → 창 RMS 2위 / 1위
# 근거(같은 세션 EDA): 단독 AUC 0.44로 구분력 없음, 기존 H2(충격 2위/1위)와 ρ 0.05. 오경보 116 · 065 · 095 · 027을 정상 쪽(0–12 %)으로, 059는 44 %, 고장 017은 50 %
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.exp import load_train
from src.models2 import Two
from src.evaluate import quick_cv, separation
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(_d, "outputs", "cache")
LOG = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; g = np.arange(len(y))
z = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T, L, fL = z["T"].item(), z["L"], z["f"]
b = butter(4, [8000, 12400], btype="band", fs=25000, output="sos")
def w2t1(x):
    w = np.sqrt((sosfiltfilt(b, x - x.mean())[:50000].reshape(20, 2500) ** 2).mean(1)); w = np.sort(w)[::-1]; return w[1] / w[0]
R = np.array([w2t1(x) for x in X])
H = np.column_stack([np.asarray(T[c], float) for c in FIXED]); S = np.asarray(T["L1_S"], float); df = float(fL[1] - fL[0]); nf = len(FIXED)
mk = lambda fx, k: (lambda: Two("lin", (), fixed=tuple(fx), nf=k, df=df, peak=tuple(PEAK)))
out = {}
for tag, F, m in (("기준 W", np.column_stack([H, S, L]), mk(FIXED, nf)), ("+W2T1", np.column_stack([H, R, S, L]), mk(FIXED + ["W2T1"], nf + 1))):
    q = quick_cv(F, y, g, m); o = q["oof"]; sp = separation(o, y, names); ab = o[:, y == 1]
    fl = {k: float(np.mean([(ab[s] <= o[s, names.index(k)]).mean() for s in range(len(o))])) for k in ("nor_116", "nor_065", "nor_059", "nor_095", "nor_027", "nor_025")}
    rk = {k: q["miss_rank"][names.index(k)] for k in ("abn_002", "abn_014", "abn_017", "abn_022")}
    out[tag] = dict(pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"], fault_rank=rk, normal_faultlike=fl)
    say(f"{tag:8s} PR-AUC {out[tag]['pr_auc']:.3f} ± {out[tag]['pr_sd']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 고장 순위 "
        + " ".join(f"{k[-3:]} {v:.2f}" for k, v in rk.items()) + " | 정상의 고장다움 " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in fl.items()))
json.dump(dict(rule="quick_cv 5-fold × seed 3, PR-AUC", results=out), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
