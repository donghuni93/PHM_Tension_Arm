# %% [markdown]
# PHO-005 (휴대폰 세션, 11-22) 기준 W의 ET 자리만 다른 분류기로 교체 → quick_cv (5-fold × seed 3, PR-AUC, 임계값 없음)
# 입력·결합(Two 'lin': ET 안쪽 4겹 OOF logit + 박자 z → 로지스틱)은 그대로, src.models2.make_et만 이 스크립트 안에서 바꿔 끼운다(src/ 수정 없음)
#   1 ET (기준, 원래 make_et)
#   2 CatBoost depth 4 · l2 5 · 500회 · lr 0.05 · SqrtBalanced
#   3 RBF SVC: RobustScaler → SVC(C 1, gamma scale, probability, balanced)
#   4 GPC: RobustScaler → GaussianProcessClassifier(1.0 * RBF(1.0))
#   5 소프트 투표 5:5: ET + (3)
# 하이퍼파라미터 고정, 튜닝 없음. 스케일링은 파이프라인 안(폴드 안에서 fit). 확률 0/1은 Two의 lg가 1/600으로 잘라 logit이 유한
import os, sys, json, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import VotingClassifier
import src.models2 as M2
from src.exp import load_train
from src.models2 import Two
from src.evaluate import quick_cv, separation
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(_d, "outputs", "cache")
LOG = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()

# %% 데이터 · 기준 W 입력 (PHO-004와 같음)
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; g = np.arange(len(y))
z = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T, L, fL = z["T"].item(), z["L"], z["f"]
H = np.column_stack([np.asarray(T[c], float) for c in FIXED]); S = np.asarray(T["L1_S"], float); df = float(fL[1] - fL[0]); nf = len(FIXED)
F = np.column_stack([H, S, L])
mk = lambda: Two("lin", (), fixed=tuple(FIXED), nf=nf, df=df, peak=tuple(PEAK))

# %% 바꿔 끼울 분류기 (make_et와 같은 모양: make(seed=..)() → 추정기)
ET0 = M2.make_et
def cat(seed=0, **k):
    from catboost import CatBoostClassifier
    return lambda: CatBoostClassifier(depth=4, l2_leaf_reg=5, iterations=500, learning_rate=0.05, auto_class_weights="SqrtBalanced",
                                      random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False)
svc = lambda seed=0, **k: (lambda: make_pipeline(RobustScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)))
gpc = lambda seed=0, **k: (lambda: make_pipeline(RobustScaler(), GaussianProcessClassifier(1.0 * RBF(1.0), random_state=seed)))
vote = lambda seed=0, **k: (lambda: VotingClassifier([("et", ET0(seed=seed)()), ("svc", svc(seed=seed)())], voting="soft"))
CFG = [("ET (기준)", ET0), ("CatBoost", cat), ("RBF SVC", svc), ("GPC", gpc), ("투표 ET+SVC", vote)]
try:
    import catboost  # noqa
except ImportError:
    CFG = [c for c in CFG if c[0] != "CatBoost"]; say("catboost 없음 → CatBoost 구성 건너뜀")

# %% 실행
out = {}
for tag, make in CFG:
    M2.make_et = make; t0 = time.time()
    try:
        q = quick_cv(F, y, g, mk)
    finally:
        M2.make_et = ET0
    sec = time.time() - t0; o = q["oof"]; sp = separation(o, y, names); ab = o[:, y == 1]
    fl = {k: float(np.mean([(ab[s] <= o[s, names.index(k)]).mean() for s in range(len(o))])) for k in ("nor_116", "nor_065", "nor_059", "nor_025")}
    rk = {k: q["miss_rank"][names.index(k)] for k in ("abn_002", "abn_014", "abn_017", "abn_022")}
    out[tag] = dict(pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), roc_auc=float(q["roc_auc"].mean()),
                    risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"], fault_rank=rk, normal_faultlike=fl, sec=round(sec, 1))
    say(f"{tag:10s} PR-AUC {out[tag]['pr_auc']:.3f} ± {out[tag]['pr_sd']:.3f} · ROC {out[tag]['roc_auc']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 고장 순위 "
        + " ".join(f"{k[-3:]} {v:.2f}" for k, v in rk.items()) + " | 정상의 고장다움 " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in fl.items()) + f" | {sec:.0f} s")
json.dump(dict(rule="quick_cv 5-fold × seed 3, PR-AUC, ET 자리만 교체 (Two 'lin')", results=out), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
