# %% [markdown]
# 25년 3조 파이프라인 재현 → 4_outputs/results_3조.json
# - 노트북(Final_Submission-1.ipynb)의 load_signal · extract_features · 4구간 피처를 그대로 옮김 (28개)
# - ① 3조 방식 5-fold (검증 폴드에서 임계값 고름 = 누수) ② 같은 피처·SVM_C2를 우리 중첩 CV로
#   ③ 파일별 OOF 확률 ④ val 40개 확률 (노트북 출력과 대조) ⑤ 우리(26) 데이터에 그대로 적용
# 실행: conda run -n qe-phm python 1_참고자료/25년3조_비교/pipeline_3조.py

# %% 설정
import os, sys, json, re, warnings
from copy import deepcopy
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, precision_recall_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from load25 import load25, SRC
from src.exp import load_train
from src.evaluate import nested_cv
warnings.filterwarnings("ignore")

# %% 3조 코드 (노트북 셀 4 그대로. FS = 5000 가정, 최대 |x|로 나눔)
FS = 5000
TOTAL_LEN = 50000
FREQ_BANDS = [(0, 500), (500, 2000), (2000, 5000)]
n_seg = 4


def load_signal(s):                     # 원본: 파일 경로 → 여기서는 배열을 받음 (나머지 같음)
    s = s - np.mean(s)
    s = np.pad(s, (0, TOTAL_LEN - len(s))) if len(s) < TOTAL_LEN else s[:TOTAL_LEN]
    return s / (np.max(np.abs(s)) + 1e-8)


def extract_features(sig):
    sig = sig - np.mean(sig)
    abs_mean = np.mean(np.abs(sig)); abs_max = np.max(np.abs(sig))
    rms = np.sqrt(np.mean(sig ** 2)); var = np.var(sig)
    sk = skew(sig); ku = kurtosis(sig, fisher=False)
    crest = abs_max / (rms + 1e-8); impulse = abs_max / (abs_mean + 1e-8); shape = rms / (abs_mean + 1e-8)
    p2p = np.ptp(sig)
    sig_win = sig[int(5.0 * FS):int(7.0 * FS)]
    abs_mean_w = np.mean(np.abs(sig_win)); abs_max_w = np.max(np.abs(sig_win))
    rms_w = np.sqrt(np.mean(sig_win ** 2)); var_w = np.var(sig_win)
    ku_w = kurtosis(sig_win, fisher=False)
    crest_w = abs_max_w / (rms_w + 1e-8); impulse_w = abs_max_w / (abs_mean_w + 1e-8); p2p_w = np.ptp(sig_win)
    N = len(sig)
    yf = np.abs(rfft(sig)) / N; xf = rfftfreq(N, 1 / FS)
    total_power = np.sum(yf ** 2)
    band_ratios = [np.sum(yf[(xf >= lo) & (xf < hi)] ** 2) / (total_power + 1e-8) for lo, hi in FREQ_BANDS]
    f = {"abs_mean": abs_mean, "abs_max": abs_max, "rms": rms, "var": var, "skewness": sk, "kurtosis": ku,
         "peak_to_peak": p2p, "crest_factor": crest, "impulse_factor": impulse, "shape_factor": shape,
         "band_low_ratio": band_ratios[0], "band_mid_ratio": band_ratios[1], "band_high_ratio": band_ratios[2],
         "dom_freq": xf[np.argmax(yf)], "rms_win": rms_w, "var_win": var_w, "kurtosis_win": ku_w,
         "crest_factor_win": crest_w, "impulse_factor_win": impulse_w, "peak_to_peak_win": p2p_w}
    seg_len = len(sig) // n_seg                       # 4구간 RMS·첨도 (셀 4 뒷부분)
    for i in range(n_seg):
        seg = sig[i * seg_len:(i + 1) * seg_len if i < n_seg - 1 else len(sig)]
        f[f"rms_seg{i+1}"] = float(np.sqrt(np.mean(seg ** 2)))
        f[f"kurt_seg{i+1}"] = float(kurtosis(seg, fisher=False, bias=False))
    return f


def table(X):
    T = pd.DataFrame([extract_features(load_signal(x)) for x in X])
    return T


RSEED = 42
imp = lambda: ("imputer", SimpleImputer(strategy="median"))
sc = lambda: ("scaler", StandardScaler())
svm = lambda C, g="scale", prob=True: Pipeline([imp(), sc(), ("clf", SVC(kernel="rbf", C=C, gamma=g, probability=prob, class_weight="balanced", random_state=RSEED))])
MODELS = {
    "LogReg": Pipeline([imp(), sc(), ("clf", LogisticRegression(max_iter=200, class_weight="balanced", random_state=RSEED))]),
    "SVM_C1": svm(1.0), "SVM_C2": svm(2.0), "SVM_C5": svm(5.0), "SVM_C2_g0_1": svm(2.0, 0.1), "SVM_C2_g0_01": svm(2.0, 0.01),
    "RF_200": Pipeline([imp(), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RSEED))]),
    "RF_400_md8": Pipeline([imp(), ("clf", RandomForestClassifier(n_estimators=400, max_depth=8, class_weight="balanced", random_state=RSEED))]),
    "RF_400_md12": Pipeline([imp(), ("clf", RandomForestClassifier(n_estimators=400, max_depth=12, class_weight="balanced", random_state=RSEED))]),
    "GradientBoost": Pipeline([imp(), ("clf", GradientBoostingClassifier(random_state=RSEED))]),
}


def tune_threshold(y_true, proba, beta=2):           # 노트북 그대로: 검증 폴드 정답으로 임계값을 고른다
    ps, rs, ths = precision_recall_curve(y_true, proba)
    pr_auc = auc(rs, ps)
    best_t, best_f2 = 0.5, -1
    for t in ths:
        f2 = fbeta_score(y_true, (proba >= t).astype(int), beta=beta)
        if f2 > best_f2:
            best_t, best_f2 = t, f2
    P, R, F2 = precision_recall_fscore_support(y_true, (proba >= best_t).astype(int), beta=beta, average="binary")[:3]
    return best_t, P, R, F2, pr_auc


def cv_3jo(X_all, y_all):
    """노트북 셀 5: 모델 10종 5-fold, 폴드마다 검증 폴드로 임계값 튜닝"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
    out = {}
    for name, base in MODELS.items():
        r = {k: [] for k in ["P", "R", "F2", "thr", "PR_AUC"]}
        for tr, va in skf.split(X_all, y_all):
            pipe = deepcopy(base).fit(X_all[tr], y_all[tr])
            proba = pipe.predict_proba(X_all[va])[:, 1]
            t, P, R, F2, pa = tune_threshold(y_all[va], proba)
            for k, v in zip(["thr", "P", "R", "F2", "PR_AUC"], [t, P, R, F2, pa]):
                r[k].append(float(v))
        out[name] = dict(F2=np.mean(r["F2"]), F2_sd=np.std(r["F2"]), P=np.mean(r["P"]), R=np.mean(r["R"]),
                         thr_med=float(np.median(r["thr"])), PR_AUC=np.mean(r["PR_AUC"]), folds=r)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["F2"]))


def nested(F, y):
    """우리 평가 규칙: 바깥 5 × 안쪽 4, seed 5. 임계값은 안쪽 OOF로만. 점수 = SVC decision_function"""
    r = nested_cv(F, y, np.arange(len(y)), lambda: svm(2.0, prob=False), n_jobs=-1)
    return dict(f2=float(r["f2"].mean()), f2_sd=float(r["f2"].std()), recall=float(r["recall"].mean()),
                precision=float(r["precision"].mean()), fpr=float(r["fpr"].mean()),
                miss=r["miss_counts"].tolist(), fp=r["fp_counts"].tolist())


def oof_proba(F, y, seeds=range(5)):
    """seed 5개 × 5-fold OOF 확률 (SVM_C2). 파일별 평균"""
    P = np.zeros((len(seeds), len(y)))
    for k, s in enumerate(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(F, y):
            P[k, te] = deepcopy(MODELS["SVM_C2"]).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    return P


def auc_table(T, y):
    return {c: float(roc_auc_score(y, T[c].replace([np.inf, -np.inf], np.nan).fillna(0))) for c in T.columns}


# %% 25 데이터
X25, y25, n25 = load25()
tr = y25 >= 0
T25 = table(X25)
FEATS = list(T25.columns)
F25 = np.nan_to_num(T25.values, nan=0.0, posinf=0.0, neginf=0.0)
Ftr, ytr = F25[tr], y25[tr]

res = dict(features=FEATS, names25=n25)
res["cv3_25"] = cv_3jo(Ftr, ytr)
best = next(iter(res["cv3_25"])); thr37 = res["cv3_25"][best]["thr_med"]
res["best"], res["thr_cv"], res["thr_submit"] = best, thr37, 0.20
res["nested_25"] = nested(Ftr, ytr)
P = oof_proba(Ftr, ytr)
res["oof_25"] = dict(mean=P.mean(0).tolist(), sd=P.std(0).tolist(),
                     miss_thr_cv=((P < thr37) & (ytr == 1)).sum(0).tolist(), fp_thr_cv=((P >= thr37) & (ytr == 0)).sum(0).tolist(),
                     miss_02=((P < .2) & (ytr == 1)).sum(0).tolist(), fp_02=((P >= .2) & (ytr == 0)).sum(0).tolist())

# val: 전체 train으로 학습 (노트북 best_pipe) → 확률
bp = deepcopy(MODELS[best]).fit(Ftr, ytr)
vp = bp.predict_proba(F25[~tr])[:, 1]
nb = json.load(open(os.path.join(SRC, "Final_Submission-1.ipynb"), encoding="utf-8"))
txt = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code" and "val_proba = best_pipe" in "".join(c["source"])
              and "print(val_proba)" in "".join(c["source"]) for o in c.get("4_outputs", []))
nbp = np.array([float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", txt)])
res["val"] = dict(names=[n for n, m in zip(n25, ~tr) if m], proba=vp.tolist(), proba_notebook=nbp.tolist(),
                  max_diff=float(np.abs(vp - nbp).max()) if len(nbp) == len(vp) else None,
                  pred_02=(vp >= 0.20).astype(int).tolist(), pred_cv=(vp >= thr37).astype(int).tolist(),
                  pred_02_notebook=(nbp >= 0.20).astype(int).tolist() if len(nbp) == len(vp) else None)
res["auc_25"] = auc_table(T25[tr].reset_index(drop=True), ytr)
res["feat25"] = {c: [None if not np.isfinite(v) else float(v) for v in T25[c]] for c in FEATS}

# %% 26 데이터 (우리 train 150개)에 3조 코드를 그대로 (FS = 5000 가정까지 그대로 → 실제로는 25 kHz × 2초)
X26, y26, n26 = load_train()
T26 = table(X26)
F26 = np.nan_to_num(T26.values, nan=0.0, posinf=0.0, neginf=0.0)
res["names26"] = n26
res["cv3_26"] = cv_3jo(F26, y26)
res["nested_26"] = nested(F26, y26)
res["auc_26"] = auc_table(T26, y26)
res["feat26"] = {c: [None if not np.isfinite(v) else float(v) for v in T26[c]] for c in FEATS}


# %% 저장 + 요약
def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(f"{float(o):.10g}")   # 유효숫자 10자리 (abs_max는 1e-6 단위 차이)
    if isinstance(o, np.integer): return int(o)
    return o


json.dump(jsonable(res), open(os.path.join(HERE, "4_outputs", "results_3조.json"), "w", encoding="utf-8"), ensure_ascii=False)
L = [f"best {best}, thr_cv {thr37:.3f}"]
for tag in ["25", "26"]:
    L.append(f"\n== {tag} 데이터: 3조 방식 5-fold (검증 폴드 임계값 튜닝) ==")
    for k, v in res[f"cv3_{tag}"].items():
        L.append(f"  {k:14s} F2 {v['F2']:.3f} ± {v['F2_sd']:.3f}  P {v['P']:.3f} R {v['R']:.3f} thr {v['thr_med']:.2f} PR-AUC {v['PR_AUC']:.3f}")
    n = res[f"nested_{tag}"]
    L.append(f"  중첩 CV SVM_C2: F2 {n['f2']:.3f} ± {n['f2_sd']:.3f}, 재현율 {n['recall']:.3f}, 정밀도 {n['precision']:.3f}, 오경보율 {n['fpr']:.3f}")
    names = n25[:150] if tag == "25" else n26
    L.append("  자주 놓친 고장(3/5+): " + " ".join(nm for nm, c in zip(names, n["miss"]) if c >= 3))
    L.append("  자주 오경보(3/5+): " + " ".join(nm for nm, c in zip(names, n["fp"]) if c >= 3))
v = res["val"]
L.append(f"\nval: 노트북 확률 {len(v['proba_notebook'])}개, 재현 최대 차이 {v['max_diff']}, thr 0.20 고장 예측 {sum(v['pred_02'])}개 (노트북 {sum(v['pred_02_notebook'] or [])})")
L.append("단변량 AUC 25 / 26: " + ", ".join(f"{c} {res['auc_25'][c]:.2f}/{res['auc_26'][c]:.2f}" for c in FEATS))
open(os.path.join(HERE, "4_outputs", "pipeline_log.txt"), "w", encoding="utf-8").write("\n".join(L))
print("done")
