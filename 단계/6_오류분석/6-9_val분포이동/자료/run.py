# 6-10 (11-22) val 분포 이동 점검 — 적대적 검증(train 대 val 구분 AUC) · 열별 이동 · V021 원신호 수치. 라벨 없는 진단
# 실행: conda run -n qe-phm python run.py → 결과.txt
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
ROOT = _d
sys.path.insert(0, ROOT); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.signal import butter, sosfiltfilt, spectrogram
from src.exp2util import NS, tsne_space_val
OUT = os.path.join(os.path.dirname(__file__), "결과.txt"); L = []
p = lambda *a: L.append(" ".join(str(x) for x in a))
ns = NS(); y = ns.y; names = np.array(ns.names)
Zt, Zv, keep, cols = tsne_space_val(ns)
yk = y[keep]; nk = names[keep]
# ① 적대적 검증: train 정상 vs val, train 전체 vs val
def adv(A, B, tag):
    X = np.vstack([A, B]); d = np.r_[np.zeros(len(A)), np.ones(len(B))]
    res = []
    for s in range(5):
        for mdl in ("et", "lr"):
            m = ExtraTreesClassifier(300, min_samples_leaf=3, random_state=s, n_jobs=-1) if mdl == "et" else LogisticRegression(C=0.3, max_iter=2000)
            pr = cross_val_predict(m, X, d, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]
            res.append((mdl, roc_auc_score(d, pr), pr))
    for mdl in ("et", "lr"):
        a = [r[1] for r in res if r[0] == mdl]; p(f"{tag} [{mdl}] AUC {np.mean(a):.3f} ± {np.std(a):.3f}")
    return np.mean([r[2] for r in res if r[0] == "et"], 0)
H_idx = [i for i, c in enumerate(cols) if not c.startswith("AE")]; A_idx = [i for i, c in enumerate(cols) if c.startswith("AE")]
p("## ① 적대적 검증 (구분 AUC 0.5 = 같은 분포)")
for tag, idx in (("H+AE", list(range(len(cols)))), ("H만", H_idx), ("AE만", A_idx)):
    adv(Zt[yk == 0][:, idx], Zv[:, idx], f"train정상 vs val | {tag}")
    adv(Zt[:, idx], Zv[:, idx], f"train전체 vs val | {tag}")
# 기준: train 정상 vs train 고장
adv(Zt[yk == 0], Zt[yk == 1], "참고: train정상 vs train고장 | H+AE")
# ② 열별 이동: val 중앙값의 train 정상 백분위, val-정상 AUC
p("\n## ② 열별 이동 (val 중앙의 train 정상 백분위, val vs train정상 AUC, val vs train고장 AUC)")
rows = []
for j, c in enumerate(cols):
    a_n = roc_auc_score(np.r_[np.zeros((yk == 0).sum()), np.ones(len(Zv))], np.r_[Zt[yk == 0, j], Zv[:, j]])
    a_a = roc_auc_score(np.r_[np.zeros((yk == 1).sum()), np.ones(len(Zv))], np.r_[Zt[yk == 1, j], Zv[:, j]])
    a_f = roc_auc_score(yk, Zt[:, j])
    pc = (Zt[yk == 0, j] < np.median(Zv[:, j])).mean() * 100
    rows.append((abs(a_n - .5), c, pc, a_n, a_a, a_f))
rows.sort(reverse=True)
p("열 | val중앙 정상백분위 | AUC(val vs 정상) | AUC(val vs 고장) | AUC(고장 vs 정상, train)")
for _, c, pc, a_n, a_a, a_f in rows[:15]: p(f"{c} | {pc:.0f} | {a_n:.2f} | {a_a:.2f} | {a_f:.2f}")
# ③ '확실 정상' val 23개만으로도 이동하나
SURE_N = [9, 32, 33, 2, 22, 6, 38, 40, 8, 23, 34, 29, 28, 11, 26, 30, 15, 19, 14, 5, 3, 7, 20]
nvn = [int(v[-3:]) for v in ns.nv]; sn = np.array([v in SURE_N for v in nvn])
p("\n## ③ 확실 정상 val 23개 vs train 정상 (AE 열)")
for j in A_idx:
    p(f"{cols[j]}: 확실정상 val 중앙 백분위 {(Zt[yk == 0, j] < np.median(Zv[sn, j])).mean() * 100:.0f} · AUC {roc_auc_score(np.r_[np.zeros((yk == 0).sum()), np.ones(sn.sum())], np.r_[Zt[yk == 0, j], Zv[sn, j]]):.2f}")
adv(Zt[yk == 0], Zv[sn], "확실정상 val 23 vs train정상 | H+AE")
adv(Zt[yk == 0][:, H_idx], Zv[sn][:, H_idx], "확실정상 val 23 vs train정상 | H만")
# ④ V021 원신호
k = [i for i, v in enumerate(ns.nv) if v.endswith("021")][0]; x = ns.Xv[k]
p("\n## ④ V021 원신호")
p(f"평균 {x.mean():.3f} g · sd {x.std():.4f} · 최대|x| {np.abs(x - x.mean()).max():.3f} at {np.argmax(np.abs(x - x.mean())) / 25000:.4f} s")
sos = butter(4, 200, btype="highpass", fs=25000, output="sos"); h = sosfiltfilt(sos, x - x.mean())
for a, b in ((0, .01), (.01, .02), (.02, .03), (.03, .04), (.04, .06), (.06, .1), (.1, 2)):
    s = h[int(a * 25000):int(b * 25000)]; p(f"  {a:.2f}–{b:.2f} s: 최대 {np.abs(s).max():.4f} g · rms {s.std():.4f}")
raw = x - np.median(x); p("raw 첫 0.05 s 최대", np.abs(raw[:1250]).max().round(4), "나머지 최대", np.abs(raw[1250:]).max().round(4))
# 비교: 다른 파일 중 충격이 0.05 s 안에만 있는 경우
f, t, S = spectrogram(x - x.mean(), fs=25000, nperseg=512, noverlap=256)
Sm = 10 * np.log10(S.mean(1) + 1e-20)
for lo, hi in ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 9500), (9500, 10500), (10500, 12500)):
    p(f"  {lo}-{hi} Hz 평균 dB {Sm[(f >= lo) & (f < hi)].mean():.1f}")
top = np.argsort(-Sm[f > 1000])[:6]; p("  1 kHz 위 봉우리 상위", [int(v) for v in f[f > 1000][top]])
# 9.5–10.5 kHz 지속 여부: 프레임별 최대 주파수
band = (f >= 9000) & (f < 11000); fr = f[band][np.argmax(S[band], 0)]
p(f"  9–11 kHz 프레임 최대 주파수 중앙 {np.median(fr):.0f} Hz · IQR {np.percentile(fr, 25):.0f}–{np.percentile(fr, 75):.0f}")
# 비교 파일들 동일 통계
for nm in ("nor_065", "nor_116", "nor_055", "nor_039", "abn_020"):
    i = list(names).index(nm); xx = ns.X[i]; f2, t2, S2 = spectrogram(xx - xx.mean(), fs=25000, nperseg=512, noverlap=256)
    fr2 = f2[band][np.argmax(S2[band], 0)]; Sm2 = 10 * np.log10(S2.mean(1) + 1e-20)
    p(f"  {nm}: 9–11k 프레임 최대 주파수 중앙 {np.median(fr2):.0f} IQR {np.percentile(fr2, 25):.0f}–{np.percentile(fr2, 75):.0f} · 9.5–10.5k dB {Sm2[(f2 >= 9500) & (f2 < 10500)].mean():.1f}")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("ok")
