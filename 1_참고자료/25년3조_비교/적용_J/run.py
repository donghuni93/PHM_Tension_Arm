# %% [markdown]
# 25년 데이터에 우리 J(EXP-044)를 적용 (참고용, 정식 단계 밖). 우리 train 150개로 학습한 J를 25년 신호에 그대로 판정시킨다
# - 25년 = 5 kHz × 10초. 5배 업샘플(resample_poly 5/1) → 25 kHz × 10초 → 2초 창 5개로 자름(우리 입력 길이 50,000과 같음)
# - 업샘플은 2.5 kHz 위 정보를 만들지 못한다(위쪽은 비어 있음). 그래서 J의 고주파 피처 · AE · 8.3k 규칙은 25년 신호에서 '빈 대역'을 본다
# - 평가: 25년 train 150(라벨 있음)에 대한 파일 단위 F2 · 재현율 · 오경보율(학습 없음, 점수만 적용). 임계값 · 모델 모두 우리 것 그대로
# 실행: conda run -n qe-phm python "1_참고자료/25년3조_비교/적용_J/run.py"
# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); sys.path.insert(0, os.path.join(_d, "1_참고자료", "25년3조_비교"))
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from src.features2 import cached_feat2
from scipy.signal import resample_poly
from sklearn.metrics import fbeta_score, recall_score
from load25 import load25
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); out = open(os.path.join(HERE, "결과.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n"); out.flush()

ns = NS(); n = ns.n; y = ns.y
X25, y25, n25 = load25(); lab = y25 >= 0; X25, y25, n25 = X25[lab], y25[lab], [n for n, l in zip(n25, lab) if l]   # train 150만 (val은 라벨 없음)
P("25년 train", X25.shape, "정상", int((y25 == 0).sum()), "고장", int((y25 == 1).sum()))

# %% 업샘플 + 2초 창
UP = np.stack([resample_poly(x, 5, 1) for x in X25]); NW = UP.shape[1] // 50000
W = UP[:, :NW * 50000].reshape(len(UP), NW, 50000).reshape(-1, 50000); yw = np.repeat(y25, NW)
P("업샘플 후", UP.shape, "창", W.shape, "파일당 창", NW)

# %% J 학습 (EXP-044 코드 그대로, 우리 train 150 전체)
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float)
idx = np.arange(n, dtype=float); M_rule = np.column_stack([ns.M, idx, r83])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M_rule, y); mm = mJ.m_

# %% 25년 창에 적용
Tw, Lw, _ = cached_feat2("y25up", W)
Mw = np.hstack([Tw[ns.FIXED].values, Tw[["L1_S"]].values, Lw])
sw, rw = rule83_flags(W); r83w = (sw | rw).astype(float)
ae_tr = _fit_ae(LP_TR[ns.y == 0]); _, LPw = segment_logpsd(W); mm.ae_override_ = _ae_score(ae_tr, LPw)
Mw2 = np.column_stack([Mw, np.zeros(len(Mw)), r83w]); f = mJ.fires(Mw2); pw = mJ.decision_function(Mw2) >= .5
pf = pw.reshape(len(y25), NW)
P("\n창 단위 고장 판정 비율: 정상 파일", f"{pw[yw == 0].mean():.3f}", "· 고장 파일", f"{pw[yw == 1].mean():.3f}")
for rule, pred in (("창 1개라도", pf.any(1)), ("창 과반(3/5)", pf.sum(1) >= 3), ("창 5개 모두", pf.all(1))):
    P(f"파일 판정 [{rule}] F2 {fbeta_score(y25, pred, beta=2):.3f} · 재현율 {recall_score(y25, pred):.3f} · 오경보율 {pred[y25 == 0].mean():.3f} · 고장 판정 {int(pred.sum())}/150")

# %% 진단: 어떤 부품이 켜졌나, 점수 분포
for nm, k in (("결합(main)", "main"), ("E2", "E2"), ("8.3k", "R83")):
    v = np.asarray(f[k], bool); P(f"켜짐 {nm}: 정상 창 {v[yw == 0].mean():.3f} · 고장 창 {v[yw == 1].mean():.3f}")
from sklearn.metrics import roc_auc_score
for nm, v in (("ET logit", f["a"]), ("박자 z", f["z"]), ("AE z", f["ae"])):
    v = np.asarray(v, float); fv = v.reshape(len(y25), NW).mean(1)
    P(f"{nm}: 25년 정상 파일 중앙 {np.median(fv[y25 == 0]):.2f} · 고장 중앙 {np.median(fv[y25 == 1]):.2f} · AUC(고장 vs 정상, 25년 안) {roc_auc_score(y25, fv):.3f}")
# 우리 데이터 기준 값(OOF 아님, 전체 학습 점수)과 비교하기 위한 기준
fo = mJ.fires(M_rule)
for nm, k in (("ET logit", "a"), ("AE z", "ae")):
    v = np.asarray(fo[k], float); P(f"[기준] 우리 train {nm}: 정상 중앙 {np.median(v[y == 0]):.2f} · 고장 중앙 {np.median(v[y == 1]):.2f}")
out.close()
