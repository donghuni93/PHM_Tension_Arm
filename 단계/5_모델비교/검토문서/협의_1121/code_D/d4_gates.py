"""D 제안 검증 2: (1) 컷의 폴드 안 재선택 안정성(정상 99% 분위·정상 전용 구간) (2) val 분포 이동에 대한 판정 민감도 (3) 비선형 결합(ET 스택)에 축을 주었을 때. 라벨 사용 허용(탐색), 중첩 CV 아님."""
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features2 import cached_feat2
P = log_to("d4_gates_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva); oofz = core.oof_z_; lr_oof = core.lr_.decision_function(oofz)
rule_tr = (Dt.rule83 | Dt.knock).values; rule_va = (Dv.rule83 | Dv.knock).values
F = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0); Ftr, Fva = F.loc[names], F.loc[nv]
F2tr, Ltr, fr = cached_feat2("train", X); F2va, Lva, _ = cached_feat2("val", Xv)
F2tr.index = names; F2va.index = nv

P("== 1. 컷 폴드 안 재선택 안정성: '학습 폴드 정상 99% 분위' 규칙 (5겹 x 20 seed), 테스트 폴드에 적용 ==")
def q99_rule(vals_tr, vals_te, ytr, mask_tr=None):
    nm = vals_tr[ytr == 0]; nm = nm[~np.isnan(nm)]; c = np.percentile(nm, 99); return c, vals_te > c
def run(name, v, extra=None):
    on = {i: 0 for i in range(n)}; cuts = []
    for s in range(20):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
            c, flag = q99_rule(v[a], v[b], y[a]); cuts.append(c)
            if extra is not None: flag = flag & extra[b]
            for i in b[flag]: on[i] += 1
    cuts = np.array(cuts); fl = [(names[i], k) for i, k in on.items() if k > 0 and y[i] == 1]; nl = [(names[i], k) for i, k in on.items() if k > 0 and y[i] == 0]
    P(f"  {name}: 컷 중앙 {np.median(cuts):.3f} (5-95% {np.percentile(cuts,5):.3f}~{np.percentile(cuts,95):.3f}, 변동계수 {cuts.std()/abs(cuts.mean()):.2f}) | 20회 중 켜진 고장 {sorted(fl, key=lambda t:-t[1])} | 켜진 정상 {sorted(nl, key=lambda t:-t[1])}")
v_S = F2tr["L1_S"].values.astype(float); run("L1_S 주기 잠금 > 정상 99%", v_S)
v_h = F2tr["H18_hfl"].values.astype(float); run("H18_hfl 하중 대비 고주파 > 정상 99% & 충격>=1", v_h, extra=(Ftr.n_imp.values >= 1))
v_k = Dt.knock.values.astype(float) if hasattr(Dt, "knock") else None
v_rf = Ftr.res_fit.values.astype(float); P("  res_fit(8.0-8.7k, ζ<=0.02) 0 아닌 파일: 정상", [(names[i], round(v_rf[i], 2)) for i in np.flatnonzero(nor & (v_rf > 0))], "고장", [(names[i], round(v_rf[i], 2)) for i in np.flatnonzero((y == 1) & (v_rf > 0))])
P("  → 정상 최대 res_fit 없음(0): '깨끗한 공진 충격 있음' 은 컷 없이 정의됨 (fit>0, 사전 사전 문턱 없음) vs 옛 규칙 fit>=0.7")
for c in (0.0, 0.5, 0.6, 0.7):
    P(f"  res_fit > {c}: 정상 {int((nor&(v_rf>c)).sum())} 고장 {int(((y==1)&(v_rf>c)).sum())} {[names[i] for i in np.flatnonzero((y==1)&(v_rf>c))]}")
# 단일 사건 내림 게이트 n_ev <= c: 학습 고장 최소 n_ev - 1
nev = Ftr.n_ev.values; hits = {}; cs = []
for s in range(20):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
        c = nev[a][y[a] == 1].min() - 1; cs.append(c)
        for i in b[nev[b] <= c]: hits[i] = hits.get(i, 0) + 1
P(f"  n_ev<=c 게이트(c=학습 고장 최소-1): c 분포 {dict(zip(*np.unique(cs, return_counts=True)))} | 테스트 폴드에서 켜진 고장 {[(names[i],k) for i,k in hits.items() if y[i]==1]} 켜진 정상 {sum(1 for i in hits if y[i]==0)}종")
fev = sorted(Ftr.n_ev[y == 1].values)[:6]; P("  고장 n_ev 가장 작은 6개:", fev, [names[i] for i in np.argsort(np.where(y == 1, nev, 99))[:6]])
P("  n_ev<=1 정상 파일:", [names[i] for i in np.flatnonzero(nor & (nev <= 1))])
P("  OOF 틀림 파일 중 n_ev<=1:", [(names[i], int(nev[i])) for i in np.flatnonzero((nev <= 1) & (lr_oof >= THR) & nor)], "/ lr만 틀린 정상 오경보 전체:", [(names[i], int(nev[i])) for i in np.flatnonzero((lr_oof >= THR) & nor)])
P("  val에서 n_ev<=1:", [nv[i] for i in np.flatnonzero(Fva.n_ev.values <= 1)])

P("== 2. val 판정의 AE 이동 민감도 (A(b) 규칙 포함, 임계 1.0): AE z에서 s를 뺌 ==")
lrw = core.lr_.coef_[0]; b0 = core.lr_.intercept_[0]
for s in (0, 0.25, 0.5, 0.75, 1.0, 1.25):
    l = lrw[0] * (Dv.ae_z.values - s) + lrw[1] * Dv.et_z.values + b0
    p = (l >= THR) | rule_va
    P(f"  s={s:.2f}: 판정 {int(p.sum())} |lr-1|<0.5 {int(((np.abs(l-THR)<0.5)&~rule_va).sum())} 경계 {int(((np.abs(l-THR)<1.0)&~rule_va).sum())}")
P("  ET z에서 t를 빼면(AE 그대로):", [(t, int((((lrw[0]*Dv.ae_z.values + lrw[1]*(Dv.et_z.values - t) + b0) >= THR) | rule_va).sum())) for t in (0, 0.25, 0.5, 0.75)])
P("  결합 점수 기여(val 중앙): AE 항 %.2f ET 항 %.2f 절편 %.2f / train 정상 중앙: AE 항 %.2f ET 항 %.2f" % (lrw[0]*np.median(Dv.ae_z), lrw[1]*np.median(Dv.et_z), b0, lrw[0]*np.median(oofz[nor,0]), lrw[1]*np.median(oofz[nor,1])))
P(f"  임계 1.0 이동 환산: AE z 0.1 = lr {lrw[0]*0.1:.2f} / ET z 0.1 = {lrw[1]*0.1:.2f} ; 경계 폭 ±0.5는 AE z ±{0.5/lrw[0]:.2f}")

P("== 3. val 경계 파일(규칙 꺼짐, |lr-1|<1)의 새 정의 피처 ==")
bd = [i for i in range(40) if (not rule_va[i]) and abs(Dv.lr.values[i] - THR) < 1.0]
T3 = Fva.iloc[bd][["n_imp", "n_ev", "n_hi", "body_frac", "lo_frac"]].copy(); T3["lr"] = Dv.lr.values[bd].round(2); T3["ae_z"] = Dv.ae_z.values[bd].round(2); T3["et_z"] = Dv.et_z.values[bd].round(2)
T3["L1_S"] = F2va["L1_S"].values[bd].round(2)
try:
    ax = pd.read_csv(os.path.join(HERE, "val_axes_scores.csv"), index_col=0); T3["lr+res+n_imp"] = ax.loc[T3.index, "+res+n_imp"].values; T3["lr+n_ev"] = ax.loc[T3.index, "+n_ev"].values; T3["기준(재학습)"] = ax.loc[T3.index, "기준"].values
except Exception as e: P("axes 읽기 실패", e)
P(T3.to_string())
P("  val 경계 14 중 n_ev<=2:", int((T3.n_ev <= 2).sum()), " n_imp<=3:", int((T3.n_imp <= 3).sum()), " 정상 train n_ev<=2 비율 %.2f, 고장 %.2f" % ((Ftr.n_ev[nor] <= 2).mean(), (Ftr.n_ev[y == 1] <= 2).mean()))

P("== 4. 비선형 결합: ET 스택 [AE z, ET z, 축...] vs 로지스틱 (5겹 x 5 seed, 규칙 없음) ==")
ft = lambda c: Ftr[c].values.astype(float)
AX = {"기준": np.empty((n, 0)), "+n_ev,n_hi,res": np.c_[np.log1p(ft("n_ev")), np.log1p(ft("n_hi")), ft("res_fit")], "+n_imp,res": np.c_[np.log1p(ft("n_imp")), ft("res_fit")],
      "+S,n_ev,n_hi,res": np.c_[F2tr["L1_S"].values, np.log1p(ft("n_ev")), np.log1p(ft("n_hi")), ft("res_fit")]}
KEY = ["abn_022", "abn_002", "abn_017", "abn_009", "nor_116", "nor_065", "nor_059", "nor_055", "nor_025", "nor_037"]
ROW = []
for k, A in AX.items():
    M = np.hstack([oofz, A]); S = np.zeros((5, n))
    for si in range(5):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=si).split(np.arange(n), y):
            m = ExtraTreesClassifier(300, min_samples_leaf=2, max_features=None, class_weight="balanced", random_state=si, n_jobs=1).fit(M[a], y[a]); S[si, b] = m.predict_proba(M[b])[:, 1]
    sm = S.mean(0)
    P(f"  ET스택 {k}: ROC {np.mean([roc_auc_score(y,s) for s in S]):.3f} PR-AUC {np.mean([average_precision_score(y,s) for s in S]):.3f} | 핵심 파일 정상 사이 백분위: " + " ".join(f"{f[-7:]}:{100*(sm[nor]<=sm[names.index(f)]).mean():.0f}" for f in KEY))
P("완료")
