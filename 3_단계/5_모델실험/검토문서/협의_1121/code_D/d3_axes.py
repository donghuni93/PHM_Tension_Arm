"""D 제안 검증: 라벨을 보고(보고용 아님, 사용 허용) 고장/정상이 어디서 갈리는지 보고, 폴드 안 재선택 안정성·결합 점수 이동·5-NN 순도를 잰다.
새 충격 정의(src/impacts.py, 캐시 impacts.pkl) 기준 파일 피처 → 정상 기준 z → A의 (AE z, ET z) 평면에 축으로 추가.
탐색(quick) 수준. 중첩 CV 아님."""
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from sklearn.neighbors import NearestNeighbors
P = log_to("d3_axes_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva)
oofz = core.oof_z_; lr_oof = core.lr_.decision_function(oofz)
NA, IMP = pickle.load(open(os.path.join(CACHE, "impacts.pkl"), "rb"))
rule_tr = (Dt.rule83 | Dt.knock).values; rule_va = (Dv.rule83 | Dv.knock).values

def fe(im):
    t, f, z, ft = im["t"], im["f"], im["z"], im["fit"]; k = len(t)
    if k == 0: return dict(n_imp=0, n_ev=0, n_hi=0, n_hi_ev=0, res_fit=0., res_cnt=0, body_frac=np.nan, lo_frac=np.nan, hi_frac=0.)
    ev = np.r_[0, np.cumsum(np.diff(t) > 0.05)]
    hi = f >= 7960
    res = (f >= 8000) & (f < 8700) & (z <= 0.02)
    return dict(n_imp=k, n_ev=int(ev.max() + 1), n_hi=int(hi.sum()), n_hi_ev=len(set(ev[hi])), res_fit=float(ft[res].max()) if res.any() else 0.,
                res_cnt=int((res & (ft >= 0.7)).sum()), body_frac=float(((f >= 5410) & (f < 7960)).mean()), lo_frac=float((f < 3510).mean()), hi_frac=float(hi.mean()))
F = pd.DataFrame([fe(im) for im in IMP], index=NA)
F.to_csv(os.path.join(HERE, "D_file_feats.csv"), encoding="utf-8-sig")
Ftr, Fva = F.loc[names], F.loc[nv]
P("== 1. 파일 피처 분포 (새 충격 정의, 라벨 사용) ==")
for c in ("n_imp", "n_ev", "n_hi", "n_hi_ev", "res_fit", "res_cnt", "body_frac", "lo_frac"):
    v = Ftr[c].values.astype(float); m = ~np.isnan(v)
    auc = roc_auc_score(y[m], v[m])
    P(f"{c}: AUC {auc:.3f} | 정상 중앙 {np.nanmedian(v[nor]):.2f} p5/p95 {np.nanpercentile(v[nor],5):.2f}/{np.nanpercentile(v[nor],95):.2f} | 고장 중앙 {np.nanmedian(v[y==1]):.2f} min/p10 {np.nanmin(v[y==1]):.2f}/{np.nanpercentile(v[y==1],10):.2f} | val 중앙 {np.nanmedian(Fva[c].values.astype(float)):.2f}")
P("n_imp<=1: 정상 %d 고장 %d val %d | n_ev<=1: 정상 %d 고장 %d val %d | n_imp<=2: 정상 %d 고장 %d val %d" % (
    ((Ftr.n_imp <= 1) & nor).sum(), ((Ftr.n_imp <= 1) & ~nor).sum(), (Fva.n_imp <= 1).sum(), ((Ftr.n_ev <= 1) & nor).sum(), ((Ftr.n_ev <= 1) & ~nor).sum(), (Fva.n_ev <= 1).sum(),
    ((Ftr.n_imp <= 2) & nor).sum(), ((Ftr.n_imp <= 2) & ~nor).sum(), (Fva.n_imp <= 2).sum()))
P("n_hi>=1: 정상 %d(%s) 고장 %d val %d | res_cnt>=1: 정상 %d(%s) 고장 %d(%s) val %d" % (
    ((Ftr.n_hi >= 1) & nor).sum(), [names[i] for i in np.flatnonzero((Ftr.n_hi >= 1).values & nor)], ((Ftr.n_hi >= 1) & ~nor).sum(), (Fva.n_hi >= 1).sum(),
    ((Ftr.res_cnt >= 1) & nor).sum(), [names[i] for i in np.flatnonzero((Ftr.res_cnt >= 1).values & nor)], ((Ftr.res_cnt >= 1) & ~nor).sum(), [names[i] for i in np.flatnonzero((Ftr.res_cnt >= 1).values & ~nor)], (Fva.res_cnt >= 1).sum()))
P("고장 중 n_hi>=3: %d / 정상 n_hi>=3: %d" % (((Ftr.n_hi >= 3) & ~nor).sum(), ((Ftr.n_hi >= 3) & nor).sum()))
KEY = ["abn_022", "abn_002", "abn_017", "abn_009", "nor_116", "nor_065", "nor_059", "nor_037", "nor_110", "nor_095", "nor_055", "nor_097", "nor_101", "nor_120"]
P("== 2. 핵심 파일 (오류 14) 새 정의 피처 ==")
P(Ftr.loc[KEY].round(2).to_string())

# ---- 경계 구간: OOF lr이 임계값 ±1.5 안 (규칙 꺼짐)
zone = (np.abs(lr_oof - THR) < 1.5) & ~rule_tr
P(f"== 3. 경계 구간 train OOF |lr-1|<1.5 규칙 꺼짐: {int(zone.sum())}개 (고장 {int((zone&(y==1)).sum())}, 정상 {int((zone&nor).sum())}) ==")
P("고장:", [names[i] for i in np.flatnonzero(zone & (y == 1))], "정상:", [names[i] for i in np.flatnonzero(zone & nor)])
for c in ("n_imp", "n_ev", "n_hi", "n_hi_ev", "res_fit", "body_frac", "lo_frac"):
    v = Ftr[c].values.astype(float); m = zone & ~np.isnan(v)
    if len(set(y[m])) == 2: P(f"  구간 안 AUC {c}: {roc_auc_score(y[m], v[m]):.3f} (n={int(m.sum())})")
# 구간 밖 고장 포함 전체 "미탐·오경보" 파일
ER = [names[i] for i in range(n) if (y[i] == 1 and lr_oof[i] < THR and not rule_tr[i]) or (y[i] == 0 and lr_oof[i] >= THR)]
P("OOF에서 lr만으로 틀린 파일(규칙 꺼짐):", ER)
P(Ftr.loc[ER].round(2).to_string())

# ---- 폴드 안 재선택 안정성
P("== 4. 폴드 안 재선택 안정성 (5겹 x 20 seed = 100 학습 세트, 라벨로 같은 방법으로 컷을 다시 고름) ==")
def stab(name, fn):
    out = []
    for s in range(20):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y): out.append(fn(a))
    out = np.array(out, float); vals, cnt = np.unique(out[~np.isnan(out)], return_counts=True)
    P(f"  {name}: 고른 컷 분포 " + ", ".join(f"{v:g}:{c}" for v, c in zip(vals, cnt)) + f" (최빈 비율 {cnt.max()/len(out):.2f})")
    return out
nimp = Ftr.n_imp.values; nev = Ftr.n_ev.values; nhi = Ftr.n_hi.values; hif = Ftr.hi_frac.values
# (a) 정상 전용 구간: 학습 고장 min(n_imp) - 1  → "n_imp <= c" 면 정상 쪽 (고장 0)
stab("정상전용 n_imp<=c (c = 학습 고장 최소 n_imp - 1)", lambda a: nimp[a][y[a] == 1].min() - 1)
stab("정상전용 n_ev<=c (c = 학습 고장 최소 n_ev - 1)", lambda a: nev[a][y[a] == 1].min() - 1)
# (b) 고주파 울림: n_hi >= c 인 정상이 학습 정상의 2% 이하인 최소 c
def c_hi(a):
    for c in range(1, 12):
        if (nhi[a][y[a] == 0] >= c).mean() <= 0.02: return c
    return np.nan
stab("고주파 n_hi>=c (c = 학습 정상 2% 이하가 되는 최소 c)", c_hi)
# 그 컷에서 학습 고장 몫
def cov(a):
    c = c_hi(a); return (nhi[a][y[a] == 1] >= c).mean() if not np.isnan(c) else np.nan
o = np.array([cov(a) for s in range(5) for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y)])
P(f"  그 컷이 켜는 학습 고장 비율: 평균 {np.nanmean(o):.2f} 최소 {np.nanmin(o):.2f} 최대 {np.nanmax(o):.2f}")
# (c) 폴드 밖 검증: 학습에서 고른 컷을 테스트 폴드에 적용 → 켜진 정상/고장 (정상전용 컷)
tp = fp = 0; fp_names = {}
for s in range(20):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
        c = nimp[a][y[a] == 1].min() - 1; on = b[nimp[b] <= c]
        for i in on:
            if y[i] == 1: tp += 1; fp_names[names[i]] = fp_names.get(names[i], 0) + 1
print(); P(f"  정상전용 n_imp 컷을 테스트 폴드에 적용: 켜진 고장 {tp}회(100번 중) {fp_names}")

# ---- lr에 축 추가: 정상 기준 z (폴드 안 정상 중앙·90%-중앙 폭), CV로 OOF lr
def zscore_cols(M, tr_idx, ytr):
    nm = tr_idx[ytr == 0]; med = np.nanmedian(M[nm], 0); sp = np.nanpercentile(M[nm], 90, 0) - med; sp = np.where(sp > 1e-9, sp, 1.0)
    return med, sp
def build(name):
    ft = lambda c: Ftr[c].values.astype(float)
    if name == "n_imp": return np.c_[np.log1p(ft("n_imp"))]
    if name == "few": return np.c_[(ft("n_imp") <= 1).astype(float)]
    if name == "n_hi": return np.c_[np.log1p(ft("n_hi"))]
    if name == "n_hi_ev": return np.c_[np.log1p(ft("n_hi_ev"))]
    if name == "res": return np.c_[ft("res_fit")]
    if name == "body": return np.c_[np.nan_to_num(ft("body_frac"), nan=0.9)]
    if name == "ev": return np.c_[np.log1p(ft("n_ev"))]
    raise KeyError(name)
def cv_lr(extra_names, seeds=range(10), flip=()):
    base = oofz.copy()
    E = [build(k) for k in extra_names]; M = np.hstack([base] + E) if E else base
    res = []; L = np.zeros((len(seeds), n))
    for si, s in enumerate(seeds):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
            Mz = M.copy()
            if E:
                med, sp = zscore_cols(M[:, 2:], a, y[a]); Mz[:, 2:] = (M[:, 2:] - med) / sp
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Mz[a], y[a]); L[si, b] = lr.decision_function(Mz[b])
    return L
def summarize(tag, L):
    roc = np.mean([roc_auc_score(y, l) for l in L]); pr = np.mean([average_precision_score(y, l) for l in L])
    # 규칙 켜진 고장도 포함한 '규칙 없는' 점수. 임계값은 시드별 풀링 OOF F2 최대(탐색용, 낙관)
    f2s = []; thrs = []
    for l in L:
        ts = np.unique(l); fs = [fbeta_score(y, (l >= t).astype(int), beta=2) for t in ts]; k = int(np.argmax(fs)); f2s.append(fs[k]); thrs.append(ts[k])
    lm = L.mean(0)
    P(f"{tag}: ROC {roc:.3f} PR-AUC {pr:.3f} F2@최적임계(낙관) {np.mean(f2s):.3f}±{np.std(f2s):.3f} 임계 {np.mean(thrs):.2f}±{np.std(thrs):.2f}")
    return lm, np.mean(thrs)
P("== 5. 결합 점수에 축 추가 (규칙 없음; A-규칙 = AE z, ET z 로지스틱) ==")
Lb = cv_lr([]); lb, tb = summarize("기준 AE+ET", Lb)
CONF = {"+n_imp": ["n_imp"], "+few(n<=1)": ["few"], "+n_hi": ["n_hi"], "+res": ["res"], "+body": ["body"], "+n_ev": ["ev"], "+res+n_imp": ["res", "n_imp"], "+res+n_hi+n_imp": ["res", "n_hi", "n_imp"], "+res+n_hi+n_imp+few": ["res", "n_hi", "n_imp", "few"]}
LM = {"기준": (lb, tb)}
for k, v in CONF.items(): LM[k] = summarize(k, cv_lr(v))
P("== 6. 핵심 파일의 결합 점수 (정상 사이 백분위: 고장은 높을수록 좋음 / 정상은 낮을수록 좋음), 구성별 ==")
tab = []
for f in KEY + ["abn_003", "abn_006", "abn_010", "abn_029", "abn_014", "nor_025", "nor_106", "nor_019", "nor_092"]:
    i = names.index(f); row = dict(file=f)
    for k, (lm, t) in LM.items(): row[k] = f"{100*(lm[nor] <= lm[i]).mean():.0f}"
    tab.append(row)
P(pd.DataFrame(tab).to_string(index=False))
P("정상 OOF 평균 lr의 경계 가까움: |lr - 임계|<0.5 인 파일 수 (train 150) / 임계는 구성별 최적")
for k, (lm, t) in LM.items(): P(f"  {k}: {int((np.abs(lm - t) < 0.5).sum())}  (정상 오경보 {int(((lm >= t) & nor).sum())}, 고장 미탐 {int(((lm < t) & (y==1)).sum())})")

# ---- 5-NN 순도 (결합 평면 + 축), 라벨 보고용
P("== 7. 고장 5-NN 순도 (z 공간: AE z, ET z + 축; 정상 기준 robust z; 무작위 기대 = 0.20) ==")
def purity(M):
    Mz = (M - np.nanmedian(M[nor], 0)) / np.maximum(np.nanpercentile(M[nor], 90, 0) - np.nanmedian(M[nor], 0), 1e-6)
    Mz = np.nan_to_num(np.clip(Mz, -6, 6)); nn = NearestNeighbors(n_neighbors=6).fit(Mz); idx = nn.kneighbors(Mz, return_distance=False)[:, 1:]
    return float(y[idx[y == 1]].mean()), float(1 - y[idx[nor]].mean())
for k, v in [("기준(AE z, ET z)", []), *[(kk, vv) for kk, vv in CONF.items()]]:
    M = np.hstack([oofz] + [build(q) for q in v]) if v else oofz
    pf, pn = purity(M); P(f"  {k}: 고장 순도 {pf:.3f} 정상 순도 {pn:.3f}")

# ---- val: 최종 모델(train 150 전체)로 새 축 적용했을 때 val 경계 밀집
P("== 8. val 경계 밀집 (train 전체로 로지스틱 학습, 임계 = train OOF 최적, 규칙 없음) ==")
def fv(c): return Fva[c].values.astype(float)
def build_v(name):
    return {"n_imp": np.c_[np.log1p(fv("n_imp"))], "few": np.c_[(fv("n_imp") <= 1).astype(float)], "n_hi": np.c_[np.log1p(fv("n_hi"))], "res": np.c_[fv("res_fit")], "body": np.c_[np.nan_to_num(fv("body_frac"), nan=0.9)], "ev": np.c_[np.log1p(fv("n_ev"))]}[name]
zv = np.c_[Dv.ae_z.values, Dv.et_z.values]
allidx = np.arange(n)
for k, v in [("기준", [])] + list(CONF.items()):
    M = np.hstack([oofz] + [build(q) for q in v]) if v else oofz; Mv = np.hstack([zv] + [build_v(q) for q in v]) if v else zv
    if v:
        med, sp = zscore_cols(M[:, 2:], allidx, y); M = M.copy(); Mv = Mv.copy(); M[:, 2:] = (M[:, 2:] - med) / sp; Mv[:, 2:] = (Mv[:, 2:] - med) / sp
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(M, y); l_tr = lr.decision_function(M); l_v = lr.decision_function(Mv)
    ts = np.unique(l_tr); fs = [fbeta_score(y, (l_tr >= t).astype(int), beta=2) for t in ts]; t0 = ts[int(np.argmax(fs))]
    sd = l_tr[nor].std()
    P(f"  {k}: 임계 {t0:.2f} val 판정 {int((l_v>=t0).sum())} |val-임계|<0.5 {int((np.abs(l_v-t0)<0.5).sum())} <1.0 {int((np.abs(l_v-t0)<1.0).sum())} (정상 OOF-in-sample lr sd {sd:.2f}) 계수 {np.round(lr.coef_[0],2)}")
    LM[k + "_val"] = l_v
pd.DataFrame({"file": nv, **{k: LM[k + "_val"] for k in ["기준"] + list(CONF) if k + "_val" in LM}}).round(2).to_csv(os.path.join(HERE, "val_axes_scores.csv"), index=False, encoding="utf-8-sig")
P("완료")
