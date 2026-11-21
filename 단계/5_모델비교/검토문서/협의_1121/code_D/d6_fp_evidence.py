"""D 진단 6: 오경보(정상) 중심. 같은 점수대(AE z, ET z 둘 다 높음)에 있는 정상 vs 고장을 라벨로 비교해 '정상 증거'를 찾고,
val 이동 여부와 폴드 안 안정성을 잰다. 라벨 사용(탐색, 중첩 CV 아님)."""
from common import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
P = log_to("d6_fp_evidence_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva); oofz = core.oof_z_; lr_oof = core.lr_.decision_function(oofz)
rule_tr = (Dt.rule83 | Dt.knock).values; rule_va = (Dv.rule83 | Dv.knock).values
F = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0); G = pd.read_csv(os.path.join(HERE, "D_repeat_feats.csv"), index_col=0)
H = F.join(G)
# 파일 평균(DC) 과 게인
dc = pd.Series(np.r_[X.mean(1), Xv.mean(1)], index=names + nv, name="dc"); H = H.join(dc)
Htr, Hva = H.loc[names], H.loc[nv]
# 1. 높은 점수대: A의 결합 점수(규칙 없음)가 임계 근처 이상인데 규칙은 꺼진 파일
zone = (lr_oof >= THR - 1.0) & ~rule_tr
P(f"== 1. 높은 점수대 (OOF lr >= {THR-1.0:.1f}, 규칙 꺼짐): {int(zone.sum())}개, 고장 {int((zone&(y==1)).sum())} 정상 {int((zone&nor).sum())}")
P("고장:", [names[i] for i in np.flatnonzero(zone & (y == 1))]); P("정상:", [names[i] for i in np.flatnonzero(zone & nor)])
for c in ("n_imp", "n_ev", "n_hi", "dom", "n_big", "rattle", "h1", "r21", "body_frac", "lo_frac", "dc"):
    v = Htr[c].values.astype(float); m = zone & ~np.isnan(v)
    if len(set(y[m])) == 2:
        a = roc_auc_score(y[m], v[m]); P(f"  점수대 안 AUC {c}: {a:.3f} (n={int(m.sum())}) 정상 중앙 {np.nanmedian(v[m&nor]):.2f} 고장 중앙 {np.nanmedian(v[m&(y==1)]):.2f}")
P("점수대 안 n_ev 분포 (정상 / 고장):")
for k in range(1, 12):
    P(f"  n_ev={k}: 정상 {int((zone&nor&(Htr.n_ev.values==k)).sum())} 고장 {int((zone&(y==1)&(Htr.n_ev.values==k)).sum())}")
P("  n_ev>=12: 정상 %d 고장 %d" % ((zone&nor&(Htr.n_ev.values>=12)).sum(), (zone&(y==1)&(Htr.n_ev.values>=12)).sum()))
# 2. 오경보 파일 정상 증거 요약
FP = ["nor_116", "nor_065", "nor_059", "nor_055", "nor_025", "nor_037", "nor_110", "nor_095", "nor_097", "nor_101", "nor_120"]
MS = ["abn_022", "abn_002", "abn_017", "abn_009"]
def pr(c, v):
    r = Htr[c].values.astype(float)
    return f"{100*np.nanmean(r[nor] <= v):.0f}/{100*np.nanmean(r[y==1] <= v):.0f}"
P("== 2. 오류 파일 정상 증거 (n_ev, n_imp, dom, n_big, dc | 값 (정상 백분위/고장 백분위)) ==")
for f in FP + MS:
    r = Htr.loc[f]; P(f"{f}: n_imp {r.n_imp:.0f}({pr('n_imp',r.n_imp)}) n_ev {r.n_ev:.0f}({pr('n_ev',r.n_ev)}) dom {r.dom:.2f}({pr('dom',r.dom)}) n_big {r.n_big:.0f} h1 {r.h1:.0f}({pr('h1',r.h1)}) dc {r.dc:.2f}({pr('dc',r.dc)}) AEz {oofz[names.index(f),0]:.2f} ETz {oofz[names.index(f),1]:.2f} lr {lr_oof[names.index(f)]:.2f}")
# 3. DC(평균) 방향: 평균 -1 g 파일
P("== 3. 평균(DC) 분포 ==")
P("정상 dc 중앙 %.3f p5/p95 %.3f/%.3f 최소 %.2f | 고장 중앙 %.3f 최소 %.2f | val 최소 %.2f" % (np.median(Htr.dc[nor]), np.percentile(Htr.dc[nor],5), np.percentile(Htr.dc[nor],95), Htr.dc[nor].min(), np.median(Htr.dc[~nor]), Htr.dc[~nor].min(), Hva.dc.min()))
P("dc < -0.5: 정상", list(np.array(names)[nor & (Htr.dc.values < -0.5)]), "고장", list(np.array(names)[(y==1) & (Htr.dc.values < -0.5)]), "val", list(np.array(nv)[Hva.dc.values < -0.5]))
# 4. val 이동: 계수형 증거 피처가 val에서 정상과 같은 분포인가 (정상 분포 대비 val 비율)
P("== 4. 증거 피처의 val 이동 (정상 120 vs val 40 같은 비율이면 이동 없음; 고장 섞임 최대 20-25% 고려) ==")
for lab, f in [("n_ev<=1", lambda d: d.n_ev <= 1), ("n_ev<=2", lambda d: d.n_ev <= 2), ("n_imp<=2", lambda d: d.n_imp <= 2), ("dom>=0.9", lambda d: d.dom >= 0.9), ("n_big<=1", lambda d: d.n_big <= 1),
               ("n_hi>=1", lambda d: d.n_hi >= 1), ("body_frac>=0.9", lambda d: d.body_frac >= 0.9), ("lo_frac>=0.5", lambda d: d.lo_frac >= 0.5)]:
    a = f(Htr)[nor].mean(); b = f(Hva).mean(); c_ = f(Htr)[~nor].mean(); P(f"  {lab}: 정상 {a:.2f} 고장 {c_:.2f} val {b:.2f}")
# 5. 폴드 안 재선택 안정성: '정상 전용 컷' 두 가지(n_ev<=c, dom>=c) 의 고른 값과, 켜진 정상 중 A의 오경보 / 켜진 고장
def stab(name, key, le=True, grid=range(0, 8)):
    cs = []; hit = {}; offn = {}
    v = Htr[key].values.astype(float)
    for s in range(20):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
            fa = v[a][y[a] == 1]; c = min(fa) - 1 if le else None; cs.append(c)
            for i in b[v[b] <= c]: (hit if y[i] == 1 else offn)[names[i]] = (hit if y[i] == 1 else offn).get(names[i], 0) + 1
    P(f"  {name}: 컷 분포 {dict(zip(*[x.tolist() for x in np.unique(cs, return_counts=True)]))} | 테스트에서 켜진 고장 {hit} | 켜진 정상 종류 {len(offn)}, 그중 A오경보: " + str({k: v_ for k, v_ in offn.items() if k in FP}))
P("== 5. 폴드 안 재선택 (컷 = 학습 고장 최소값 - 1) ==")
stab("n_ev<=c", "n_ev"); stab("n_imp<=c", "n_imp")
# 6. 정상 5-NN 순도 (z 공간) : 오경보 목록이 정상 무리 안에 있나
def zspace(M):
    Mz = (M - np.nanmedian(M[nor], 0)) / np.maximum(np.nanpercentile(M[nor], 90, 0) - np.nanmedian(M[nor], 0), 1e-6); return np.nan_to_num(np.clip(Mz, -6, 6))
def pur(M, lst):
    Mz = zspace(M); idx = NearestNeighbors(n_neighbors=6).fit(Mz).kneighbors(Mz, return_distance=False)[:, 1:]
    ii = [names.index(f) for f in lst]; return float(1 - y[idx[ii]].mean()), float(y[idx[y == 1]].mean())
FPtr = ["nor_116", "nor_065", "nor_059", "nor_055", "nor_025", "nor_037", "nor_110", "nor_095"]
P("== 6. 5-NN 순도 (z공간: AE z, ET z + 축). 오경보 정상 8개의 정상 순도 / 전체 고장 순도 ==")
ax = {"n_ev": np.log1p(Htr[["n_ev"]].values), "n_imp": np.log1p(Htr[["n_imp"]].values), "dom": Htr[["dom"]].values, "few": (Htr[["n_ev"]].values <= 1).astype(float),
      "n_big": np.log1p(Htr[["n_big"]].values), "h1": np.log1p(Htr[["h1"]].values), "rattle": np.log1p(Htr[["rattle"]].values)}
for lab, ks in [("기준", []), ("+n_ev", ["n_ev"]), ("+n_imp", ["n_imp"]), ("+dom", ["dom"]), ("+few", ["few"]), ("+n_ev,dom", ["n_ev", "dom"]), ("+n_imp,dom", ["n_imp", "dom"]), ("+n_ev,rattle", ["n_ev", "rattle"]), ("+n_ev,h1", ["n_ev", "h1"])]:
    M = np.hstack([oofz] + [ax[k] for k in ks]); a, b = pur(M, FPtr); P(f"  {lab}: 오경보 정상순도 {a:.2f} 고장순도 {b:.3f}")
# 7. 같은 점수대에서 정상 증거 가중: 로지스틱(AE z, ET z, 증거 축) 시드 5×5겹 계수 안정성
from sklearn.linear_model import LogisticRegression
P("== 7. 로지스틱 계수 안정성 (5겹 x 20 seed 학습 세트, 정상 기준 z) ==")
def coefs(ks):
    M = np.hstack([oofz] + [ax[k] for k in ks]); out = []
    for s in range(20):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
            Mz = M.copy(); nm = a[y[a] == 0]; med = np.median(M[nm, 2:], 0); sp = np.maximum(np.percentile(M[nm, 2:], 90, 0) - med, 1e-6); Mz[:, 2:] = (M[:, 2:] - med) / sp
            out.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Mz[a], y[a]).coef_[0])
    return np.array(out)
for ks in (["n_ev"], ["n_imp"], ["dom"], ["n_ev", "dom"], ["n_imp", "dom"]):
    c = coefs(ks); P(f"  {'+'.join(ks)}: 계수 평균 {np.round(c.mean(0),2)} sd {np.round(c.std(0),2)} 부호 일치율(추가 축) {[round(float((np.sign(c[:,j])==np.sign(c[:,j].mean())).mean()),2) for j in range(2,c.shape[1])]}")
P("완료")
