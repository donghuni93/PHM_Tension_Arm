"""A가 val에서 흔들리는 원인 분해 (1): 규칙 의존 · 결합 점수 밀도 · val 분포 이동(AE 게인). 라벨은 보고용."""
from common import *
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, fbeta_score
from src.final import _fit_ae, _ae_score
P = log_to("d2_shift_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva)
oofz = core.oof_z_; lr_oof = core.lr_.decision_function(oofz)
rule_tr = (Dt.rule83 | Dt.knock).values; rule_va = (Dv.rule83 | Dv.knock).values
P("== 1. 규칙 의존 ==")
P(f"train: 규칙이 켜진 고장 {int((rule_tr & (y==1)).sum())}/30, 규칙이 켜진 정상 {int((rule_tr & nor).sum())}/120 (규칙 기준값은 train 전체 정상으로 잡은 것이라 in-sample)")
pred_lr = lr_oof >= THR; pred_all = pred_lr | rule_tr
f2 = lambda p: fbeta_score(y, p.astype(int), beta=2)
P(f"train OOF(4겹 1회, 임계값 1.0): 규칙 없이 F2 {f2(pred_lr):.3f} (재현 {pred_lr[y==1].mean():.2f}, 오경보 {pred_lr[nor].mean():.3f}) → 규칙 포함 F2 {f2(pred_all):.3f} (재현 {pred_all[y==1].mean():.2f}, 오경보 {pred_all[nor].mean():.3f})")
only_rule = rule_tr & ~pred_lr
P("규칙만으로 고장이 되는 train 파일:", [f"{names[i]}[{'고장' if y[i] else '정상'}]" for i in np.flatnonzero(only_rule)])
pv_lr = Dv.lr.values >= THR; pv_all = pv_lr | rule_va
P(f"val: 규칙 켜짐 {int(rule_va.sum())} {list(np.array(nv)[rule_va])} / lr만 판정 {int(pv_lr.sum())} / 규칙 포함 {int(pv_all.sum())} / 규칙이 바꾼 파일 {list(np.array(nv)[pv_all & ~pv_lr])}")
P("== 2. 결합 점수 밀도 (임계값 1.0 주변, 규칙 켜진 파일 제외) ==")
for w in (0.25, 0.5, 1.0):
    mt = (np.abs(lr_oof - THR) < w) & ~rule_tr; mv = (np.abs(Dv.lr.values - THR) < w) & ~rule_va
    P(f"|lr-1|<{w}: train OOF {int(mt.sum())}/{int((~rule_tr).sum())} = {mt.sum()/(~rule_tr).sum():.3f} · val {int(mv.sum())}/{int((~rule_va).sum())} = {mv.sum()/(~rule_va).sum():.3f}")
P("train OOF lr 분포: 정상 p50/p90/p95/p99 = " + "/".join(f"{np.percentile(lr_oof[nor],q):.2f}" for q in (50, 90, 95, 99)) + " · 고장 p10/p25/p50 = " + "/".join(f"{np.percentile(lr_oof[y==1],q):.2f}" for q in (10, 25, 50)))
P("val lr 분위: p10/p25/p50/p75/p90 = " + "/".join(f"{np.percentile(Dv.lr.values,q):.2f}" for q in (10, 25, 50, 75, 90)))
P("train OOF 임계값별 (lr만) 정상 위/고장 아래 파일 수 - 임계값 0.5/1.0/1.5:", [(t, int((lr_oof[nor] >= t).sum()), int((lr_oof[y == 1] < t).sum())) for t in (0.5, 1.0, 1.5)])
P("== 3. val 분포 이동: AE z · ET z ==")
for k, nm in enumerate(("AE", "ET")):
    zt, zv = oofz[:, k], Dv[["ae_z", "et_z"][k]].values
    P(f"{nm} z: train 정상 중앙 {np.median(zt[nor]):.2f} p75 {np.percentile(zt[nor],75):.2f} p95 {np.percentile(zt[nor],95):.2f} | val 중앙 {np.median(zv):.2f} p25 {np.percentile(zv,25):.2f} p75 {np.percentile(zv,75):.2f} | val>정상p95 {int((zv>np.percentile(zt[nor],95)).sum())}/40 | val 하위 25% 중앙 {np.median(np.sort(zv)[:10]):.2f} vs 정상 하위 25% 중앙 {np.median(np.sort(zt[nor])[:30]):.2f}")
# 무작위 표본(고장 20 %)의 val 중앙값 분포 — 이동이 표본 운인가
rng = np.random.default_rng(0); sims = []
for _ in range(2000):
    s = np.r_[rng.choice(np.flatnonzero(nor), 32, replace=False), rng.choice(np.flatnonzero(y == 1), 8, replace=False)]
    sims.append((np.median(oofz[s, 0]), np.median(oofz[s, 1])))
sims = np.array(sims)
P(f"무작위 40개(정상 32 + 고장 8) 중앙값 AE z {np.percentile(sims[:,0],5):.2f}~{np.percentile(sims[:,0],95):.2f}(5-95%) · ET z {np.percentile(sims[:,1],5):.2f}~{np.percentile(sims[:,1],95):.2f} → val AE 중앙 {np.median(Dv.ae_z):.2f}은 표본 운 밖, ET 중앙 {np.median(Dv.et_z):.2f}은 {'안' if np.percentile(sims[:,1],5) <= np.median(Dv.et_z) <= np.percentile(sims[:,1],95) else '밖'}")
P("== 4. AE 이동은 게인인가 (AE 입력 = 절대 log PSD, 파일 게인을 안 뺀다) ==")
gain_tr = Ttr["LP"].mean((1, 2)); gain_va = Tva["LP"].mean((1, 2))
P(f"파일 게인(평균 log10 PSD): 정상 중앙 {np.median(gain_tr[nor]):.2f} sd {gain_tr[nor].std():.2f} · 고장 중앙 {np.median(gain_tr[y==1]):.2f} · val 중앙 {np.median(gain_va):.2f} (정상 대비 {(np.median(gain_va)-np.median(gain_tr[nor]))/gain_tr[nor].std():+.2f} sd)")
rho_n = spearmanr(gain_tr[nor], oofz[nor, 0])[0]; rho_v = spearmanr(gain_va, Dv.ae_z.values)[0]
P(f"AE z와 게인의 Spearman: train 정상 {rho_n:.2f} · val {rho_v:.2f} · train 고장 {spearmanr(gain_tr[y==1], oofz[y==1,0])[0]:.2f}")
A = np.c_[np.ones(nor.sum()), gain_tr[nor] - np.median(gain_tr[nor]), (gain_tr[nor] - np.median(gain_tr[nor])) ** 2]
beta = np.linalg.lstsq(A, oofz[nor, 0], rcond=None)[0]
g = gain_va - np.median(gain_tr[nor]); pred = beta[0] + beta[1] * g + beta[2] * g ** 2
P(f"정상 AE z ~ 게인(2차) 회귀 R2 {1 - ((oofz[nor,0]-A@beta)**2).sum()/((oofz[nor,0]-oofz[nor,0].mean())**2).sum():.2f} → val 예측 중앙 {np.median(pred):.2f} (실제 {np.median(Dv.ae_z):.2f}), 상관 {np.corrcoef(pred, Dv.ae_z.values)[0,1]:.2f}")
P("== 5. 게인을 뺀 AE'(탐색, 새 구성 아님): 구간마다 평균 log PSD를 빼고 같은 AE를 학습 ==")
def prep(LP): return LP - LP.mean(-1, keepdims=True)
LPn, LPv = prep(Ttr["LP"]), prep(Tva["LP"])
raw = np.empty(n)
for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(np.arange(n), y):
    ae = _fit_ae(LPn[a][y[a] == 0]); raw[b] = _ae_score(ae, LPn[b])
ae_all = _fit_ae(LPn[nor]); raw_v = _ae_score(ae_all, LPv)
med = np.median(raw[nor]); d = abs(np.percentile(raw[nor], 90) - med); z2 = (raw - med) / d; zv2 = (raw_v - med) / d
P(f"AE' z: 정상 sd {z2[nor].std():.2f} p95 {np.percentile(z2[nor],95):.2f} · 고장 중앙 {np.median(z2[y==1]):.2f} · val 중앙 {np.median(zv2):.2f} (원래 AE: 정상 sd {oofz[nor,0].std():.2f} · 고장 중앙 {np.median(oofz[y==1,0]):.2f} · val 중앙 {np.median(Dv.ae_z):.2f}) · val>정상p95 {int((zv2>np.percentile(z2[nor],95)).sum())}/40")
P(f"OOF AUC AE {roc_auc_score(y, oofz[:,0]):.3f} → AE' {roc_auc_score(y, z2):.3f} · PR-AUC {average_precision_score(y, oofz[:,0]):.3f} → {average_precision_score(y, z2):.3f}")
for f in ("abn_002", "abn_029", "abn_022", "abn_005", "abn_014", "abn_017"):
    i = names.index(f); P(f"  {f}: AE z {oofz[i,0]:.2f}(정상 중 백분위 {100*(oofz[nor,0]<=oofz[i,0]).mean():.0f}) → AE' z {z2[i]:.2f}({100*(z2[nor]<=z2[i]).mean():.0f})")
from sklearn.linear_model import LogisticRegression
Z2 = np.c_[z2, oofz[:, 1]]; lr2 = LogisticRegression(C=1.0, class_weight="balanced").fit(Z2, y)
l2 = lr2.decision_function(Z2); lv2 = lr2.decision_function(np.c_[zv2, Dv.et_z.values])
P(f"AE' + ET 로지스틱 계수 {lr2.coef_.round(2)}, 절편 {lr2.intercept_.round(2)} (원래 {core.lr_.coef_.round(2)}, {core.lr_.intercept_.round(2)})")
P(f"train OOF 같은 임계값 1.0: F2 {f2(l2>=1.0):.3f} (원래 lr만 {f2(pred_lr):.3f}) 재현 {(l2>=1)[y==1].mean():.2f} 오경보 {(l2>=1)[nor].mean():.3f}")
for w in (0.5, 1.0):
    P(f"val |lr'-1|<{w}: {int(((np.abs(lv2-1)<w)&~rule_va).sum())} (원래 {int(((np.abs(Dv.lr.values-1)<w)&~rule_va).sum())}) · train OOF {int(((np.abs(l2-1)<w)&~rule_tr).sum())} (원래 {int(((np.abs(lr_oof-1)<w)&~rule_tr).sum())})")
pv2 = (lv2 >= 1.0) | rule_va
P(f"val 판정 수 {int(pv2.sum())} (원래 {int(pv_all.sum())}) · 바뀐 파일: 새로 고장 {list(np.array(nv)[pv2 & ~pv_all])} · 정상으로 {list(np.array(nv)[~pv2 & pv_all])}")
pd.DataFrame(dict(file=nv, ae_z=Dv.ae_z, ae_prime_z=zv2, et_z=Dv.et_z, lr=Dv.lr, lr_prime=lv2, gain=gain_va, rule=rule_va)).round(3).to_csv(os.path.join(HERE, "val_AE게인보정.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(dict(file=names, y=y, ae_z=oofz[:, 0], ae_prime_z=z2, et_z=oofz[:, 1], lr=lr_oof, lr_prime=l2, gain=gain_tr)).round(3).to_csv(os.path.join(HERE, "train_AE게인보정.csv"), index=False, encoding="utf-8-sig")
# 라벨 없는 분위 정렬 보정: val의 하위 25 %를 정상의 하위 25 %에 맞춤(val은 정상이 80 % 안팎이라 하위는 정상이 대부분)
for k, nm in enumerate(("AE", "ET")):
    zt = oofz[:, k]; zv = Dv[["ae_z", "et_z"][k]].values
    sh = np.median(np.sort(zv)[:10]) - np.median(np.sort(zt[nor])[:30]); P(f"{nm} 하위 25% 정렬 이동량 {sh:+.2f}")
shA = np.median(np.sort(Dv.ae_z.values)[:10]) - np.median(np.sort(oofz[nor, 0])[:30]); shE = np.median(np.sort(Dv.et_z.values)[:10]) - np.median(np.sort(oofz[nor, 1])[:30])
lv3 = core.lr_.decision_function(np.c_[Dv.ae_z.values - shA, Dv.et_z.values - shE]); pv3 = (lv3 >= 1.0) | rule_va
P(f"AE만 정렬: 판정 {int((core.lr_.decision_function(np.c_[Dv.ae_z.values-shA, Dv.et_z.values])>=1).sum())}; AE·ET 둘 다 정렬: 판정 {int(pv3.sum())} (원래 {int(pv_all.sum())}) 새로 고장 {list(np.array(nv)[pv3 & ~pv_all])} 정상으로 {list(np.array(nv)[~pv3 & pv_all])}; |lr-1|<0.5 {int(((np.abs(lv3-1)<0.5)&~rule_va).sum())}")
