# F 피처 담당 step2: 라벨 없는 울림 분포 피처 후보 + ext150 D 묶음 정리 + quick_cv(ET 축만, 탐색용). 결과는 result_F.txt (UTF-8)
import os, sys, collections
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from src.exp import make_et
from src.evaluate import quick_cv
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "result_F.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n"); OUT.flush()

# ---------- 데이터 ----------
IMP = pd.read_csv(os.path.join(HERE, "impacts_new.csv"), encoding="utf-8-sig")
NI = pd.read_csv(os.path.join(HERE, "nimp.csv"), encoding="utf-8-sig")
files = NI.file.tolist(); lab = NI.y.values; N = len(files); ntr = 150
y = lab[:ntr].astype(int); isn = y == 0
MM = pd.read_csv(os.path.join(_d, "3_단계", "4_피처", "아카이브", "4-8_1121_모드지도", "features_modemap.csv"), encoding="utf-8-sig").set_index("file").loc[files]
EX = pd.read_csv(os.path.join(_d, "4_outputs", "cache", "feat_ext.csv")); EXv = pd.read_csv(os.path.join(_d, "4_outputs", "cache", "feat_ext_val.csv"))
assert list(EX.columns) == list(EXv.columns) and len(EX) == 150 and len(EXv) == 40
E_all = np.nan_to_num(np.vstack([EX.values, EXv.values]))
OLD = pd.read_csv(os.path.join(_d, "3_단계", "2_EDA", "2-3_충격울림주파수", "자료", "figs", "impacts.csv"), encoding="utf-8-sig")
def oldfrac(lo, hi):
    return OLD.assign(b=(OLD.f >= lo) & (OLD.f < hi)).groupby("file").b.mean().reindex(files).values.astype(float)
R24_old = np.nan_to_num(oldfrac(2125, 4125), nan=-1.0); RBODY_old = np.nan_to_num(oldfrac(5500, 7750), nan=-1.0)

# ---------- 라벨 없이 고정한 정의 ----------
BODY = 7000.0      # 본체 모드: 0단계 사전 규칙 KDE 봉우리 6.9-7.1 kHz -> 7.0 kHz
VALLEY = 3800.0    # 저/고 경계: 0단계 사전 규칙 골짜기 3.6-3.9 kHz (6개 설정 모두) -> 3.8 kHz
KEYS = ["lo_share", "lo_share_w", "f_med", "f_max", "dev_med", "dev_max", "up_dev_max", "dn_dev_max", "devfit_max"]
def feats_from(f, fit):
    if len(f) == 0:
        return {k: np.nan for k in KEYS}
    lg = np.log2(f / BODY); dev = np.abs(lg); w = fit + 1e-9
    return dict(lo_share=float(np.mean(f < VALLEY)), lo_share_w=float(np.sum(w * (f < VALLEY)) / np.sum(w)),
                f_med=float(np.median(f) / 1000), f_max=float(np.max(f) / 1000),
                dev_med=float(np.median(dev)), dev_max=float(np.max(dev)),
                up_dev_max=float(max(0.0, lg.max())), dn_dev_max=float(max(0.0, -lg.min())), devfit_max=float(np.max(dev * fit)))
G = {k: v for k, v in IMP.groupby("file")}
rows, rows_odd, rows_even = [], [], []
for fl in files:
    if fl in G:
        g = G[fl].sort_values("t"); f = g.f.values; ft = g.fit.values
    else:
        f = np.zeros(0); ft = np.zeros(0)
    rows.append(feats_from(f, ft)); rows_odd.append(feats_from(f[0::2], ft[0::2])); rows_even.append(feats_from(f[1::2], ft[1::2]))
FD = pd.DataFrame(rows, index=files); FO = pd.DataFrame(rows_odd, index=files); FE = pd.DataFrame(rows_even, index=files)
FD["f_tsa"] = MM.f_tsa.values / 1000.0
FD["dev_tsa"] = np.abs(np.log2(MM.f_tsa.values / BODY))
nimp_arr = NI.n_imp.values
P("== 정의 고정(라벨 없음): 본체 BODY=7.0 kHz(KDE 봉우리), 저/고 경계 VALLEY=3.8 kHz(KDE 골짜기), 파일 대표 울림 f_tsa = 4-8 정렬 평균 ==")
P("f_tsa NaN 파일 수 train/val:", int(FD.f_tsa[:ntr].isna().sum()), int(FD.f_tsa[ntr:].isna().sum()), "| 충격 0개 파일 train정상/고장/val:", int(((nimp_arr[:ntr] == 0) & isn).sum()), int(((nimp_arr[:ntr] == 0) & ~isn).sum()), int((nimp_arr[ntr:] == 0).sum()))
CAND = ["f_tsa", "dev_tsa", "lo_share", "lo_share_w", "f_med", "f_max", "dev_med", "dev_max", "up_dev_max", "dn_dev_max", "devfit_max"]

# ---------- 진단 1 ----------
def rho_max(v, M):
    ok = ~np.isnan(v); best = (0.0, "")
    for j in range(M.shape[1]):
        c = spearmanr(v[ok], M[ok, j]).correlation
        if c == c and abs(c) > abs(best[0]): best = (c, EX.columns[j])
    return best
P("\n== 후보 피처 진단 ==")
P("feature | nor_p5 | nor_med | nor_p95 | val_med | val_out(정상5-95% 밖 val 비율, 기대 .10) | (val-정상 중앙)/정상 sd | ext150 최대|rho|(열) | A울림2열 최대|rho| | 홀짝 rho(충격>=4) | NaN(train/val)")
diag = []
for c in CAND:
    v = FD[c].values; vn = v[:ntr][isn]; vn = vn[~np.isnan(vn)]; vv = v[ntr:]; vvn = vv[~np.isnan(vv)]
    p5, med, p95 = np.percentile(vn, [5, 50, 95]); sd = vn.std(ddof=1) + 1e-12
    vout = float(np.mean((vvn < p5) | (vvn > p95))); shift = (np.median(vvn) - med) / sd
    r, rc = rho_max(v[:ntr], E_all[:ntr])
    okk = ~np.isnan(v[:ntr])
    ra = max(abs(spearmanr(v[:ntr][okk], a[:ntr][okk]).correlation) for a in (R24_old, RBODY_old))
    sel = nimp_arr >= 4
    if c in FO.columns:
        ok = sel & ~np.isnan(FO[c].values) & ~np.isnan(FE[c].values); rh = spearmanr(FO[c].values[ok], FE[c].values[ok]).correlation
    else:
        rh = np.nan
    P(f"{c} | {p5:.3g} | {med:.3g} | {p95:.3g} | {np.median(vvn):.3g} | {vout:.3f} | {shift:+.2f} | {r:+.2f} ({rc}) | {ra:.2f} | {rh:.2f} | {int(np.isnan(v[:ntr]).sum())}/{int(np.isnan(vv).sum())}")
    diag.append(dict(feature=c, nor_p5=p5, nor_med=med, nor_p95=p95, val_med=float(np.median(vvn)), val_out=vout, shift_sd=shift, ext_maxrho=r, ext_maxrho_col=rc, A2_maxrho=ra, split_half=rh))
pd.DataFrame(diag).to_csv(os.path.join(HERE, "F_diag.csv"), index=False, encoding="utf-8-sig")
for nm, v in (("A_R24(예전정의 2.125-4.125k 몫)", R24_old), ("A_RBODY(예전정의 5.5-7.75k 몫)", RBODY_old)):
    vn = v[:ntr][isn]; vn = vn[vn >= 0]; vv = v[ntr:]; vv = vv[vv >= 0]; p5, med, p95 = np.percentile(vn, [5, 50, 95])
    P(f"[참고] {nm}: 정상 p5/med/p95 {p5:.2f}/{med:.2f}/{p95:.2f} | val med {np.median(vv):.2f} | val_out {np.mean((vv < p5) | (vv > p95)):.3f}")

# ---------- 보고 ----------
P("\n== 보고(라벨은 정의 고정 뒤에만): 값(정상 중 이 파일 이하 비율) ==")
tgt = ["abn_002", "abn_003", "abn_006", "abn_010", "abn_029", "abn_014", "abn_017", "abn_022", "abn_005", "abn_015", "nor_116", "nor_055", "nor_059", "nor_025", "nor_019", "nor_092", "nor_043"]
P("file | n_imp | " + " | ".join(CAND))
for t in tgt:
    i = files.index(t); out = []
    for c in CAND:
        v = FD[c].values; vn = v[:ntr][isn]; vn = vn[~np.isnan(vn)]
        out.append("nan" if np.isnan(v[i]) else f"{v[i]:.2f}({(vn <= v[i]).mean() * 100:.0f}%)")
    P(f"{t} | {nimp_arr[i]} | " + " | ".join(out))
P("\n단변량 AUC(고장 30 vs 정상 120, 보고용·사후, NaN은 정상 중앙값):")
for c in CAND:
    v = FD[c].values[:ntr].copy(); m = np.nanmedian(v[isn]); v = np.where(np.isnan(v), m, v); P(f"  {c}: AUC {roc_auc_score(y, v):.3f}")

# ---------- ext150 묶음 ----------
Etr = E_all[:ntr]
R = np.abs(np.nan_to_num(spearmanr(Etr).correlation, nan=0.0)); np.fill_diagonal(R, 1.0)
L = linkage(squareform(1 - R, checks=False), "complete"); cl = fcluster(L, t=0.2, criterion="distance")
cols = list(EX.columns); groups = {}
for j, c in enumerate(cl): groups.setdefault(c, []).append(j)
rep = {}
for c, js in groups.items():
    rep[c] = js[0] if len(js) == 1 else js[int(np.argmax([np.mean([R[a, b] for b in js if b != a]) for a in js]))]
REP = sorted(rep.values())
fam = lambda c: c.split("_")[0]
P(f"\n== ext150 |rho|>=0.8 완전연결 묶음(train 150, 라벨 없음): {len(groups)}묶음, 대표 {len(REP)}열 ==")
rows = []
for c, js in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    nm_ = [cols[j] for j in js]
    if len(js) >= 3:
        P(f"묶음 {c} ({len(js)}열) 대표 {cols[rep[c]]} | 가족 {dict(collections.Counter(fam(n) for n in nm_))} | 예 {nm_[:5]}")
    rows.append(dict(cluster=int(c), size=len(js), rep=cols[rep[c]], members=";".join(nm_)))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "ext150_clusters.csv"), index=False, encoding="utf-8-sig")
vo = {}
for j, c in enumerate(cols):
    p5, p95 = np.percentile(Etr[isn, j], [5, 95]); vv = E_all[ntr:, j]; vo[c] = float(np.mean((vv < p5) | (vv > p95)))
vo = pd.Series(vo); fm = collections.defaultdict(list)
for c in cols: fm[fam(c)].append(vo[c])
P("가족별 val_out 중앙값(기대 .10):", {k: round(float(np.median(v)), 2) for k, v in fm.items()})
P("val_out >= .30 열 수:", int((vo >= .30).sum()), "/ 150 ; 상위 12:", vo.sort_values(ascending=False).head(12).round(2).to_dict())
SIZE_PREF = ("T01_", "T02_", "T03_", "T04_", "T13_", "T17_", "T19_", "B05_", "P02_rms", "E01_rms")
SIZE = [j for j, c in enumerate(cols) if c.startswith(SIZE_PREF)]; T06 = cols.index("T06_raw_mean")
REP_noT06 = [j for j in REP if j != T06]; REP_noSize = [j for j in REP_noT06 if j not in SIZE]
P(f"대표 {len(REP)}열 중 크기 계열 {len([j for j in REP if j in SIZE])}열, T06 포함 {int(T06 in REP)} -> T06 뺀 {len(REP_noT06)}열, 크기 뺀 {len(REP_noSize)}열")
P("대표 열 이름:", [cols[j] for j in REP])

# ---------- quick_cv ----------
def newF(cs):
    return np.column_stack([np.nan_to_num(FD[c].values, nan=(-1.0 if c in ("lo_share", "lo_share_w") else 0.0)) for c in cs])
A2 = np.column_stack([R24_old, RBODY_old])
Fmin = newF(["lo_share", "dev_tsa"]); F123 = newF(["f_tsa", "lo_share", "f_max"])
Fall = newF(["f_tsa", "dev_tsa", "lo_share", "f_med", "f_max", "dev_med", "dev_max", "up_dev_max", "dn_dev_max", "devfit_max"])
Fdev = newF(["dev_tsa", "dev_max", "up_dev_max", "dn_dev_max", "devfit_max"]); Flo = newF(["lo_share", "dn_dev_max"])
Fup = newF(["f_tsa", "f_max", "up_dev_max", "lo_share"])
SETS = [
    ("S0 ext150", E_all),
    ("SA ext150 + 울림2열(예전정의) = A의 ET 입력", np.hstack([E_all, A2])),
    ("SB ext150 + lo_share + dev_tsa", np.hstack([E_all, Fmin])),
    ("SC ext150 + f_tsa + lo_share + f_max", np.hstack([E_all, F123])),
    ("SD ext150 + F 연속 10열", np.hstack([E_all, Fall])),
    ("SE ext150 + 편차계열 5열", np.hstack([E_all, Fdev])),
    ("SF ext150 + lo_share + dn_dev_max", np.hstack([E_all, Flo])),
    ("SG ext150 + f_tsa + f_max + up_dev_max + lo_share", np.hstack([E_all, Fup])),
    (f"SH 대표 {len(REP)}열", E_all[:, REP]),
    (f"SI 대표 {len(REP)}열 + 울림2열(예전)", np.hstack([E_all[:, REP], A2])),
    (f"SJ 대표 {len(REP)}열 + F 10열", np.hstack([E_all[:, REP], Fall])),
    (f"SK 대표 T06 뺀 {len(REP_noT06)}열 + F 10열", np.hstack([E_all[:, REP_noT06], Fall])),
    (f"SL 대표 크기·T06 뺀 {len(REP_noSize)}열 + F 10열", np.hstack([E_all[:, REP_noSize], Fall])),
    (f"SM 대표 크기·T06 뺀 {len(REP_noSize)}열 + 울림2열(예전)", np.hstack([E_all[:, REP_noSize], A2])),
    (f"SN 대표 크기·T06 뺀 {len(REP_noSize)}열", E_all[:, REP_noSize]),
]
P("\n== quick_cv (ET 축 단독, 5-fold x seed 5, PR-AUC 평균 ± seed sd, 탐색용). 기준 SA = A의 ET 입력 ==")
RES = {nm: quick_cv(M[:ntr], y, np.arange(ntr), make_et(300, 0), seeds=range(5)) for nm, M in SETS}
base = RES[SETS[1][0]]
tg_f = ["abn_002", "abn_029", "abn_003", "abn_006", "abn_010", "abn_014", "abn_017", "abn_022", "abn_005", "abn_015"]
tg_n = ["nor_116", "nor_055", "nor_059", "nor_025"]
P("set | n열 | PR-AUC | ROC-AUC | SA 대비 PR-AUC 차(seed 짝) | 겨냥 고장 miss_rank(0=최상) | 오경보 후보 정상 백분위(정상 중 더 낮은 비율)")
nor_idx = np.flatnonzero(isn)
for nm, M in SETS:
    r = RES[nm]; d = r["pr_auc"] - base["pr_auc"]
    mr = " ".join(f"{t[-3:]}:{r['miss_rank'][files.index(t)]:.2f}" for t in tg_f)
    oof = r["oof"]; nr = []
    for t in tg_n:
        i = files.index(t); others = [k for k in nor_idx if k != i]
        nr.append(f"{t[-3:]}:{np.mean([(oof[s, others] < oof[s, i]).mean() for s in range(oof.shape[0])]):.2f}")
    P(f"{nm} | {M.shape[1]} | {r['pr_auc'].mean():.3f} ± {r['pr_auc'].std(ddof=1):.3f} | {r['roc_auc'].mean():.3f} | {d.mean():+.3f} ± {d.std(ddof=1):.3f} | {mr} | {' '.join(nr)}")

# ---------- val ET 단독 ----------
P("\n== val ET 단독: train 150 전체 학습 ET(seed 5 평균) 고장 확률. 경계 = train OOF 정상 90% 분위, 밀집 = |val-경계| < 0.5 x 정상 OOF sd (라벨 없는 정의) ==")
P("set | val 평균확률 | val>=경계 개수 | 밀집(±0.5sd) 개수 | train 정상 OOF 평균 | OOF sd(정상)")
for nm, M in SETS:
    oof = RES[nm]["oof"].mean(0); thr = np.percentile(oof[isn], 90); sd = oof[isn].std(ddof=1)
    pv = np.mean([make_et(300, s)().fit(M[:ntr], y).predict_proba(M[ntr:])[:, 1] for s in range(5)], axis=0)
    P(f"{nm} | {pv.mean():.3f} | {int((pv >= thr).sum())} | {int((np.abs(pv - thr) < 0.5 * sd).sum())} | {oof[isn].mean():.3f} | {sd:.3f}")
P("완료")
