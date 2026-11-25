# %% 2-6 ① 바닥 보정(b_floor) 효과 ② Train vs val 분포 이동(계산 ③) ③ 기준 ExtraTrees 중첩 CV (DC 그룹 묶음)
# 입력: 2-2와 같은 검출기 · 이벤트 정의. Train 150 + val 40(라벨 미사용). 5개 피처 = Gemini 7개 초안에서 off_med_rel_amp · off_mean_fit 제외.
import glob, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.impacts import detect_impacts, highpass, _atoms50, FS
from src.final import _SEG
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
A50, F50 = _atoms50(); ZS = (0.01, 0.02, 0.04, 0.08)

def fine_ring(xh, idx):
    S = np.stack([xh[p:p + _SEG] for p in idx]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2); b = fit.argmax(1)
    return np.column_stack([F50[b], np.array(ZS)[b % 4], fit.max(1)])
def clusters(idx, gap_ms=25):
    return np.concatenate([[0], np.cumsum(np.diff(idx) > gap_ms * FS / 1000)]).astype(int) if len(idx) else np.array([], int)

# %% 이벤트 (190개 파일), 캐시
cache = os.path.join(FIG, "events_190.csv")
if os.path.exists(cache):
    E = pd.read_csv(cache); names = pd.read_csv(os.path.join(FIG, "files_190.csv"))
else:
    ev, nm = [], []
    for grp_, pat in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
        for f in sorted(glob.glob(os.path.join(ROOT, "data", pat, "*.csv"))):
            n = os.path.basename(f)[:-4]; x = np.loadtxt(f); xh = highpass(x); imp = detect_impacts(x, _hp=xh)
            idx, hg = imp["idx"], imp["height_g"]; nm.append(dict(file=n, group=grp_))
            if len(idx):
                ring = fine_ring(xh, idx); cl = clusters(idx)
                for e in range(cl.max() + 1):
                    m = np.where(cl == e)[0]; j = m[np.argmax(hg[m])]
                    ev.append(dict(file=n, group=grp_, t=idx[j] / FS, f=ring[j, 0], fit=ring[j, 2], h=hg[j], x=imp["height_x"][j]))
    E = pd.DataFrame(ev); names = pd.DataFrame(nm); E.to_csv(cache, index=False); names.to_csv(os.path.join(FIG, "files_190.csv"), index=False)
E["base"] = E.h / E.x
files = names.file.tolist(); grp = names.set_index("file").group
y = (grp == "abnormal").astype(int)
base_med = E.groupby("file").base.median().reindex(files)      # 파일의 바탕 중앙값(g), 이벤트가 없으면 NaN
L = []
R83 = pd.read_csv(os.path.join(ROOT, "단계", "아카이브", "new_단계", "2_EDA", "2-4_이전발견_가져오기", "figs", "rule83_files.csv")).set_index("file").rule.astype(bool).reindex(files)

LOW, HIGH = (1600, 4500), (8000, 11000)
EV = {f: g for f, g in E.groupby("file")}
def feats(floor):
    """5개 피처. floor = None이면 보정 없음, 아니면 바탕을 max(바탕, floor)로 올림(g 단위 하한)"""
    rows = []
    for f in files:
        e = EV.get(f)
        if e is None:
            rows.append(dict(file=f, cnt_low=0, cnt_high=0, cnt_total=0, ratio_off=0.0, off_max_amp=0.0)); continue
        lo = e[(e.f >= LOW[0]) & (e.f < LOW[1])]; hi = e[(e.f >= HIGH[0]) & (e.f < HIGH[1])]; off = pd.concat([lo, hi])
        amp = 0.0
        if len(off):
            b = off.base.values if floor is None else np.maximum(off.base.values, floor)
            amp = float((off.h.values / b).max())
        rows.append(dict(file=f, cnt_low=len(lo), cnt_high=len(hi), cnt_total=len(e), ratio_off=(len(lo) + len(hi)) / (len(e) + 1), off_max_amp=amp))
    return pd.DataFrame(rows).set_index("file")
tr_files = [f for f in files if grp[f] != "val"]

# %% ① 바닥 보정
L.append("① 바닥 보정(b_floor) 효과. 하한 = Train 150개 파일의 '바탕 중앙값' 분포의 하위 q 분위(라벨 없이). off = 1.6–4.5 + 8–11 kHz 이벤트")
bm = base_med[tr_files].dropna()
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
rng = np.random.default_rng(0)
def ci(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]; return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
F0 = feats(None); fa_names = [f for f in tr_files if grp[f] == "normal" and (F0.cnt_low[f] + F0.cnt_high[f]) >= 2]
abn = [f for f in tr_files if grp[f] == "abnormal"]; nor = [f for f in tr_files if grp[f] == "normal"]
L.append(f"오경보 정상(off>=2) {len(fa_names)}개 (2-5와 동일해야 18): " + ", ".join(fa_names))
variants = [("보정 없음", None)] + [(f"하위 {q}%", float(np.percentile(bm, q))) for q in (10, 15, 20, 30)]
res = {}
for nme, fl in variants:
    Fv = feats(fl); res[nme] = Fv
    a1 = auc(Fv.off_max_amp[abn], Fv.off_max_amp[fa_names]); lo1, hi1 = ci(Fv.off_max_amp[abn], Fv.off_max_amp[fa_names])
    a2 = auc(Fv.off_max_amp[abn], Fv.off_max_amp[nor])
    fls = "-" if fl is None else f"{fl:.4f} g"
    L.append(f"{nme:8s} 하한 {fls:9s} | AUC vs 오경보18 {a1:.2f} [{lo1:.2f}, {hi1:.2f}] | AUC vs 정상120 {a2:.2f} | nor_054 값 {Fv.off_max_amp['nor_054']:.1f} | 고장 중앙 {Fv.off_max_amp[abn].median():.1f}, 오경보18 중앙 {Fv.off_max_amp[fa_names].median():.1f}")
L.append("바탕 중앙값(g) 분포(Train, 이벤트 있는 파일 %d개): p10 %.4f p15 %.4f p20 %.4f p30 %.4f 중앙 %.4f" % (len(bm), *[np.percentile(bm, q) for q in (10, 15, 20, 30, 50)]))
L.append("nor_054 바탕(g): %.4f" % base_med["nor_054"])

# %% ② 분포 이동
L.append("\n② 계산 ③ Train 150 vs val 40 (라벨 미사용) 피처 분포")
vfiles = [f for f in files if grp[f] == "val"]
for nme in ("보정 없음", "하위 20%"):
    Fv = res[nme]; L.append(f"[{nme}]")
    for c in Fv.columns:
        a, b = Fv.loc[tr_files, c], Fv.loc[vfiles, c]
        out = (b > np.percentile(a, 95)).mean()
        L.append(f"{c:12s} Train {a.median():7.2f} [{np.percentile(a,25):.2f}, {np.percentile(a,75):.2f}] p80 {np.percentile(a,80):7.2f} | val {b.median():7.2f} [{np.percentile(b,25):.2f}, {np.percentile(b,75):.2f}] p80 {np.percentile(b,80):7.2f} | val>Train p95 {out*100:4.0f}% (기대 5%) | MWU p {mannwhitneyu(a,b).pvalue:.3f} KS p {ks_2samp(a,b).pvalue:.3f}")
L.append(f"8.3k 규칙 켜짐: Train {R83[tr_files].sum()}/150, val {R83[vfiles].sum()}/40 ({', '.join(f for f in vfiles if R83[f])})")
bv = base_med[vfiles]; L.append(f"바탕 중앙값(g): Train 중앙 {bm.median():.4f}, val 중앙 {bv.median():.4f} (이벤트 없는 val {bv.isna().sum()}개)")

# %% ③ 중첩 CV
def f2s(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0
grid = np.linspace(0.05, 0.95, 19)
tf = np.array(tr_files); yy = y[tr_files].values
gid = np.array([("DC_097_101_120" if f in ("nor_097", "nor_101", "nor_120") else f) for f in tf])
CFG = [(3, 3), (3, 5), (4, 3), (4, 5)]
def mk(d, m, seed): return ExtraTreesClassifier(300, max_depth=d, min_samples_leaf=m, class_weight="balanced", random_state=seed, n_jobs=-1)
def nested(floor_q, d, m, seeds=range(5)):
    out = []; r83 = R83[tf].values
    for s in seeds:
        sk = StratifiedGroupKFold(5, shuffle=True, random_state=s); pr = np.zeros(150, int); pr_or = np.zeros(150, int)
        for itr, ite in sk.split(tf, yy, gid):
            fl = None if floor_q is None else float(np.percentile(base_med[tf[itr]].dropna(), floor_q))
            X = feats(fl).loc[tf].values
            oof = np.zeros(len(itr)); ik = StratifiedKFold(4, shuffle=True, random_state=s)
            for a, b in ik.split(itr, yy[itr]):
                oof[b] = mk(d, m, s).fit(X[itr][a], yy[itr][a]).predict_proba(X[itr][b])[:, 1]
            sc = np.array([f2s(yy[itr], (oof >= t).astype(int)) for t in grid]); thr = grid[np.where(sc == sc.max())[0]].mean()
            p = mk(d, m, s).fit(X[itr], yy[itr]).predict_proba(X[ite])[:, 1]
            pr[ite] = (p >= thr); pr_or[ite] = ((p >= thr) | r83[ite])
        out.append((f2s(yy, pr), f2s(yy, pr_or), pr.copy(), pr_or.copy()))
    return out
L.append("\n③ 기준 ExtraTrees 중첩 CV (바깥 5-fold StratifiedGroupKFold × 5 반복, 안쪽 4-fold로 문턱 선택, 트리 300). 그룹 = 097·101·120 묶음, 나머지 파일당 1그룹")
L.append("피처 5개 = cnt_low · cnt_high · cnt_total · ratio_off · off_max_amp. 하한은 학습 폴드 파일로만 계산. OR 8.3k 열 = [사후 값] (규칙 문턱이 Train 라벨을 본 뒤 정해짐)")
for floor_q, lab in ((None, "보정 없음"), (20, "하위 20%")):
    for d, m in CFG:
        o = nested(floor_q, d, m)
        a = np.array([[r[0], r[1]] for r in o]); P = np.mean([r[2] for r in o], 0); P2 = np.mean([r[3] for r in o], 0)
        miss = tf[(yy == 1) & (P < 0.5)]; fa = tf[(yy == 0) & (P >= 0.5)]
        L.append(f"{lab:8s} depth {d} leaf {m} | ET 단독 F2 {a[:,0].mean():.3f} ± {a[:,0].std():.3f} | ET OR 8.3k[사후 값] F2 {a[:,1].mean():.3f} ± {a[:,1].std():.3f} | 과반 기준(ET 단독) 미탐 {len(miss)}, 오경보 {len(fa)}")
        if (d, m) == (3, 3):
            L.append("   ET 단독 미탐(5회 중 과반): " + ", ".join(miss) + " | 오경보: " + ", ".join(fa))
            miss2 = tf[(yy == 1) & (P2 < 0.5)]; fa2 = tf[(yy == 0) & (P2 >= 0.5)]
            L.append("   OR 8.3k 후 미탐: " + ", ".join(miss2) + f" | 오경보 {len(fa2)}개")
L.append("참고: 8.3k 규칙 단독(Train 150): 고장 %d/30, 정상 %d/120 → F2 %.3f" % (R83[abn].sum(), R83[nor].sum(), f2s(yy, R83[tf].values.astype(int))))
open(os.path.join(FIG, "계산_수치.txt"), "w", encoding="utf-8").write("\n".join(L)); print("done")
