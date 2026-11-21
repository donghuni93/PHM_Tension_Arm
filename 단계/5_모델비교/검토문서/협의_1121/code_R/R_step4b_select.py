"""R 4b (11-22 원칙): 라벨로 규칙이 보던 양을 골라, 폴드 안 재선택 안정성 · 연속 피처 · 전문가 u · 세 번째 축 · t-SNE 5-NN 순도를 잰다.
라벨은 선택에 적극 사용하되 매 폴드의 학습 행에서만(폴드 안) 고른다. 시험 행은 선택에 쓰지 않는다. 중첩 CV 아님(탐색용).
출력: out/R4b.txt, out/R4b_*.csv, out/R4b_tsne.png
"""
import os, sys, pickle, io, json
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
here = os.path.dirname(os.path.abspath(__file__)); root = here
while not os.path.isdir(os.path.join(root, "src")): root = os.path.dirname(root)
sys.path.insert(0, root); os.chdir(root)
import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from src.exp import load_train, cached_table, feature_table_ext
from src.features2 import cached_feat2, PeakRef, robust_z

R = pickle.load(open(os.path.join(here, "out", "raw.pkl"), "rb"))
names, split, yy, imps, T2, f, PR = R["names"], R["split"], R["y"], R["imps"], R["T2"], R["f"], R["PR"]
n_all = len(names); TRI = np.flatnonzero(split == "train"); VAL = np.flatnonzero(split == "val")
y = yy[TRI]; n = len(y); nor = y == 0
sn = [s.replace("abn_", "A").replace("nor_", "N").replace("val_", "V") for s in names]
buf = io.StringIO()
def p(*a): print(*a, file=buf)

# ----- 파일별 원재료 (전체 190개, 선택은 폴드 안)
rf = [im["rings"][:, 0] if len(im["rings"]) else np.zeros(0) for im in imps]
rz = [im["rings"][:, 1] if len(im["rings"]) else np.zeros(0) for im in imps]
rt = [im["rings"][:, 2] if len(im["rings"]) else np.zeros(0) for im in imps]
def ring_v(lo, hi, zmax):
    out = np.zeros(n_all)
    for i in range(n_all):
        m = (rf[i] >= lo) & (rf[i] < hi) & (rz[i] <= zmax)
        if m.any(): out[i] = rt[i][m].max()
    return out
RING = {(lo, hi, z): ring_v(lo, hi, z) for lo in (7750, 8000, 8250) for hi in (8500, 8750, 9000) for z in (0.01, 0.02, 0.04)}
def band_pw(lo, hi):
    b = (f >= lo) & (f < hi); ia = np.flatnonzero(b)
    P, W = PR[:, b].max(1), np.zeros(n_all)
    for i in range(n_all):
        k = ia[PR[i, b].argmax()]; h = PR[i, k] / 2; l = r = k
        while l > 0 and PR[i, l - 1] >= h: l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
        W[i] = f[r] - f[l]
    return P, W
SPEC = {(lo, hi): band_pw(lo, hi) for lo in (8000, 8100, 8200) for hi in (8450, 8600, 8750)}
hx = [np.sort(im["height_x"])[::-1] for im in imps]
K1n = np.array([np.log10(h[1]) if len(h) > 1 else np.log10(4.5) for h in hx])
K3n = T2["H2_ratio21"].values.astype(float); H18 = T2["H18_hfl"].values.astype(float)
nimp = np.expm1(T2["H1_nimp"].values).round().astype(int)
H18g = np.where(nimp >= 1, H18, -9.0)          # 충격 0개는 켜지지 않게 (E2의 '충격>=1')


# ----- 규칙 형태별 폴드 안 선택기 (학습 행 tr -> 정의 dict)
def pick_ring(tr):
    best = None
    for key, v in RING.items():
        a = roc_auc_score(y[tr], v[TRI][tr]) - 1e-4 * (key[1] - key[0]) / 250      # 동점이면 더 좁은 대역
        if best is None or a > best[0]: best = (a, key)
    key = best[1]; v = RING[key][TRI]; mx = v[tr][nor[tr]].max()
    cand = [t for t in np.arange(0.5, 0.96, 0.05) if t > mx + 1e-9]
    return dict(key=key, t=float(cand[0] if cand else 0.95))       # 학습 정상이 하나도 안 켜지는 가장 낮은 문턱 (fit 하한 0.5)


def u_ring(d): return RING[d["key"]] / d["t"]


def pick_spec(tr):
    best = None
    for key, (P, W) in SPEC.items():
        for pt in np.arange(0.4, 1.21, 0.05):
            for wt in (100, 125, 150, 175, 200, 225, 250):
                on = np.minimum(P[TRI] / pt, W[TRI] / wt) >= 1
                if on[tr][nor[tr]].sum() == 0:
                    sc = (int(on[tr][~nor[tr]].sum()), pt * wt)      # 정상 0개 켜짐 중 고장 많은 쪽, 동점이면 문턱 곱이 큰 쪽(여유)
                    if best is None or sc > best[0]: best = (sc, key, pt, wt)
    _, key, pt, wt = best
    return dict(key=key, pt=float(pt), wt=float(wt))


def u_spec(d):
    P, W = SPEC[d["key"]]; return np.minimum(P / d["pt"], W / d["wt"])


def pick_J(tr):
    best = None
    for a in np.arange(1.0, 2.21, 0.1):
        for b in np.arange(0.4, 1.01, 0.05):
            on = (K1n[TRI] >= a) & (K3n[TRI] >= b)
            if on[tr][nor[tr]].sum() == 0:
                sc = (int(on[tr][~nor[tr]].sum()), a * b)
                if best is None or sc > best[0]: best = (sc, a, b)
    _, a, b = best
    return dict(a=float(a), b=float(b))


def u_J(d): return np.minimum(K1n / d["a"], K3n / d["b"])


def pick_L(tr):
    return dict(t=float(H18g[TRI][tr][nor[tr]].max() + 1e-6))


PICK = {"ring": pick_ring, "spec": pick_spec, "J": pick_J, "L": pick_L}
EXPK = ["ring", "spec", "J", "L"]


def u_all(sel, tr):
    """선택 sel + 학습 행 tr -> 파일 190개의 전문가 u (>=1 켜짐)"""
    U = {"ring": u_ring(sel["ring"]), "spec": u_spec(sel["spec"]), "J": u_J(sel["J"])}
    mL = np.median(H18g[TRI][tr][nor[tr]]); tL = sel["L"]["t"]
    U["L"] = np.where(nimp >= 1, (H18 - mL) / max(tL - mL, 1e-6), 0.0)
    return U


def select_all(tr): return {k: PICK[k](tr) for k in EXPK}


# ================= 1. 폴드 안 재선택 안정성 (5겹 x 20 seed) =================
chosen = {k: [] for k in EXPK}
hit = np.zeros((n, 4)); cnt = np.zeros(n); folds = []; union_f = np.zeros(n)
for seed in range(20):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
        sel = select_all(tr); folds.append((seed, tr, te, sel))
        for k in EXPK: chosen[k].append(sel[k])
        U = u_all(sel, tr); on = np.zeros(len(te), bool)
        for j, k in enumerate(EXPK):
            hit[te, j] += (U[k][TRI][te] >= 1); on |= U[k][TRI][te] >= 1
        union_f[te] += on; cnt[te] += 1
p("[1] 폴드 안 재선택 안정성: 5겹 x 20 seed = 100개 학습 폴드")
c = Counter(d["key"] for d in chosen["ring"]); p("  ring (8.3k 깨끗한 울림) 대역(lo,hi,zeta최대) 최빈: " + "; ".join(f"{k}: {v}%" for k, v in c.most_common(4)))
c = Counter(round(d["t"], 2) for d in chosen["ring"]); p("    fit 문턱 분포: " + "; ".join(f"{k}: {v}%" for k, v in sorted(c.items())))
c = Counter(d["key"] for d in chosen["spec"]); p("  spec (8.3k 스펙트럼 돌출·폭) 대역 최빈: " + "; ".join(f"{k}: {v}%" for k, v in c.most_common(4)))
c = Counter((round(d["pt"], 2), d["wt"]) for d in chosen["spec"]); p("    (P문턱, W문턱) 최빈: " + "; ".join(f"{k}: {v}%" for k, v in c.most_common(5)))
pts = np.array([d["pt"] for d in chosen["spec"]]); wts = np.array([d["wt"] for d in chosen["spec"]])
p(f"    P문턱 분위(0,25,50,75,100) {np.round(np.percentile(pts,[0,25,50,75,100]),2)}  W문턱 {np.percentile(wts,[0,25,50,75,100])}")
c = Counter((round(d["a"], 1), round(d["b"], 2)) for d in chosen["J"]); p("  J (유격 반복: K1n>=a 그리고 K3n>=b): (a,b) 최빈: " + "; ".join(f"{k}: {v}%" for k, v in c.most_common(6)))
aa = np.array([d["a"] for d in chosen["J"]]); bb = np.array([d["b"] for d in chosen["J"]])
p(f"    a 분위(0,25,50,75,100) {np.round(np.percentile(aa,[0,25,50,75,100]),2)}  b {np.round(np.percentile(bb,[0,25,50,75,100]),2)}")
tt = np.array([d["t"] for d in chosen["L"]]); p(f"  L (하중 대비 고주파 H18, 충격>=1) 문턱 분위(0,25,50,75,100) {np.round(np.percentile(tt,[0,25,50,75,100]),3)}")
hr = hit / np.maximum(cnt[:, None], 1)
p("\n  시험 행에서 켜진 비율 (파일이 시험 폴드에 든 횟수 중). 고장 = 재현, 정상 = 오경보")
for j, k in enumerate(EXPK):
    p(f"   {k:>5}: 고장 재현 {hr[~nor, j].mean():.3f}  정상 오경보 {hr[nor, j].mean():.4f}  (정상 중 한 번이라도 켜진 파일 {int((hr[nor, j] > 0).sum())}: {' '.join(sn[TRI[i]] + ':%.2f' % hr[i, j] for i in np.flatnonzero(nor & (hr[:, j] > 0)))})")
p("\n  고장 30개 파일별 켜짐 비율 (ring spec J L | 합집합)")
for i in np.flatnonzero(~nor):
    p(f"   {sn[TRI[i]]:>4}: " + " ".join(f"{hr[i,j]:.2f}" for j in range(4)) + f" | {union_f[i]/cnt[i]:.2f}")
p(f"  합집합: 고장 재현 {union_f[~nor].sum()/cnt[~nor].sum():.3f}, 정상 오경보 {union_f[nor].sum()/cnt[nor].sum():.4f}; 정상 켜진 파일: " + " ".join(f"{sn[TRI[i]]}:{union_f[i]/cnt[i]:.2f}" for i in np.flatnonzero(nor & (union_f > 0))))
dfhr = pd.DataFrame(hr, columns=EXPK); dfhr.insert(0, "file", [names[i] for i in TRI]); dfhr.insert(1, "y", y); dfhr["union"] = union_f / cnt
dfhr.to_csv(os.path.join(here, "out", "R4b_fold_hit.csv"), index=False, encoding="utf-8-sig")

# ================= 2. 전체 train으로 고른 정의 (보고·그림용) =================
trall = np.arange(n)
SEL = select_all(trall); Ufull = u_all(SEL, trall)
p("\n[2] 전체 train 150으로 고른 정의 (보고용; 폴드 안 안정성은 위 표):")
p("   " + json.dumps({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()} for k, d in SEL.items()}, ensure_ascii=False))
for k in EXPK:
    on = Ufull[k] >= 1
    p(f"   {k:>5}: 고장 {on[TRI][~nor].sum():2d} ({' '.join(sn[TRI[i]] for i in np.flatnonzero(~nor) if on[TRI][i])}) 정상 {on[TRI][nor].sum()} val {on[VAL].sum()} ({' '.join(sn[i] for i in VAL if on[i])})")
A3 = np.max(np.column_stack([np.clip(Ufull[k], 0, None) for k in EXPK]), 1); on3 = A3 >= 1
p(f"   A3(max u): 고장 {on3[TRI][~nor].sum()} 정상 {on3[TRI][nor].sum()} val {on3[VAL].sum()} ({' '.join(sn[i] for i in VAL if on3[i])})")
p("   원래 규칙 대상 고장 A3 켜짐: " + " ".join(f"{s}:{'O' if on3[sn.index(s)] else '.'}" for s in ["A002", "A003", "A006", "A010", "A029", "A030", "A014", "A017", "A004", "A008", "A011", "A019", "A028", "A005", "A001", "A013", "A022"]))
p("   val 상위 u: " + "; ".join(f"{k}: " + " ".join(f'{sn[i]}={Ufull[k][i]:.2f}' for i in VAL[np.argsort(-Ufull[k][VAL])][:4]) for k in EXPK))
p("   val 이동 (u 중앙 정상/val, u>=0.5 비율 정상/val): " + "; ".join(f"{k}: {np.median(Ufull[k][TRI][nor]):.2f}/{np.median(Ufull[k][VAL]):.2f}, {np.mean(Ufull[k][TRI][nor]>=.5):.3f}/{np.mean(Ufull[k][VAL]>=.5):.3f}" for k in EXPK))
pd.DataFrame({"file": names, "split": split, **{f"u_{k}": Ufull[k] for k in EXPK}, "A3": A3}).to_csv(os.path.join(here, "out", "R4b_u.csv"), index=False, encoding="utf-8-sig")

# ================= 3. 세 번째 축: 로지스틱 CV (PR-AUC), 폴드 안 재선택 =================
FMj = joblib.load(os.path.join(root, "outputs", "final", "final_model.joblib")); Zoof = np.asarray(FMj.core_.oof_z_)
vz = pd.read_csv(os.path.join(here, "..", "code_D", "val_A_parts.csv"), encoding="utf-8-sig"); Zval = vz[["ae_z", "et_z"]].values
assert Zoof.shape == (n, 2) and len(Zval) == 40
Zall = np.vstack([Zoof, Zval])
p("\n[3] 로지스틱 CV (탐색용 PR-AUC, 5겹 x seed 0-9, 선택은 폴드 안). AE z · ET z = 최종 모델 OOF 고정 (A 구조의 두 축). 임계값 없음")
keyf = ["A002", "A003", "A006", "A010", "A029", "A030", "A014", "A017", "A004", "A022", "A005", "A001"]
def run_lr(feat_fn, seeds=range(10)):
    aps, aucs = [], []; rk = np.zeros(n)
    for seed in seeds:
        oof = np.zeros(n)
        for s2, tr, te, sel in [x for x in folds if x[0] == seed]:
            F = feat_fn(u_all(sel, tr))
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(F[TRI][tr], y[tr]); oof[te] = m.decision_function(F[TRI][te])
        aps.append(average_precision_score(y, oof)); aucs.append(roc_auc_score(y, oof))
        rk += np.array([(oof[nor] >= o).mean() if y[i] else np.nan for i, o in enumerate(oof)])
    return np.mean(aps), np.std(aps), np.mean(aucs), rk / len(list(seeds))
cl = lambda U, ks: np.max(np.column_stack([np.clip(U[k], 0, 3) for k in ks]), 1)
cfg = {
    "M0 [AE, ET]": lambda U: Zall,
    "M1 [AE, ET, A3=max(ring,spec,J,L)]": lambda U: np.column_stack([Zall, cl(U, EXPK)]),
    "M1b [AE, ET, A3=max(ring,J,L)]": lambda U: np.column_stack([Zall, cl(U, ("ring", "J", "L"))]),
    "M2 [AE, ET, u_ring,u_spec,u_J,u_L]": lambda U: np.column_stack([Zall] + [np.clip(U[k], 0, 3) for k in EXPK]),
}
for nm_, fn in cfg.items():
    ap, sd, au, rk = run_lr(fn)
    p(f"   {nm_:>36}: PR-AUC {ap:.3f} ± {sd:.3f}  ROC-AUC {au:.3f} | 고장보다 점수 높은 정상 비율: " + " ".join(f"{k}:{rk[names.index(k.replace('A', 'abn_'))]:.2f}" for k in keyf))

# ================= 4. val 경계 밀집 (전체 train 학습) =================
p("\n[4] val 경계 밀집: 전체 train으로 로지스틱 학습, 경계 = 학습 정상 점수 95% 분위, 폭 = 학습 정상 점수 sd의 0.25 · 0.5 · 1배 안에 든 val 수")
for nm_, fn in cfg.items():
    F = fn(Ufull); m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(F[TRI], y)
    s_tr = m.decision_function(F[TRI]); s_va = m.decision_function(F[VAL]); thr = np.percentile(s_tr[nor], 95); sd = s_tr[nor].std()
    p(f"   {nm_:>36}: 경계 {thr:.2f} sd {sd:.2f} | ±0.25sd {int((abs(s_va-thr)<.25*sd).sum())} ±0.5sd {int((abs(s_va-thr)<.5*sd).sum())} ±1sd {int((abs(s_va-thr)<sd).sum())} | val>=경계 {int((s_va>=thr).sum())} | 계수 {np.round(m.coef_[0],2)}")

# ================= 5. t-SNE 5-NN 순도 =================
Xtr, ytr, ntr = load_train(); Tt, Lt, fr = cached_feat2("train", Xtr)
FJ = json.load(open(os.path.join(root, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
pk = PeakRef(float(fr[1] - fr[0])).fit(Lt[nor]).transform(Lt)
Tt = Tt.copy(); Tt["H16_peak_h"], Tt["H17_peak_w"] = pk[:, 0], pk[:, 1]
BASE = Tt[list(FJ["final"])].values
EXT = cached_table("ext", feature_table_ext, Xtr).values


def zs(A, ref):
    Z = np.empty_like(A, float)
    for j in range(A.shape[1]): Z[:, j] = robust_z(A[ref, j])(A[:, j])
    return Z


def purity(Z, test_faults):
    idx = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z[test_faults], return_distance=False)[:, 1:]
    return float(y[idx].mean())


fa_idx = np.flatnonzero(~nor)
UF_all = np.column_stack([np.clip(Ufull[k][TRI], 0, 3) for k in EXPK])
p("\n[5] 고장 5-NN 순도 (고장 점의 가까운 이웃 5개 중 고장 비율, 무작위 0.20). base16 = 4-7 최종 16열")
p("   in-sample (전체 train으로 고른 정의):")
p(f"   {'ext150':>26}: {purity(zs(EXT, nor), fa_idx):.3f}")
p(f"   {'base16':>26}: {purity(zs(BASE, nor), fa_idx):.3f}")
for nm_, cols in [("base16 + u_ring", [0]), ("base16 + u_ring,u_spec", [0, 1]), ("base16 + u_ring,spec,J", [0, 1, 2]), ("base16 + u 4개", [0, 1, 2, 3])]:
    p(f"   {nm_:>26}: {purity(zs(np.hstack([BASE, UF_all[:, cols]]), nor), fa_idx):.3f}")
p(f"   {'base16 + A3(max u)':>26}: {purity(zs(np.hstack([BASE, UF_all.max(1)[:, None]]), nor), fa_idx):.3f}")
p("   폴드 안 재선택 (선택 = 학습 폴드, 순도 = 시험 폴드의 고장 점, 이웃 = 전체 150, 표준화 = 학습 폴드 정상; 100 폴드 평균)")
acc = {}
for seed, tr, te, sel in folds:
    U = u_all(sel, tr); Uc = np.column_stack([np.clip(U[k][TRI], 0, 3) for k in EXPK]); tf = te[~nor[te]]; ref = tr[nor[tr]]
    if len(tf) == 0: continue
    for nm_, M in [("ext150", EXT), ("base16", BASE), ("base16 + u_ring", np.hstack([BASE, Uc[:, [0]]])), ("base16 + u_ring,spec", np.hstack([BASE, Uc[:, [0, 1]]])),
                   ("base16 + u_ring,spec,J", np.hstack([BASE, Uc[:, [0, 1, 2]]])), ("base16 + u 4개", np.hstack([BASE, Uc])), ("base16 + A3(max u)", np.hstack([BASE, Uc.max(1)[:, None]]))]:
        acc.setdefault(nm_, []).append(purity(zs(M, ref), tf))
for k, v in acc.items(): p(f"   {k:>26}: 평균 {np.mean(v):.3f}  (폴드 간 sd {np.std(v):.3f})")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    from sklearn.manifold import TSNE
    for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if any(_f == x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = _f; break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.3))
    spaces = [("ext150", zs(EXT, nor)), ("base16", zs(BASE, nor)), ("base16 + u(ring,spec,J,L)", zs(np.hstack([BASE, UF_all]), nor))]
    mark = [sn.index(s) for s in ["A002", "A003", "A006", "A010", "A014", "A017", "A029"]]
    for a, (nm_, Z) in zip(ax, spaces):
        best = None
        for s in range(3):
            t = TSNE(2, perplexity=15, init="random", random_state=s, learning_rate="auto"); E = t.fit_transform(Z)
            if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, E)
        E = best[1]
        a.scatter(*E[nor].T, s=22, color="#2a78d6", alpha=.55, lw=0, label="정상"); a.scatter(*E[~nor].T, s=48, color="#eb6834", lw=0, label="고장")
        for i in mark:
            ii = list(TRI).index(i); a.annotate(sn[i], E[ii], xytext=(3, 3), textcoords="offset points", fontsize=8, color="#b33a10", weight="bold")
        a.set_title(f"{nm_}\n고장 5-NN 순도 {purity(Z, fa_idx):.3f} (무작위 0.20)", fontsize=10); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8)
    fig.suptitle("t-SNE (perplexity 15, 라벨은 색칠에만). 셋째 칸의 u는 전체 train으로 고른 정의라 순도는 in-sample", x=.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(here, "out", "R4b_tsne.png"), dpi=85); plt.close(fig)
    p("\n  t-SNE 그림: out/R4b_tsne.png")
except Exception as e:
    p("t-SNE 그림 실패:", repr(e))
open(os.path.join(here, "out", "R4b.txt"), "w", encoding="utf-8").write(buf.getvalue())
print("done")
