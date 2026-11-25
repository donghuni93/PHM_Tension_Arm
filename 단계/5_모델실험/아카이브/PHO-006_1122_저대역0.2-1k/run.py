# %% [markdown]
# PHO-006 (휴대폰 세션, 11-22) 저대역 0.2–1 kHz 피처 2열을 J의 ET 입력에 추가 (팀원 피처 검토: 효섭 SK_200_3k · 나영 '조용한 구간 저주파')
# - 가설: H1(저대역에도 충격성이 남는다 → 스펙트럼 첨도↑), H2(부싱 경화 → 충격 없는 구간의 노면 전달 스펙트럼 모양이 바뀐다 → 0.2–1k 몫 변화)
# - 피처 정의 (라벨 보기 전 고정)
#   SK02_1k : 평균 제거 → STFT(512점 Hann, 홉 256) → 200–1000 Hz 칸마다 스펙트럼 첨도 SK(f) = E|X|⁴ / (E|X|²)² − 2 → 칸 평균
#   Q02_1k  : 새 충격 정의(src.impacts.detect_impacts)로 찾은 충격 ±25 ms를 덮은 프레임을 빼고(조용한 구간), 남은 STFT 프레임의 평균 파워로
#             log10( P(200–1000 Hz) / P(200–12500 Hz) ). 비율이라 게인에 둔감
#   0.2–1 kHz는 H 블록 1 kHz 고역(결정.md) 아래라 지금 J의 ET가 보지 않는 대역. 백색화(AR 잔차)는 사용자 판단(오탐 키울 우려)으로 제외
# - 점검: 단독 AUC · 길 끌림 ρ(정상의 RMS와 스피어만) · 관심 파일 위치(정상 중 %) · val 이동(정상 5–95% 밖 비율, 기대 10%)
# - 비교: quick_cv 기준 W(ET H15 + 박자 → 로지스틱, PHO-001과 같음) + 하나씩 / 둘 다 → 중첩 CV J vs J + 열
# 실행: conda run -n qe-phm python "단계/5_모델실험/PHO-006_1122_저대역0.2-1k/run.py"

# %% 설정
import os, sys, json, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.models2 import Two, TwoAE
from src.evaluate import quick_cv, separation, best_f2_threshold
from src.impacts import detect_impacts
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from scipy.signal import stft
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000
QUICK_ONLY = os.environ.get("QUICK_ONLY") == "1"
ex = Experiment(__file__, "PHO-006", "저대역 0.2–1 kHz 피처 2열 (팀원 피처 검토)", ["H1", "H2"],
                member="이동훈", model="J + ET 입력에 SK02_1k · Q02_1k",
                changes="J(EXP-044)의 ET 입력에 0.2–1 kHz 스펙트럼 첨도 · 조용한 구간 0.2–1k PSD 몫 추가 (효섭 SK_200_3k · 나영 조용한 구간 저주파)",
                key_changes=["ET 입력에 SK02_1k(0.2–1 kHz 스펙트럼 첨도) 추가", "ET 입력에 Q02_1k(충격 없는 구간의 0.2–1k PSD 몫) 추가", "AE · 박자 · E2 · 8.3k 규칙 · 문턱은 J와 같음"])
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0
LOG = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOG.append(s)

# %% 피처 (라벨 없음, 캐시)
NPER, HOP = 512, 256
def feats(x):
    x = np.asarray(x, float); x = x - x.mean()
    f, t, Z = stft(x, fs=FS, nperseg=NPER, noverlap=NPER - HOP, boundary=None, padded=False)
    P = np.abs(Z) ** 2; lo = (f >= 200) & (f <= 1000); all_ = (f >= 200)
    sk = float(np.mean(np.mean(P[lo] ** 2, 1) / np.mean(P[lo], 1) ** 2 - 2))
    idx = detect_impacts(x)["idx"]; w = int(0.025 * FS)
    c = (t * FS).astype(int); half = NPER // 2
    quiet = np.ones(len(t), bool)
    for p in idx:
        quiet &= ~((c + half >= p - w) & (c - half <= p + w))
    Pq = P[:, quiet] if quiet.sum() >= 10 else P
    m = Pq.mean(1)
    return sk, float(np.log10(m[lo].sum() / m[all_].sum())), int(quiet.sum()), len(idx)
def table(tag, X):
    p = os.path.join(CACHE, f"pho006_{tag}.npy")
    if os.path.exists(p): return np.load(p)
    A = np.array(Parallel(n_jobs=-1)(delayed(feats)(x) for x in X)); np.save(p, A); return A
t0 = time.time(); A, Av = table("train", ns.X), table("val", ns.Xv)
NEW = ["SK02_1k", "Q02_1k"]; Fn, Fv = A[:, :2], Av[:, :2]
say(f"피처 계산 {time.time() - t0:.0f}s · 조용한 프레임 중앙 {np.median(A[:, 2]):.0f}/{int(1 + (50000 - NPER) // HOP)} (최소 {A[:, 2].min():.0f})")

# %% 점검: 단독 AUC · 길 끌림 · 관심 파일 · val 이동
rms = np.sqrt(((ns.X - ns.X.mean(1, keepdims=True)) ** 2).mean(1))
FOC = ["abn_002", "abn_017", "abn_022", "nor_116", "nor_065", "nor_059"]
pct = lambda v, i: (v[nor] < v[i]).mean() * 100
chk = {}
say("\n피처 | AUC | 길 끌림 ρ(정상 RMS) | val 5–95% 밖 | " + " · ".join(short(f) for f in FOC) + " (정상 중 %)")
for j, c in enumerate(NEW):
    v, vv = Fn[:, j], Fv[:, j]; q5, q95 = np.percentile(v[nor], [5, 95])
    auc = roc_auc_score(y, v); rho = spearmanr(v[nor], rms[nor])[0]; out = float(((vv < q5) | (vv > q95)).mean())
    chk[c] = dict(auc=round(auc, 3), rho=round(float(rho), 3), val_out=round(out, 3), val_hi=round(float((vv > q95).mean()), 3), val_lo=round(float((vv < q5).mean()), 3),
                  files={f: round(pct(v, names.index(f)), 0) for f in FOC})
    say(f"{c} | {auc:.3f} | {rho:+.2f} | {out:.0%} (위 {chk[c]['val_hi']:.0%} · 아래 {chk[c]['val_lo']:.0%}) | " + " · ".join(f"{chk[c]['files'][f]:.0f}" for f in FOC))
say(f"참고 ρ(SK, Q) 전체 = {spearmanr(Fn[:, 0], Fn[:, 1])[0]:+.2f}")
ex.note("피처 점검", chk)

# %% 그림: 정상 5–95% 띠 위에 고장 · val · 관심 파일
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
rng = np.random.default_rng(0)
for ax, j, c in zip(axs, range(2), NEW):
    v, vv = Fn[:, j], Fv[:, j]; q5, q95 = np.percentile(v[nor], [5, 95])
    ax.axhspan(q5, q95, color="#d9e6f5", alpha=.7, label="train 정상 5–95%")
    for k, (vals, col, lab, mk_) in enumerate(((v[nor], NOR, "train 정상", "s"), (v[~nor], ABN, "train 고장", "o"), (vv, "#555555", "val", "^"))):
        ax.scatter(k + rng.uniform(-.18, .18, len(vals)), vals, s=22, color=col, marker=mk_, alpha=.7, lw=0, label=lab)
    for f in FOC:
        i = names.index(f); k = 0 if y[i] == 0 else 1
        ax.annotate(short(f), (k, v[i]), xytext=(14, 0), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=.6))
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["정상", "고장", "val"]); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"{c} · AUC {chk[c]['auc']:.2f} · 길 끌림 ρ {chk[c]['rho']:+.2f} · val 띠 밖 {chk[c]['val_out']:.0%}", fontsize=10)
axs[0].set_ylabel("SK (0.2–1 kHz 칸 평균)"); axs[1].set_ylabel("log10 P(0.2–1k) / P(0.2–12.5k), 조용한 구간"); axs[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); ex.save_fig(fig, "피처_분포", "새 2열의 정상 5–95% 띠와 고장 · val 분포. 관심 파일 = 놓치기 쉬운 고장 002 017 022, J 오경보 116 065 059")

# %% quick_cv: 기준 W + 하나씩 / 둘 다
H = ns.T[ns.FIXED].values; S = ns.T[["L1_S"]].values; L = ns.L
mkW = lambda fx: (lambda: Two("lin", (), fixed=tuple(fx), nf=len(fx), df=ns.df, peak=tuple(ns.PEAK)))
Q = {}
def qrun(tag, cols):
    F = np.hstack([H, Fn[:, cols], S, L]); q = quick_cv(F, y, g, mkW(ns.FIXED + [NEW[c] for c in cols])); sp = separation(q["oof"], y, names)
    Q[tag] = dict(pr_auc=round(float(q["pr_auc"].mean()), 3), pr_sd=round(float(q["pr_auc"].std()), 3), risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"],
                  rank={short(names[i]): r for i, r in q["miss_rank"].items() if r > 0.02})
    say(f"{tag:14s} PR-AUC {Q[tag]['pr_auc']:.3f} ± {Q[tag]['pr_sd']:.3f} · 위험 정상 {sp['risk_n']:.1f} · 확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 순위>2%: {Q[tag]['rank']}")
say("\n=== quick_cv (5-fold × seed 3, PR-AUC, 기준 W) ===")
qrun("기준 W", []); qrun("W + SK02_1k", [0]); qrun("W + Q02_1k", [1]); qrun("W + 둘 다", [0, 1])
ex.note("quick_cv", Q)
if QUICK_ONLY:
    open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(LOG)); sys.exit(0)

# %% 중첩 CV: J vs J + 열 (J 값은 EXP-044 기록을 같은 분할로 다시 돌린 값)
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
idx = np.arange(ns.n, dtype=float)
def Mx(cols):  return np.column_stack([H, Fn[:, cols], S, L, idx, r83])
def mkJ(cols): fx = ns.FIXED + [NEW[c] for c in cols]; return lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(fx), nf=len(fx), df=ns.df, peak=tuple(ns.PEAK)))
CFG = [("J + 둘 다", [0, 1], True), ("J (EXP-044 재실행)", [], False), ("J + SK02_1k", [0], False), ("J + Q02_1k", [1], False)]
for nm, cols, main in CFG:
    ex.record("중첩: " + nm, nested_cv(Mx(cols), y, g, mkJ(cols), n_jobs=-1), names, main=main, mid=not main)

# %% 핵심 그림: logistic_plane (J vs J + 둘 다, OOF seed 0)
def oof_fires(cols, seed=0):
    o = dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
    M = Mx(cols)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
        f = mkJ(cols)().fit(M[tr], y[tr]).fires(M[te])
        for k in o: o[k][te] = f[k]
    return o
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
for ax, (nm, cols) in zip(axs, (("J", []), ("J + 둘 다", [0, 1]))):
    fo = oof_fires(cols); a, aez = fo["a"], fo["ae"]; caught = fo["main"] | fo["E2"] | fo["R83"]
    m = mkJ(cols)().fit(Mx(cols), y).m_
    x0, x1, y0, y1 = a.min() - .3, a.max() + .3, aez.min() - .3, aez.max() + .3
    xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250)); D = m._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
    ax.contour(xx, yy, D, levels=[m.tc_], colors="k", linewidths=1.3, linestyles="--")
    rule = fo["E2"] | fo["R83"]
    ax.scatter(a[nor & ~caught], aez[nor & ~caught], color=NOR, s=30, marker="s", lw=0, alpha=.7, label="정상 → 정상")
    ax.scatter(a[nor & caught], aez[nor & caught], color=NOR, s=95, marker="s", edgecolor="#d62728", lw=2, label=f"정상 → 오경보 ({int((nor & caught).sum())})")
    ax.scatter(a[~nor & ~rule & caught], aez[~nor & ~rule & caught], color=ABN, s=60, lw=0, label="고장 → 결합이 잡음")
    ax.scatter(a[~nor & ~caught], aez[~nor & ~caught], color=ABN, s=120, marker="X", edgecolor="k", lw=1, label=f"고장 → 놓침 ({int((~nor & ~caught).sum())})")
    for f_ in FOC:
        i = names.index(f_)
        if not (y[i] and rule[i]): ax.annotate(short(f_), (a[i], aez[i]), xytext=(7, 7), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
    ax.set_xlim(x0, x1); ax.set_xlabel("ET 점수 (logit, OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
    r = ex.res["configs"]["중첩: " + ("J (EXP-044 재실행)" if nm == "J" else nm)]
    ax.set_title(f"{nm} — 중첩 F2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · 오경보율 {r['fpr']:.3f} (규칙으로 잡힌 고장 {int((~nor & rule).sum())}개는 뺌)", fontsize=10)
axs[0].set_ylabel("AE z (OOF)"); fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "ET × AE 평면 (OOF seed 0, 박자 z = 0 단면 경계). 왼쪽 J, 오른쪽 J + 저대역 2열", key=True)

# %% 오류 분석 표 + val 판정 (대표 구성, 제출 파일 아님)
R = ex.res["configs"]["중첩: J + 둘 다"]; R0 = ex.res["configs"]["중첩: J (EXP-044 재실행)"]
cnt = np.array([(5 - R["miss_counts"].get(n_, 0)) if y[i] else R["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])
files = sorted(set(["abn_002", "abn_017", "abn_022"] + [n_ for n_ in names if n_.startswith("nor") and (R["fp_counts"].get(n_, 0) or R0["fp_counts"].get(n_, 0))]))
ex_ = lambda j: {n_: f"{Fn[names.index(n_), j]:.3f} ({pct(Fn[:, j], names.index(n_)):.0f}%)" for n_ in files}
ex.save_md("오류분석_표", "# PHO-006 오류 분석 (J + 둘 다, 중첩 CV seed 5개 판정 횟수)\n\n괄호 = 정상 중 백분위. J 오경보 횟수: "
           + " · ".join(f"{short(k)} {v}" for k, v in R0["fp_counts"].items()) + "\n\n"
           + error_md_table(names, y, cnt, files, extra={"SK02_1k": ex_(0), "Q02_1k": ex_(1)}) + "\n")
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); aev = _ae_score(ae_tr, LPv)
idv = np.zeros(len(ns.Xv)); Sv = ns.Tv[["L1_S"]].values; Hv = ns.Tv[ns.FIXED].values
pv = {}
for nm, cols in (("J", []), ("J + 둘 다", [0, 1])):
    mJ = mkJ(cols)().fit(Mx(cols), y); mJ.m_.ae_override_ = aev
    pv[nm] = mJ.decision_function(np.column_stack([Hv, Fv[:, cols], Sv, ns.Lv, idv, r83v])) >= .5
diff = [ns.nv[i][-3:] for i in np.flatnonzero(pv["J"] != pv["J + 둘 다"])]
say(f"\nval 고장 판정: J {int(pv['J'].sum())}/40 · J + 둘 다 {int(pv['J + 둘 다'].sum())}/40 · 갈린 파일 {diff}")
V = pd.DataFrame(dict(파일=[n_[-3:] for n_ in ns.nv], J=np.where(pv["J"], "고장", "정상"), J_둘다=np.where(pv["J + 둘 다"], "고장", "정상"),
                      SK02_1k=Fv[:, 0].round(3), Q02_1k=Fv[:, 1].round(3)))
ex.save_md("val_판정", f"# PHO-006 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\nJ {int(pv['J'].sum())}/40 · J + 둘 다 {int(pv['J + 둘 다'].sum())}/40 · 갈린 파일: {' '.join(diff) or '없음'}\n\n{md_table(V)}\n")
ex.note("val", dict(J=int(pv["J"].sum()), J_both=int(pv["J + 둘 다"].sum()), diff=diff))
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
