# %% [markdown]
# EXP-064 피처 정리 (11-25, 근거 4-18 J피처검증 · 파이프라인점검.md 3-3)
# - 기준(구성 0) = R2 + R3 검출기 (11-25 사용자 결정): j_tables(fmax=11000, atoms="50", res_new=True), events 없음.
#   J-축소 + K83, 그룹 CV(097 · 101 · 120 한 그룹), 이상치 059 · 065 · 116 학습 포함, full. 분할 = EXP-062 구성 3과 같음
# - 구성 1–5 = 기준에서 하나씩 바꾼 것 (쌍 비교 = seed별 F2 차이). 라벨을 보고 α · 문턱 · 대역을 다시 고르지 않는다
#   1 H2 → D_K1 (features2.d_k1: 두 번째 충격 높이 ÷ 검출기 입력 σ, H1′ 반복 타격)   2 − H16 · H17 (봉우리 열 없음)
#   3 − H13 (몫 2–4k)   4 − H7 (울림 1–2k, 축소는 남은 울림 열에만)   5 넷 다
# - 참고: EXP-062 구성 3(옛 검출기 = 250 Hz 사전 · 옛 RES)도 같은 분할로 다시 돌려 기준과의 차이를 적는다 (중간 단계 결과)
# - val은 라벨이 없다. 판정 수 · 바뀐 파일 · seed 안정성만 본다 (val로 고르지 않는다)
# 실행: QE_FULL=1 conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings, time
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL, best_f2_threshold
from src.exp2util import NS, NOR, ABN, tsne_default
from src.final import JShrink, j_tables, J_FIXED, J_PEAK
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행 (JShrink ET 300과 seed 5를 맞춘다)"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
FMAX = 11000
SESSION = ("nor_097", "nor_101", "nor_120")
ex = Experiment(__file__, "EXP-064", "피처 정리 (H2 → D_K1 · − H16 · H17 · − H13 · − H7)", ["H1", "H1′", "H2"],
                member="이동훈", model="J-축소 + K83 (src/final.py JShrink) · 11 kHz 상한 · R2 + R3 검출기",
                changes="기준 = R2 + R3(50 Hz 울림 사전 · RES 새 검출기). 피처 하나씩: H2 → D_K1 · − H16 · H17 · − H13 · − H7 · 넷 다. 이상치 학습 포함, 그룹 CV, full",
                key_changes=["기준 검출기 = R2 + R3 (j_tables atoms=\"50\", res_new=True). EXP-062 구성 3 대비 바뀜",
                             "1: H2(2위/1위) → D_K1(두 번째 충격 높이 ÷ σ, JShrink cols)", "2: 봉우리 H16 · H17 뺌 (JShrink peak=[])",
                             "3: H13 몫 2–4k 뺌", "4: H7 울림 1–2k 뺌 (축소는 남은 울림 열)", "5: 넷 다",
                             "이름 변경(값 같음): H18_hfl → L2_hfl, L1_S → L1_lock"])
ns = NS(); y, names = ns.y, np.array(ns.names); n = ns.n; nv = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
OUT = outlier_idx(names)

D1 = [("D_K1" if c == "H2_ratio21" else c) for c in J_FIXED]
drop = lambda cols, *cs: [c for c in cols if c not in cs]
ALL = drop(D1, "H13_share_2-4k", "H7_ring_1-2k")
# 구성: (이름, JShrink 옵션)
CFG = [("0 기준 (R2 + R3)", {}), ("1 H2 → D_K1", dict(cols=D1)), ("2 − H16 · H17", dict(peak=[])),
       ("3 − H13", dict(cols=drop(J_FIXED, "H13_share_2-4k"))), ("4 − H7", dict(cols=drop(J_FIXED, "H7_ring_1-2k"))),
       ("5 넷 다", dict(cols=ALL, peak=[]))]
NM = [c[0] for c in CFG]; KW = dict(CFG); B = NM[0]; M5 = NM[5]
REF = "참고 EXP-062 구성 3 (옛 검출기)"


def jt(tag, X, **kw):
    """j_tables 캐시 outputs/cache/jtab_{tag}.npz (새 태그, 기존 캐시는 건드리지 않음)"""
    p = os.path.join(CACHE, f"jtab_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}
    T = j_tables(X, FMAX, **kw); np.savez(p, **T); return T


t0 = time.time()
TB = (jt("train_11k_064", ns.X, atoms="50", res_new=True), jt("val_11k_064", ns.Xv, atoms="50", res_new=True))
TR = (jt("train_11k", ns.X), jt("val_11k", ns.Xv))                     # EXP-062 구성 3 표
print(f"표 {time.time() - t0:.0f}s", flush=True)
# 새 표(이름 변경 · DK1 추가 후 코드) = EXP-063 캐시와 같은가: R2 표(063a50)의 H · S · L · LP, R3 표(063rn)의 RES
a50 = np.load(os.path.join(CACHE, "jtab_train_11k_063a50.npz")); rn = np.load(os.path.join(CACHE, "jtab_train_11k_063rn.npz"))
same = {k: bool(np.allclose(TB[0][k], a50[k])) for k in ("H", "S", "L", "LP", "P83", "W83", "K83")}
same["RES (= 063rn)"] = bool(np.allclose(TB[0]["RES"], rn["RES"]))
ex.note("기준 표 = EXP-063 캐시", same); print("기준 표 = EXP-063 캐시:", same, flush=True)
g_sess = np.arange(n); g_sess[np.isin(names, SESSION)] = list(names).index(SESSION[0])


def sub(T, i):
    return {k: (v if np.ndim(v) == 0 else v[i]) for k, v in T.items()}


class JIdx:
    """nested_cv용 껍데기: 입력 행 = [파일 번호]. JShrink(kw)를 그 파일들의 표로 학습, 점수 = 최종 판정(결합 OR E2 OR 8.3k, 0/1)"""
    def __init__(self, T, kw): self.T, self.kw = T, kw
    def fit(self, F, y): self.m_ = JShrink(**self.kw).fit(sub(self.T, np.asarray(F)[:, 0].astype(int)), y); return self
    def decision_function(self, F): return self.m_.parts(sub(self.T, np.asarray(F)[:, 0].astype(int)))["pred"].values.astype(float)


# %% 중첩 CV (full, 모든 구성 같은 분할 = EXP-062 구성 3)
RES = {}
for nm in NM:
    t0 = time.time()
    RES[nm] = nested_cv(np.arange(n)[:, None].astype(float), y, g_sess, lambda kw=KW[nm]: JIdx(TB[0], kw), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=nm == B)
    print(f"  {time.time() - t0:.0f}s", flush=True)
RES[REF] = nested_cv(np.arange(n)[:, None].astype(float), y, g_sess, lambda: JIdx(TR[0], {}), n_jobs=-1)
ex.record(REF, RES[REF], names, mid=True)
pd_ = lambda a, b: dict(diff=round(float((RES[a]["f2"] - RES[b]["f2"]).mean()), 3), diff_sd=round(float((RES[a]["f2"] - RES[b]["f2"]).std()), 3),
                        seed_diffs=np.round(RES[a]["f2"] - RES[b]["f2"], 3).tolist(), base_sd=round(float(RES[b]["f2"].std()), 3),
                        exceeds_sd=bool(abs((RES[a]["f2"] - RES[b]["f2"]).mean()) > RES[b]["f2"].std()))
pair = {nm: pd_(nm, B) for nm in NM[1:]}
pair["0 기준 − EXP-062 구성 3"] = pd_(B, REF)
ex.note("쌍 비교 (구성 − 기준, seed별)", pair); print(pair, flush=True)

# %% 전체 학습 (train 150) × seed 5 → 규칙 켜짐 · val 판정 · val 안정성
SEEDS = range(5)


def fit_one(T, Tv, kw, s):
    m = JShrink(seed=s, **kw).fit(T, y); return m, m.parts(T), m.parts(Tv)


jobs = [(nm, s) for nm in NM for s in SEEDS] + [(REF, 0)]
out = Parallel(n_jobs=-1)(delayed(fit_one)(*(TR if nm == REF else TB), ({} if nm == REF else KW[nm]), s) for nm, s in jobs)
ALLF = dict(zip(jobs, out))
FIT = {}
for nm in NM + [REF]:
    m, Pt, Pv = ALLF[(nm, 0)]
    comb_oof = m.c_.decision_function(m.Z_) >= m.tc_                                   # 학습 파일의 결합 판정은 안쪽 OOF 입력으로
    FIT[nm] = dict(m=m, Pt=Pt, Pv=Pv, comb=comb_oof, fin=comb_oof | Pt.E2.values | Pt.rule83.values, vfin=Pv.pred.values.astype(bool))
# val 안정성: seed 5개 전체 학습 모델의 val 판정 · 여유 · ET logit
STAB, VP, VM, VE = {}, {}, {}, {}
for nm in NM:
    VP[nm] = np.array([ALLF[(nm, s)][2].pred.values for s in SEEDS]); VM[nm] = np.array([ALLF[(nm, s)][2].margin.values for s in SEEDS])
    VE[nm] = np.array([ALLF[(nm, s)][2].et_logit.values for s in SEEDS])
for nm in NM:
    flip = (VP[nm].min(0) != VP[nm].max(0)); de = VE[nm].mean(0) - VE[B].mean(0)
    STAB[nm] = {"seed마다 판정이 바뀌는 val 수": int(flip.sum()), "바뀌는 val": " ".join(short(nv[i]) for i in np.flatnonzero(flip)) or "—",
                "val |여유| 중앙값 (seed 평균 여유)": round(float(np.median(np.abs(VM[nm].mean(0)))), 3),
                "val 고장 판정 (seed 0 · 과반)": f"{int(VP[nm][0].sum())} · {int((VP[nm].mean(0) > .5).sum())}",
                "val ET logit 평균": round(float(VE[nm].mean()), 2), "train 정상 ET OOF logit 중앙값": round(float(np.median(FIT[nm]['m'].Z_[y == 0, 0])), 2),
                "기준 대비 ET logit |Δ| > 1 val 수": int((np.abs(de) > 1).sum()) if nm != B else "—",
                "|Δ| > 1 파일 (Δ)": " ".join(f"{short(nv[i])}({de[i]:+.1f})" for i in np.flatnonzero(np.abs(de) > 1)) if nm != B else "—"}
    print(nm, STAB[nm], flush=True)
ex.note("val 안정성 (seed 5 전체 학습)", STAB)
fires = {}
lst = lambda P, nn, col: " ".join(short(nn[i]) for i in np.flatnonzero(P[col].values)) or "—"
for nm, F in FIT.items():
    Pt, Pv = F["Pt"], F["Pv"]
    cnt2 = lambda col: f"{int((Pt[col] & (y == 1)).sum())}/{int((Pt[col] & (y == 0)).sum())}"
    fires[nm] = {"train E2 (고장/정상)": cnt2("E2"), "train E2 파일": lst(Pt, names, "E2"),
                 "train 8.3k 스펙트럼 (고장/정상)": cnt2("spec83"), "train RES (고장/정상)": cnt2("res83"), "train RES 파일": lst(Pt, names, "res83"),
                 "train 8.3k 합 파일": lst(Pt, names, "rule83"),
                 "val E2": lst(Pv, np.array(nv), "E2"), "val 8.3k": lst(Pv, np.array(nv), "rule83"), "val RES 최대": round(float(Pv.RES.max()), 3),
                 "val 고장 판정": f"{int(F['vfin'].sum())}/40", "결합 계수 (ET · 박자 · AE)": np.round(F["m"].c_.coef_[0], 3).tolist(),
                 "결합 문턱": round(F["m"].tc_, 3), "E2 문턱": round(F["m"].t18_, 3)}
ex.note("규칙 켜짐 · 결합 계수 (train 150 전체 학습, seed 0)", fires)
ex.save_md("규칙_켜짐", "train 150 전체 학습 모델 (seed 0). 8.3k = 스펙트럼(P83 · W83 · K83) OR 공진(RES ≥ 0.7). E2 = L2 위 꼬리 규칙(문턱 = 학습 정상 L2_hfl 99 %)\n\n"
           + md_table(pd.DataFrame(fires).T.reset_index().rename(columns={"index": "구성"}))
           + "\n\n## val 안정성 (seed 0–4로 각각 전체 학습 → val)\n\n여유 = 결합 점수 − 문턱. Δ = 기준 대비 val ET logit(seed 평균) 변화\n\n"
           + md_table(pd.DataFrame(STAB).T.reset_index().rename(columns={"index": "구성"})))

# %% 오류 분석 표 (가로): 파일별 고장 판정 횟수 (seed 5번 중)
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in NM + [REF]}
files = [names[i] for i in range(n) if any((C[nm][i] > 0) != (y[i] == 1) or (y[i] == 1 and C[nm][i] < 5) for nm in C)]
ix = lambda f: list(names).index(f)
extra = {f"{nm} (5번 중 고장)": {f: int(C[nm][ix(f)]) for f in files} for nm in NM[:5] + [REF]}
extra["5 · E2 / 8.3k (전체 학습)"] = {f: ("E2 " if FIT[M5]["Pt"].E2.values[ix(f)] else "") + ("8.3k" if FIT[M5]["Pt"].rule83.values[ix(f)] else "") or "—" for f in files}
ex.save_md("오류분석_표", f"모델 판단 행 = {M5}. 아래 행 = 다른 구성의 고장 판정 횟수. 열 = 어느 구성에서든 한 번이라도 틀린 파일\n\n"
           + error_md_table(names, y, C[M5], files, n_seeds=5, extra=extra))

# %% val 판정 (라벨 없음, 제출 아님). seed 0 전체 학습
k_ = lambda nm: nm.split()[0] if nm != REF else "ref"
NV = NM + [REF]
V = pd.DataFrame({"파일": [v[-3:] for v in nv], **{k_(nm): np.where(FIT[nm]["vfin"], "고장", "정상") for nm in NV},
                  **{f"{k_(nm)} 여유": np.round(FIT[nm]["Pv"].margin.values, 2) for nm in NV},
                  **{f"{k_(nm)} seed 고장 수": VP[nm].sum(0) for nm in NM},
                  **{f"{k_(nm)} 규칙": [("E2 " if e else "") + ("8.3k" if r else "") for e, r in zip(FIT[nm]["Pv"].E2, FIT[nm]["Pv"].rule83)] for nm in NM}})
d = V[V[[k_(nm) for nm in NV]].nunique(axis=1) > 1]
vchg = {nm: " ".join(short(nv[i]) + ("+" if FIT[nm]["vfin"][i] else "−") for i in np.flatnonzero(FIT[nm]["vfin"] != FIT[B]["vfin"])) or "없음" for nm in NM[1:] + [REF]}
ex.note("val 판정 수 (seed 0)", {nm: int(FIT[nm]["vfin"].sum()) for nm in NV}); ex.note("val 기준 대비 바뀐 파일 (+ 고장으로 · − 정상으로)", vchg)
ex.save_md("val_판정", "val 고장 판정 수 (seed 0 전체 학습): " + " · ".join(f"{k_(nm)} {int(FIT[nm]['vfin'].sum())}/40" for nm in NV)
           + "\n\n기준(0) 대비 바뀐 파일 (+ 고장으로 · − 정상으로): " + " · ".join(f"{k_(nm)} {v}" for nm, v in vchg.items())
           + "\n\n여유 = 결합 점수 − 문턱. seed 고장 수 = seed 0–4 전체 학습 중 고장 판정 횟수. 규칙 = E2 · 8.3k 켜짐. ref = EXP-062 구성 3\n\n구성 사이에 판정이 다른 파일\n\n"
           + (md_table(d) if len(d) else "없음") + "\n\n전체\n\n" + md_table(V))

# %% 핵심 그림: logistic_plane 기준(0) vs 넷 다(5) (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정). 규칙만으로 잡힌 고장은 뺌
def plane(ax, nm, val=False):
    F = FIT[nm]; m = F["m"]; c, b, tc = m.c_.coef_[0], m.c_.intercept_[0], m.tc_; Z = m.Z_; t_et = float(best_f2_threshold(y, Z[:, 0]))
    if val: P = F["Pv"]; et, zb, y3, fin = P.et_logit.values, P.beat_z.values, P.ae_z.values, F["vfin"]
    else: et, zb, y3, fin = Z[:, 0], Z[:, 1], Z[:, 2], F["fin"]
    x = et + c[1] / c[0] * zb
    x0, x1 = min(x.min(), t_et) - .6, max(x.max(), t_et) + .6; y0, y1 = y3.min() - .6, y3.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    ax.contourf(gx, gy, (c[0] * gx + c[2] * gy + b >= tc).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    show = np.ones(len(x), bool)
    if not val: show &= ~((y == 1) & ~F["comb"] & fin)
    for i in np.flatnonzero(show & (np.abs(x - et) >= .3)): ax.plot([et[i], x[i]], [y3[i], y3[i]], ls=":", color="#888", lw=.8, zorder=1)
    col = np.where(fin, ABN, NOR); ref = FIT[B]
    if val:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in np.flatnonzero(fin | (fin != ref["vfin"])):
            ax.annotate(short(nv[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold" if fin[i] != ref["vfin"][i] else "normal")
    else:
        o = np.zeros(len(x), bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & (fin != (y == 1))):
            ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
P2 = [B, M5]
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 틀린 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준(0, R2 + R3) vs 넷 다(5)의 결합 경계", key=True)
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습, seed 0). 굵은 글씨 = 기준(0)과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "기준 vs 넷 다의 val 판정")

# val ET logit 이동: 기준 vs 구성 2 · 5 (seed 평균). 대각선에서 멀수록 봉우리 열이 val ET를 움직이던 파일
fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, nm in zip(axs, (NM[2], M5)):
    a, b_ = VE[B].mean(0), VE[nm].mean(0); lo, hi = min(a.min(), b_.min()) - .5, max(a.max(), b_.max()) + .5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1); ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#eee", zorder=0)
    ax.scatter(a, b_, marker="^", s=50, c=np.where(VP[nm].mean(0) > .5, ABN, NOR), edgecolor="k", lw=.5)
    for i in np.flatnonzero(np.abs(b_ - a) > 1): ax.annotate(short(nv[i]), (a[i], b_[i]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("기준(0) val ET logit (seed 평균)", color=MUTED); ax.set_ylabel(f"{nm} val ET logit", color=MUTED); ax.set_title(f"{nm}: |Δ| > 1 = {int((np.abs(b_ - a) > 1).sum())}개 (회색 띠 = ±1)", fontsize=10, loc="left")
fig.tight_layout(); ex.save_fig(fig, "val_ET이동", "봉우리 열을 뺐을 때 val ET logit 이동 (색 = seed 과반 판정)")

# %% t-SNE 기본 (핵심 그림, 기본 공간 = 기준 피처). 빨간 원 = 구성 5 오경보 (5번 중 3회+)
fp5 = np.flatnonzero((y == 0) & (RES[M5]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fp5, mark_label="구성 5 오경보 (5번 중 3회+)", title="EXP-064 (기준 피처 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10, 기준 피처). 빨간 원 = 구성 5 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
