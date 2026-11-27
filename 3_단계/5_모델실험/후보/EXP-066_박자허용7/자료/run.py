# %% [markdown]
# EXP-066 박자(L1_lock) hit 허용 ±3 % → ±7 % (11-25)
# - 가설: H3(하중 주기에 묶인 반복 타격). abn_028은 하중 주기 근처에서 반복해 치지만 ±3 % 창 밖이라 hit 0.23
# - 기준(구성 0) = EXP-062 구성 3 그대로: J-축소 + K83, j_tables(fmax=11000) 기본 검출기, 그룹 CV(097 · 101 · 120 한 그룹),
#   이상치 059 · 065 · 116 학습 포함, full. 분할 = EXP-062 구성 3과 같음
# - 구성 1 = lock_S tol 0.07 (j_tables lock_tol=0.07). S(박자) 한 열만 바뀐다. 나머지(ET 열 · 축소 · 규칙)는 그대로
# - [사후 값] ±7 %는 train 라벨(028 · 014 · 022 여유)을 본 뒤 사용자가 고른 값(11-25). 정상 T 추정 불확실성은 ~1 %라 근거가 아니다
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
from src.final import JShrink, j_tables
from src.features2 import lock_S
from src.impacts import all_impacts
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행 (JShrink ET 300과 seed 5를 맞춘다)"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
FMAX, TOL = 11000, 0.07
SESSION = ("nor_097", "nor_101", "nor_120")
ex = Experiment(__file__, "EXP-066", "박자(L1_lock) hit 허용 ±3 % → ±7 % [사후 값]", ["H3"],
                member="이동훈", model="J-축소 + K83 (src/final.py JShrink) · 11 kHz 상한 (EXP-062 구성 3)",
                changes="기준 = EXP-062 구성 3. L1_lock hit 허용 ±3 % → ±7 % (lock_S tol, [사후 값]: train 라벨을 본 뒤 고른 값). 이상치 학습 포함, 그룹 CV, full",
                key_changes=["기준 = EXP-062 구성 3, 같은 분할", "1: L1_lock hit 허용 ±7 % (j_tables lock_tol=0.07) — 박자 S 한 열만 바뀜 [사후 값]",
                             "ET 열 · 울림 축소 · E2 · 8.3k 규칙은 그대로"])
ns = NS(); y, names = ns.y, np.array(ns.names); n = ns.n; nv = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
OUT = outlier_idx(names)
ix = lambda f: list(names).index(f)

NM = ["0 기준 (EXP-062 구성 3)", "1 박자 허용 ±7 % [사후 값]"]; B, M1 = NM
WATCH = ["abn_028", "abn_014", "abn_022", "abn_001", "abn_021", "nor_048", "nor_103", "nor_091", "nor_059", "nor_116", "nor_065"]


def jt(tag, X, **kw):
    """j_tables 캐시 4_outputs/cache/jtab_{tag}.npz (새 태그, 기존 캐시는 건드리지 않음)"""
    p = os.path.join(CACHE, f"jtab_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}
    T = j_tables(X, FMAX, **kw); np.savez(p, **T); return T


def load(tag):
    z = np.load(os.path.join(CACHE, f"jtab_{tag}.npz")); return {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}


t0 = time.time()
TB = {B: (load("train_11k"), load("val_11k")),                                    # EXP-062 구성 3 캐시 그대로 (읽기만)
      M1: (jt("train_11k_tol07", ns.X, lock_tol=TOL), jt("val_11k_tol07", ns.Xv, lock_tol=TOL))}
print(f"표 {time.time() - t0:.0f}s", flush=True)
# 새 표: S 말고는 기준 캐시와 같은가, S는 얼마나 바뀌었나
same = {k: bool(np.allclose(TB[M1][0][k], TB[B][0][k]) and np.allclose(TB[M1][1][k], TB[B][1][k]))
        for k in TB[B][0] if k != "df"}
ex.note("±7 % 표 = 기준 캐시 (열별, S만 달라야 함)", same); print(same, flush=True)
g_sess = np.arange(n); g_sess[np.isin(names, SESSION)] = list(names).index(SESSION[0])


def sub(T, i):
    return {k: (v if np.ndim(v) == 0 else v[i]) for k, v in T.items()}


class JIdx:
    """nested_cv용 껍데기: 입력 행 = [파일 번호]. JShrink를 그 파일들의 표로 학습, 점수 = 최종 판정(결합 OR E2 OR 8.3k, 0/1)"""
    def __init__(self, T): self.T = T
    def fit(self, F, y): self.m_ = JShrink().fit(sub(self.T, np.asarray(F)[:, 0].astype(int)), y); return self
    def decision_function(self, F): return self.m_.parts(sub(self.T, np.asarray(F)[:, 0].astype(int)))["pred"].values.astype(float)


# %% 중첩 CV (full, 두 구성 같은 분할 = EXP-062 구성 3)
RES = {}
for nm in NM:
    t0 = time.time()
    RES[nm] = nested_cv(np.arange(n)[:, None].astype(float), y, g_sess, lambda T=TB[nm][0]: JIdx(T), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=nm == B)
    print(f"  {nm} {RES[nm]['f2'].mean():.3f} ± {RES[nm]['f2'].std():.3f}  {time.time() - t0:.0f}s", flush=True)
dd = RES[M1]["f2"] - RES[B]["f2"]
pair = dict(diff=round(float(dd.mean()), 3), diff_sd=round(float(dd.std()), 3), seed_diffs=np.round(dd, 3).tolist(),
            base_sd=round(float(RES[B]["f2"].std()), 3), exceeds_sd=bool(abs(dd.mean()) > RES[B]["f2"].std()))
ex.note("쌍 비교 (구성 1 − 기준, seed별)", pair); print(pair, flush=True)


def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in NM}
ex.note("주목 파일 고장 판정 횟수 (5번 중)", {nm: {f: int(C[nm][ix(f)]) for f in WATCH} for nm in NM})
ex.note("오류 합계 (5번 중 횟수 합: 놓침 · 오경보)", {nm: {"놓침": int(RES[nm]["miss_counts"].sum()), "오경보": int(RES[nm]["fp_counts"].sum())} for nm in NM})

# %% 전체 학습 (train 150) × seed 5 → 여유 · 계수 · 규칙 · val 판정 · val 안정성
SEEDS = range(5)


def fit_one(T, s):
    m = JShrink(seed=s).fit(T[0], y); return m, m.parts(T[0]), m.parts(T[1])


jobs = [(nm, s) for nm in NM for s in SEEDS]
out = Parallel(n_jobs=-1)(delayed(fit_one)(TB[nm], s) for nm, s in jobs)
ALLF = dict(zip(jobs, out))
FIT, MG = {}, {}
for nm in NM:
    m, Pt, Pv = ALLF[(nm, 0)]
    comb_oof = m.c_.decision_function(m.Z_) >= m.tc_                                   # 학습 파일의 결합 판정은 안쪽 OOF 입력으로
    FIT[nm] = dict(m=m, Pt=Pt, Pv=Pv, comb=comb_oof, fin=comb_oof | Pt.E2.values | Pt.rule83.values, vfin=Pv.pred.values.astype(bool))
    MG[nm] = np.array([ALLF[(nm, s)][0].c_.decision_function(ALLF[(nm, s)][0].Z_) - ALLF[(nm, s)][0].tc_ for s in SEEDS])   # 학습 여유 (OOF 입력, 근사)

# 주목 파일: hit (±3 · ±7 %), 박자 z, 결합 여유 중앙값 (seed 5)
imW = all_impacts(ns.X[[ix(f) for f in WATCH]], fmax=FMAX)
WT = {}
for k, f in enumerate(WATCH):
    i = ix(f); x = ns.X[i]
    h3 = lock_S(x, imW[k]["t"])[3]; h7 = lock_S(x, imW[k]["t"], tol=TOL)[3]
    WT[f] = {"라벨": "고장" if y[i] else "정상", "hit ±3 %": round(h3, 3), "hit ±7 %": round(h7, 3),
             **{f"{nm.split()[0]} 박자 z": round(float(FIT[nm]["m"].Z_[i, 1]), 2) for nm in NM},
             **{f"{nm.split()[0]} 여유 (중앙)": round(float(np.median(MG[nm][:, i])), 2) for nm in NM},
             **{f"{nm.split()[0]} 5번 중 고장": int(C[nm][i]) for nm in NM}}
ex.note("주목 파일 (hit · 박자 z · 결합 여유 중앙 · CV 고장 횟수)", WT)

# 박자 z 분포 (정상 120): 학습 정상 기준 강건 z, seed 무관
ZD = {}
for nm in NM:
    z = FIT[nm]["m"].Z_[:, 1]; zn = z[y == 0]
    ZD[nm] = {"정상 z > 2 수": int((zn > 2).sum()), "정상 z > 2 파일": " ".join(short(names[i]) for i in np.flatnonzero((y == 0) & (z > 2))) or "—",
              "정상 z 90 · 99 % · 최대": f"{np.percentile(zn, 90):.2f} · {np.percentile(zn, 99):.2f} · {zn.max():.2f}",
              "고장 z > 2 수": int((z[y == 1] > 2).sum()), "고장 z 중앙": round(float(np.median(z[y == 1])), 2),
              "정상 S 중앙 · 척도": f"{FIT[nm]['m'].zf_.med:.3f} · {FIT[nm]['m'].zf_.sc:.3f}"}
ex.note("박자 z 분포 (train 150)", ZD); print(ZD, flush=True)

STAB, VP, VM = {}, {}, {}
for nm in NM:
    VP[nm] = np.array([ALLF[(nm, s)][2].pred.values for s in SEEDS]); VM[nm] = np.array([ALLF[(nm, s)][2].margin.values for s in SEEDS])
for nm in NM:
    flip = (VP[nm].min(0) != VP[nm].max(0)); vz = ALLF[(nm, 0)][2].beat_z.values
    STAB[nm] = {"seed마다 판정이 바뀌는 val 수": int(flip.sum()), "바뀌는 val": " ".join(short(nv[i]) for i in np.flatnonzero(flip)) or "—",
                "val |여유| 중앙값 (seed 평균 여유)": round(float(np.median(np.abs(VM[nm].mean(0)))), 3),
                "val 고장 판정 (seed 0 · 과반)": f"{int(VP[nm][0].sum())} · {int((VP[nm].mean(0) > .5).sum())}",
                "val 박자 z > 2": " ".join(short(nv[i]) for i in np.flatnonzero(vz > 2)) or "—"}
    print(nm, STAB[nm], flush=True)
ex.note("val 안정성 (seed 5 전체 학습)", STAB)
fires, COEF = {}, {}
lst = lambda P, nn, col: " ".join(short(nn[i]) for i in np.flatnonzero(P[col].values)) or "—"
for nm, F in FIT.items():
    Pt, Pv = F["Pt"], F["Pv"]
    cnt2 = lambda col: f"{int((Pt[col] & (y == 1)).sum())}/{int((Pt[col] & (y == 0)).sum())}"
    cf = np.array([ALLF[(nm, s)][0].c_.coef_[0] for s in SEEDS]); tc = np.array([ALLF[(nm, s)][0].tc_ for s in SEEDS])
    COEF[nm] = {"계수 ET (seed 0–4 평균)": round(float(cf[:, 0].mean()), 3), "계수 박자": round(float(cf[:, 1].mean()), 3),
                "계수 AE": round(float(cf[:, 2].mean()), 3), "문턱": round(float(tc.mean()), 3),
                "seed 0 계수 · 문턱": f"{np.round(cf[0], 3).tolist()} · {tc[0]:.3f}"}
    fires[nm] = {"train E2 (고장/정상)": cnt2("E2"), "train 8.3k 합 (고장/정상)": cnt2("rule83"),
                 "val E2": lst(Pv, np.array(nv), "E2"), "val 8.3k": lst(Pv, np.array(nv), "rule83"),
                 "val 고장 판정": f"{int(F['vfin'].sum())}/40", "결합 계수 (ET · 박자 · AE)": np.round(F["m"].c_.coef_[0], 3).tolist(),
                 "결합 문턱": round(F["m"].tc_, 3), "E2 문턱": round(F["m"].t18_, 3)}
ex.note("결합 계수 · 문턱 (seed 0–4 전체 학습)", COEF)
ex.note("규칙 켜짐 · 결합 계수 (train 150 전체 학습, seed 0)", fires)
tb = lambda D, k: md_table(pd.DataFrame(D).T.reset_index().rename(columns={"index": k}))
ex.save_md("규칙_켜짐", "train 150 전체 학습 모델 (seed 0). 8.3k = 스펙트럼(P83 · W83 · K83) OR 공진(RES ≥ 0.7). E2 = L2 위 꼬리 규칙\n\n"
           + tb(fires, "구성") + "\n\n## 결합 계수 · 문턱 (seed 0–4 전체 학습)\n\n" + tb(COEF, "구성")
           + "\n\n## 박자 z 분포 (train 150, 학습 정상 기준 강건 z)\n\n" + tb(ZD, "구성")
           + "\n\n## 주목 파일 (hit = 충격 중 T ± 허용 안에 다른 충격이 있는 비율. 여유 = 결합 점수 − 문턱, seed 5 전체 학습의 OOF 입력 중앙값)\n\n" + tb(WT, "파일")
           + "\n\n## val 안정성 (seed 0–4로 각각 전체 학습 → val)\n\n" + tb(STAB, "구성"))

# %% 오류 분석 표 (가로): 파일별 고장 판정 횟수 (seed 5번 중)
files = [names[i] for i in range(n) if any((C[nm][i] > 0) != (y[i] == 1) or (y[i] == 1 and C[nm][i] < 5) for nm in C)]
files += [f for f in WATCH if f not in files]
extra = {f"{B} (5번 중 고장)": {f: int(C[B][ix(f)]) for f in files}}
extra["1 · E2 / 8.3k (전체 학습)"] = {f: ("E2 " if FIT[M1]["Pt"].E2.values[ix(f)] else "") + ("8.3k" if FIT[M1]["Pt"].rule83.values[ix(f)] else "") or "—" for f in files}
ex.save_md("오류분석_표", f"모델 판단 행 = {M1}. 아래 행 = 기준의 고장 판정 횟수. 열 = 어느 구성에서든 한 번이라도 틀린 파일 + 주목 파일\n\n"
           + error_md_table(names, y, C[M1], files, n_seeds=5, extra=extra))

# %% val 판정 (라벨 없음, 제출 아님). seed 0 전체 학습
k_ = lambda nm: nm.split()[0]
V = pd.DataFrame({"파일": [v[-3:] for v in nv], **{k_(nm): np.where(FIT[nm]["vfin"], "고장", "정상") for nm in NM},
                  **{f"{k_(nm)} 여유": np.round(FIT[nm]["Pv"].margin.values, 2) for nm in NM},
                  **{f"{k_(nm)} seed 고장 수": VP[nm].sum(0) for nm in NM},
                  **{f"{k_(nm)} 박자 z": np.round(FIT[nm]["Pv"].beat_z.values, 2) for nm in NM},
                  "규칙": [("E2 " if e else "") + ("8.3k" if r else "") for e, r in zip(FIT[M1]["Pv"].E2, FIT[M1]["Pv"].rule83)]})
d = V[V[[k_(nm) for nm in NM]].nunique(axis=1) > 1]
vchg = " ".join(short(nv[i]) + ("+" if FIT[M1]["vfin"][i] else "−") for i in np.flatnonzero(FIT[M1]["vfin"] != FIT[B]["vfin"])) or "없음"
vchg_s = " ".join(short(nv[i]) for i in np.flatnonzero((VP[M1].mean(0) > .5) != (VP[B].mean(0) > .5))) or "없음"
ex.note("val 판정 수 (seed 0)", {nm: int(FIT[nm]["vfin"].sum()) for nm in NM})
ex.note("val 기준 대비 바뀐 파일 (seed 0, + 고장으로 · − 정상으로) · 과반 기준", {"seed 0": vchg, "seed 과반": vchg_s})
ex.save_md("val_판정", "val 고장 판정 수 (seed 0 전체 학습): " + " · ".join(f"{k_(nm)} {int(FIT[nm]['vfin'].sum())}/40" for nm in NM)
           + f"\n\n기준(0) 대비 바뀐 파일 (+ 고장으로 · − 정상으로): seed 0 {vchg} · seed 과반 {vchg_s}"
           + "\n\n여유 = 결합 점수 − 문턱. seed 고장 수 = seed 0–4 전체 학습 중 고장 판정 횟수. 규칙 = 구성 1의 E2 · 8.3k 켜짐\n\n구성 사이에 판정이 다른 파일\n\n"
           + (md_table(d) if len(d) else "없음") + "\n\n전체\n\n" + md_table(V))

# %% 핵심 그림: logistic_plane 기준(0) vs 구성 1 (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정). 규칙만으로 잡힌 고장은 뺌
LAB = ["abn_028", "abn_014", "abn_022", "abn_021", "abn_001"]


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
        for i in np.flatnonzero(show & ((fin != (y == 1)) | np.isin(names, LAB))):
            ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")


leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, nm in zip(axs, NM): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합, seed 0). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 틀린 파일 · 028 014 022 021 001", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준(0) vs 박자 허용 ±7 %(1)의 결합 경계", key=True)
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
for ax, nm in zip(axs, NM): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습, seed 0). 굵은 글씨 = 기준(0)과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "기준 vs 구성 1의 val 판정")

# %% t-SNE 기본 (핵심 그림, 기본 공간 = 기준 피처). 빨간 원 = 구성 1 오경보 (5번 중 3회+)
fp1 = np.flatnonzero((y == 0) & (RES[M1]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fp1, mark_label="구성 1 오경보 (5번 중 3회+)", title="EXP-066 (기준 피처 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10, 기준 피처). 빨간 원 = 구성 1 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
