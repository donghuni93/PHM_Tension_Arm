# %% [markdown]
# EXP-070 첨도 피처: ET 열 vs 결합 입력 (11-27, 근거 5_VAL_단계 V-11 val고장도출_피처탐색)
# - 가설: H1″ (물리가설.md). 고립 강타 = 2초 중 한 순간만 크게 튐 → 4–8 kHz 첨도 · 200 ms 창 첨도가 극단값
# - 동기: V-11에서 train 정상 2개(054 · 073 / 050 · 088)만 넘는 선을 val 미판정 012 · 029 · 032 · 034 · 036이 넘는다(기대 0.5개).
#   H5(4–8 kHz 첨도)는 이미 ET 입력인데 판정에 안 쓰인다. 나무는 "이 징후만 있는 train 고장"이 없어 꼬리를 쓰지 않는다
# - 새 값: WK200 = log10(200 ms 창(절반 겹침) 첨도 최대, 1–11 kHz 대역통과, V-11 run.py feats 정의 그대로, 캐시 v11_feats.csv)
#          H5Z = 기존 H5_kurt_4-8k (j_tables H 열, log10 첨도)
# - 기준(구성 0) = EXP-062 구성 3 그대로 (EXP-068과 같은 분할)
# - 구성 1 = ET 열 + wk200_max (JShrink cols)
# - 구성 2 = 결합(로지스틱) 입력 + H5 강건 z (JShrink zcomb, 학습 정상 중앙 · MAD, 박자 z와 같은 식)
# - 구성 3 = 결합 입력 + WK200 강건 z
# - val은 라벨이 없다. 판정 수 · 바뀐 파일 · seed 안정성만 본다
# 실행: conda run -n qe-phm python run.py (quick) · 기록용 QE_FULL=1

# %% 설정
import os, sys, warnings, time
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL, best_f2_threshold
from src.exp2util import NS, NOR, ABN, tsne_default
from src.final import JShrink, J_FIXED
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
FMAX = 11000
SESSION = ("nor_097", "nor_101", "nor_120")
ex = Experiment(__file__, "EXP-070", "첨도 피처 (ET 열 vs 결합 입력, 고립 강타형)", ["H1″"],
                member="이동훈", model="J-축소 + K83 (src/final.py JShrink) · 11 kHz 상한 (EXP-062 구성 3)",
                changes="기준 = EXP-062 구성 3. 1: ET 열 + 200 ms 창 첨도 최대(WK200) · 2: 결합 입력 + H5(4–8 kHz 첨도) 강건 z · 3: 결합 입력 + WK200 강건 z (5_VAL_단계 V-11). 이상치 학습 포함, 그룹 CV",
                key_changes=["기준 = EXP-062 구성 3 (옛 검출기 250 Hz 사전 · 옛 RES), 같은 분할",
                             "1: ET 열 + wk200_max = log10(200 ms 창 첨도 최대, 1–11 kHz) (JShrink cols)",
                             "2: 결합(로지스틱) 입력 + H5_kurt_4-8k 강건 z (JShrink zcomb, 학습 정상 기준) — ET 입력은 그대로",
                             "3: 결합 입력 + wk200_max 강건 z",
                             "울림 축소 · E2 · 8.3k 규칙은 그대로"])
ns = NS(); y, names = ns.y, np.array(ns.names); n = ns.n; nv = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
OUT = outlier_idx(names)
NEW = "wk200_max"; NEWK = "WK200"

CFG = [("0 기준 (EXP-062 구성 3)", {}), ("1 ET 열 + WK200", dict(cols=J_FIXED + [NEW])),
       ("2 결합 + H5 z", dict(zcomb=["H5Z"])), ("3 결합 + WK200 z", dict(zcomb=[NEWK]))]
NM = [c[0] for c in CFG]; KW = dict(CFG); B = NM[0]; M2 = NM[-1]
WATCH = ["abn_021", "abn_028", "abn_014", "abn_022", "abn_017", "nor_059", "nor_065", "nor_116", "nor_095", "nor_110",
         "nor_046", "nor_060", "nor_054", "nor_032", "nor_055"]          # 뒤 5개 = train 고립 강타 정상 (5_VAL_단계 V-2 · 3-3)
VWATCH = ["val_012", "val_029", "val_032", "val_034", "val_036", "val_009", "val_006"]   # V-11에서 선을 넘은 val 미판정 (진단용, 고르지 않음)

# %% 표: EXP-062 구성 3 캐시(jtab_*_11k.npz) + 새 열 WSEGK
t0 = time.time()
TB = []
for tag, X in (("train", ns.X), ("val", ns.Xv)):
    z = np.load(os.path.join(CACHE, f"jtab_{tag}_11k.npz")); T = {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}
    T["H5Z"] = np.asarray(T["H"], float)[:, J_FIXED.index("H5_kurt_4-8k")]
    TB.append(T)
v11 = pd.read_csv(os.path.join(CACHE, "v11_feats.csv"), index_col=0)          # 5_VAL_단계 V-11 run.py가 만든 캐시 (라벨 없이 파일마다)
TB[0][NEWK] = np.log10(v11.loc[list(names), "WK200_1-11k_max"].values); TB[1][NEWK] = np.log10(v11.loc[nv, "WK200_1-11k_max"].values)
TB = tuple(TB)
print(f"표 {time.time() - t0:.0f}s", flush=True)
# H5 = V-11 K_4-8k와 같은 정보인가 (전처리가 달라 값은 다름, 순위만 확인)
same = {"ρ(H5, V-11 K_4-8k) train": round(float(spearmanr(TB[0]["H5Z"], v11.loc[list(names), "K_4-8k"])[0]), 3),
        "ρ(H5, V-11 K_4-8k) val": round(float(spearmanr(TB[1]["H5Z"], v11.loc[nv, "K_4-8k"])[0]), 3)}
ex.note("H5 대 V-11 K_4-8k", same); print(same, flush=True)
g_sess = np.arange(n); g_sess[np.isin(names, SESSION)] = list(names).index(SESSION[0])

# 중복: 새 열과 J 열의 Spearman ρ (train 150, 원래 H 값 = 축소 전)
Hdf = pd.DataFrame(TB[0]["H"], columns=J_FIXED)
v = TB[0][NEWK]; r = {j: spearmanr(v, Hdf[j])[0] for j in J_FIXED}; jb = max(r, key=lambda q: abs(r[q]))
RHO = {NEW: {"가장 겹치는 J 열": f"{jb} {r[jb]:+.3f}", "ρ H3_kurt_1-2k": round(float(r["H3_kurt_1-2k"]), 3), "ρ H8_ring_2-4k": round(float(r["H8_ring_2-4k"]), 3),
             "ρ H3 (정상 120)": round(float(spearmanr(v[y == 0], Hdf['H3_kurt_1-2k'][y == 0])[0]), 3)}}
ex.note("새 열 중복 (Spearman ρ, train 150)", RHO); print(RHO, flush=True)



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
    print(f"  {nm} {RES[nm]['f2'].mean():.3f} ± {RES[nm]['f2'].std():.3f}  {time.time() - t0:.0f}s", flush=True)
pd_ = lambda a, b: dict(diff=round(float((RES[a]["f2"] - RES[b]["f2"]).mean()), 3), diff_sd=round(float((RES[a]["f2"] - RES[b]["f2"]).std()), 3),
                        seed_diffs=np.round(RES[a]["f2"] - RES[b]["f2"], 3).tolist(), base_sd=round(float(RES[b]["f2"].std()), 3),
                        exceeds_sd=bool(abs((RES[a]["f2"] - RES[b]["f2"]).mean()) > RES[b]["f2"].std()))
pair = {nm: pd_(nm, B) for nm in NM[1:]}
ex.note("쌍 비교 (구성 − 기준, seed별)", pair); print(pair, flush=True)


NSD = len(RES[B]["f2"])                                   # 중첩 CV seed 수 (quick 2 · full 5)
def cnt(r): return np.where(y == 1, NSD - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in NM}
ix = lambda f: list(names).index(f)
watch = {nm: {f: int(C[nm][ix(f)]) for f in WATCH} for nm in NM}
ex.note(f"주목 파일 고장 판정 횟수 ({NSD}번 중)", watch); print(watch, flush=True)

# %% 전체 학습 (train 150) × seed 5 → 규칙 켜짐 · val 판정 · val 안정성
SEEDS = range(5)


def fit_one(kw, s):
    m = JShrink(seed=s, **kw).fit(TB[0], y); return m, m.parts(TB[0]), m.parts(TB[1])


jobs = [(nm, s) for nm in NM for s in SEEDS]
out = Parallel(n_jobs=-1)(delayed(fit_one)(KW[nm], s) for nm, s in jobs)
ALLF = dict(zip(jobs, out))
FIT = {}
for nm in NM:
    m, Pt, Pv = ALLF[(nm, 0)]
    comb_oof = m.c_.decision_function(m.Z_) >= m.tc_                                   # 학습 파일의 결합 판정은 안쪽 OOF 입력으로
    FIT[nm] = dict(m=m, Pt=Pt, Pv=Pv, comb=comb_oof, fin=comb_oof | Pt.E2.values | Pt.rule83.values, vfin=Pv.pred.values.astype(bool))
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
# ET 중요도 (seed 0 전체 학습): 새 열이 실제로 쓰이는지
imp = {nm: {c: round(float(v), 3) for c, v in zip(KW[nm].get("cols", J_FIXED) + ["H16_peak_h", "H17_peak_w"], FIT[nm]["m"].et_.feature_importances_)} for nm in NM}
ex.note("ET 불순도 중요도 (seed 0 전체 학습)", imp)
ex.save_md("규칙_켜짐", "train 150 전체 학습 모델 (seed 0). 8.3k = 스펙트럼(P83 · W83 · K83) OR 공진(RES ≥ 0.7). E2 = L2 위 꼬리 규칙(문턱 = 학습 정상 L2_hfl 99 %)\n\n"
           + md_table(pd.DataFrame(fires).T.reset_index().rename(columns={"index": "구성"}))
           + "\n\n## val 안정성 (seed 0–4로 각각 전체 학습 → val)\n\n여유 = 결합 점수 − 문턱. Δ = 기준 대비 val ET logit(seed 평균) 변화\n\n"
           + md_table(pd.DataFrame(STAB).T.reset_index().rename(columns={"index": "구성"}))
           + "\n\n## 새 열 중복 (Spearman ρ, train 150 · 정상 120)\n\n" + md_table(pd.DataFrame(RHO).T.reset_index().rename(columns={"index": "열"}))
           + "\n\n## ET 불순도 중요도 (seed 0 전체 학습)\n\n" + md_table(pd.DataFrame(imp).reset_index().rename(columns={"index": "열"})))

# %% 오류 분석 표 (가로): 파일별 고장 판정 횟수 (seed NSD번 중)
files = [names[i] for i in range(n) if any((C[nm][i] > 0) != (y[i] == 1) or (y[i] == 1 and C[nm][i] < NSD) for nm in C)]
files += [f for f in WATCH if f not in files]
extra = {f"{nm} ({NSD}번 중 고장)": {f: int(C[nm][ix(f)]) for f in files} for nm in NM[:2]}
extra["1 · E2 / 8.3k (전체 학습)"] = {f: ("E2 " if FIT[M2]["Pt"].E2.values[ix(f)] else "") + ("8.3k" if FIT[M2]["Pt"].rule83.values[ix(f)] else "") or "—" for f in files}
ex.save_md("오류분석_표", f"모델 판단 행 = {M2}. 아래 행 = 다른 구성의 고장 판정 횟수. 열 = 어느 구성에서든 한 번이라도 틀린 파일 + 주목 파일({' · '.join(short(f) for f in WATCH)})\n\n"
           + error_md_table(names, y, C[M2], files, n_seeds=NSD, extra=extra))

# %% val 판정 (라벨 없음, 제출 아님). seed 0 전체 학습
k_ = lambda nm: nm.split()[0]
V = pd.DataFrame({"파일": [v[-3:] for v in nv], **{k_(nm): np.where(FIT[nm]["vfin"], "고장", "정상") for nm in NM},
                  **{f"{k_(nm)} 여유": np.round(FIT[nm]["Pv"].margin.values, 2) for nm in NM},
                  **{f"{k_(nm)} seed 고장 수": VP[nm].sum(0) for nm in NM},
                  **{f"{k_(nm)} 규칙": [("E2 " if e else "") + ("8.3k" if r else "") for e, r in zip(FIT[nm]["Pv"].E2, FIT[nm]["Pv"].rule83)] for nm in NM},
                  NEW: np.round(TB[1][NEWK], 3)})
d = V[V[[k_(nm) for nm in NM]].nunique(axis=1) > 1]
vchg = {nm: " ".join(short(nv[i]) + ("+" if FIT[nm]["vfin"][i] else "−") for i in np.flatnonzero(FIT[nm]["vfin"] != FIT[B]["vfin"])) or "없음" for nm in NM[1:]}
ex.note("val 판정 수 (seed 0)", {nm: int(FIT[nm]["vfin"].sum()) for nm in NM}); ex.note("val 기준 대비 바뀐 파일 (+ 고장으로 · − 정상으로)", vchg)
vw = {f: {**{f"{k_(nm)} 판정 · seed 고장 수 · 여유": f"{'고장' if FIT[nm]['vfin'][nv.index(f)] else '정상'} · {int(VP[nm][:, nv.index(f)].sum())}/5 · {FIT[nm]['Pv'].margin.values[nv.index(f)]:+.2f}" for nm in NM}, "WK200 (log10)": round(float(TB[1][NEWK][nv.index(f)]), 3), "H5 (log10)": round(float(TB[1]["H5Z"][nv.index(f)]), 3)} for f in VWATCH}
ex.note("val 주목 파일 (고립 강타 012 · 032 · 034 · 036 + 009, 진단용)", vw); print(vw, flush=True)
ex.save_md("val_판정", "val 고장 판정 수 (seed 0 전체 학습): " + " · ".join(f"{k_(nm)} {int(FIT[nm]['vfin'].sum())}/40" for nm in NM)
           + "\n\n기준(0) 대비 바뀐 파일 (+ 고장으로 · − 정상으로): " + " · ".join(f"{k_(nm)} {v}" for nm, v in vchg.items())
           + "\n\n여유 = 결합 점수 − 문턱. seed 고장 수 = seed 0–4 전체 학습 중 고장 판정 횟수. 규칙 = E2 · 8.3k 켜짐. wk200_max = 200 ms 창 첨도 최대(log10)\n\n구성 사이에 판정이 다른 파일\n\n"
           + (md_table(d) if len(d) else "없음") + "\n\n전체\n\n" + md_table(V))

# %% 핵심 그림: logistic_plane 기준(0) vs 구성 1 · 2 (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정). 규칙만으로 잡힌 고장은 뺌
def plane(ax, nm, val=False):
    F = FIT[nm]; m = F["m"]; c, b, tc = m.c_.coef_[0], m.c_.intercept_[0], m.tc_; Z = m.Z_; t_et = float(best_f2_threshold(y, Z[:, 0]))
    if val: P = F["Pv"]; et, zb, y3, fin = P.et_logit.values, P.beat_z.values, P.ae_z.values, F["vfin"]
    else: et, zb, y3, fin = Z[:, 0], Z[:, 1], Z[:, 2], F["fin"]
    zs = [f(TB[1 if val else 0][k]) for f, k in zip(getattr(m, "zcf_", []), getattr(m, "zcomb", []))]
    ext = sum(c[3 + i] * v for i, v in enumerate(zs)) if zs else 0.0                    # 결합 입력에 더한 z 항 → 가로축에 접어 넣음
    x = et + (c[1] * zb + ext) / c[0]
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
        for i in np.flatnonzero(show & ((fin != (y == 1)) | np.isin(names, ["abn_021", "abn_028", "abn_014", "abn_022"]))):
            ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z" + (" + " + " + ".join(f"({c[3 + i] / c[0]:.2f})·{k} z" for i, k in enumerate(m.zcomb)) if zs else ""), color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
fig, axs = plt.subplots(2, 2, figsize=(16, 15))
for ax, nm in zip(axs.ravel(), NM): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 틀린 파일 · 021 · 028 · 014 · 022", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준(0) · ET 열 + WK200(1) · 결합 + H5 z(2) · 결합 + WK200 z(3)의 결합 경계. 가로축에 박자 · 첨도 z 항을 접어 넣음", key=True)
fig, axs = plt.subplots(2, 2, figsize=(16, 15))
for ax, nm in zip(axs.ravel(), NM): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습, seed 0). 굵은 글씨 = 기준(0)과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "구성별 val 판정")

# %% t-SNE 기본 (핵심 그림, 기본 공간 = 기준 피처). 빨간 원 = 구성 1 오경보 (5번 중 3회+)
fp2 = np.flatnonzero((y == 0) & (RES[M2]["fp_counts"] > NSD / 2))
fig, ti = tsne_default(ns, mark=fp2, mark_label=f"구성 3 오경보 ({NSD}번 중 과반)", title="EXP-070 (기준 피처 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10, 기준 피처). 빨간 원 = 구성 3 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
