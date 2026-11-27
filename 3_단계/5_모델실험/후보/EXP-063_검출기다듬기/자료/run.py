# %% [markdown]
# EXP-063 검출기 다듬기 R1–R3 (11-25 사용자 결정, 근거 2-36 검출기검증)
# - 기준 = EXP-062 구성 3 (J-축소 + K83, 11 kHz 상한, 그룹 CV 097 · 101 · 120, 이상치 059 · 065 · 116 학습 포함)
# - 바꾼 것 (하나씩, 그리고 셋 다). 모두 src/final.py j_tables 옵션, 기본값 = 예전과 같음
#   R1 events=True   충격을 25 ms 이벤트로 묶어 센다 (이웃 검출 ≤ 25 ms = 한 이벤트, 높이 · 울림 = 가장 큰 검출의 것).
#                    H1 수 · H2 2위/1위 · H7–H10 울림 비율(축소 n = 이벤트 수) · L1 hit 시점에 쓴다
#   R2 atoms="50"    울림 주파수 · ζ를 50 Hz 사전(_atoms50)으로 (250 Hz 사전은 1.9 kHz ζ를 틀림). H7–H10에 영향
#   R3 res_new=True  8.3 kHz 공진 규칙 입력 RES를 새 검출기(1–11 kHz 입력) + 50 Hz 사전으로. 문턱 RES_FIT_MIN 0.7 그대로
#   다듬기 1000 Hz · 다른 피처 정의 · 문턱은 그대로. 열 이름 그대로
# - 분할 = EXP-062 구성 3과 같음 (같은 그룹 · 같은 seed). 쌍 비교 = seed별 F2 차이
# - val은 라벨이 없다. 판정 수와 기준 대비 바뀐 파일만 본다 (val로 고르지 않는다)
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행 (JShrink ET 300과 seed 5를 맞춘다)"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
FMAX = 11000
SESSION = ("nor_097", "nor_101", "nor_120")
ex = Experiment(__file__, "EXP-063", "검출기 다듬기 R1–R3", ["H1", "H2"],
                member="이동훈", model="J-축소 + K83 (src/final.py JShrink) · 11 kHz 상한 · 검출기 다듬기",
                changes="EXP-062 구성 3에 R1 25 ms 이벤트 · R2 50 Hz 울림 사전 · R3 RES 새 검출기 (하나씩 · 셋 다). 이상치 학습 포함, 그룹 CV, full",
                key_changes=["R1: 충격 수 · 2위/1위 · 울림 비율 · 박자 hit를 25 ms 이벤트 단위로 (j_tables events=True)",
                             "R2: 울림 주파수 · ζ = 50 Hz 사전 (atoms=\"50\")", "R3: RES = 새 검출기(1–11 kHz) + 50 Hz 사전, 문턱 0.7 그대로 (res_new=True)",
                             "분할 · 모델 · 다른 피처 = EXP-062 구성 3 그대로"])
ns = NS(); y, names = ns.y, np.array(ns.names); n = ns.n; nv = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
OUT = outlier_idx(names)

# 구성: (이름, 캐시 태그, j_tables 옵션). 0의 태그 r0 = 새 코드로 다시 계산 (EXP-062 캐시 train_11k와 같은지 아래에서 확인)
CFG = [("0 기준 (EXP-062 구성 3)", "r0", {}), ("1 + R1 이벤트", "ev", dict(events=True)), ("2 + R2 50 Hz 사전", "a50", dict(atoms="50")),
       ("3 + R3 RES 새 검출기", "rn", dict(res_new=True)), ("4 R1 + R2 + R3", "all", dict(events=True, atoms="50", res_new=True))]
NM = [c[0] for c in CFG]; B = NM[0]


def jt(tag, X, kw):
    """j_tables 캐시 4_outputs/cache/jtab_{tag}.npz (새 태그, 기존 캐시는 건드리지 않음)"""
    p = os.path.join(CACHE, f"jtab_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}
    T = j_tables(X, FMAX, **kw); np.savez(p, **T); return T


t0 = time.time()
TB = {nm: (jt(f"train_11k_063{tag}", ns.X, kw), jt(f"val_11k_063{tag}", ns.Xv, kw)) for nm, tag, kw in CFG}
print(f"표 {time.time() - t0:.0f}s", flush=True)
# 새 코드의 기본(구성 0) = EXP-062 캐시와 같은가
old = np.load(os.path.join(CACHE, "jtab_train_11k.npz")); oldv = np.load(os.path.join(CACHE, "jtab_val_11k.npz"))
same0 = {k: bool(np.allclose(TB[B][0][k], old[k]) and np.allclose(TB[B][1][k], oldv[k])) for k in old.files}
ex.note("구성 0 표 = EXP-062 캐시(jtab_train_11k · val_11k)", same0); print("구성 0 = EXP-062 캐시:", same0, flush=True)
g_sess = np.arange(n); g_sess[np.isin(names, SESSION)] = list(names).index(SESSION[0])


def sub(T, i):
    return {k: (v if np.ndim(v) == 0 else v[i]) for k, v in T.items()}


class JIdx:
    """nested_cv용 껍데기: 입력 행 = [파일 번호]. JShrink를 그 파일들의 표로 학습, 점수 = 최종 판정(결합 OR E2 OR 8.3k, 0/1)"""
    def __init__(self, T): self.T = T
    def fit(self, F, y): self.m_ = JShrink().fit(sub(self.T, np.asarray(F)[:, 0].astype(int)), y); return self
    def decision_function(self, F): return self.m_.parts(sub(self.T, np.asarray(F)[:, 0].astype(int)))["pred"].values.astype(float)


# %% 중첩 CV (full, 모든 구성 같은 분할 = EXP-062 구성 3)
RES = {}
for nm in NM:
    t0 = time.time(); T = TB[nm][0]
    RES[nm] = nested_cv(np.arange(n)[:, None].astype(float), y, g_sess, lambda T=T: JIdx(T), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=nm.startswith("4"))
    print(f"  {time.time() - t0:.0f}s", flush=True)
# 쌍 비교: seed별 F2 차이 (같은 분할)
pair = {}
for nm in NM[1:]:
    d = RES[nm]["f2"] - RES[B]["f2"]
    pair[nm] = dict(diff=round(float(d.mean()), 3), diff_sd=round(float(d.std()), 3), seed_diffs=np.round(d, 3).tolist(),
                    base_sd=round(float(RES[B]["f2"].std()), 3), exceeds_sd=bool(abs(d.mean()) > RES[B]["f2"].std()))
ex.note("쌍 비교 (구성 − 기준, seed별)", pair); print(pair, flush=True)

# %% 전체 학습 모델 (train 150) → 규칙 켜짐 · val 판정
FIT = {}
for nm in NM:
    Ttr, Tv = TB[nm]; m = JShrink().fit(Ttr, y); Pt, Pv = m.parts(Ttr), m.parts(Tv)
    comb_oof = m.c_.decision_function(m.Z_) >= m.tc_                                   # 학습 파일의 결합 판정은 안쪽 OOF 입력으로
    FIT[nm] = dict(m=m, Pt=Pt, Pv=Pv, comb=comb_oof, fin=comb_oof | Pt.E2.values | Pt.rule83.values, vfin=Pv.pred.values.astype(bool))
fires = {}
lst = lambda P, nn, col: " ".join(short(nn[i]) for i in np.flatnonzero(P[col].values)) or "—"
for nm, F in FIT.items():
    Pt, Pv = F["Pt"], F["Pv"]
    cnt2 = lambda col: f"{int((Pt[col] & (y == 1)).sum())}/{int((Pt[col] & (y == 0)).sum())}"
    fires[nm] = {"train E2 (고장/정상)": cnt2("E2"), "train E2 파일": lst(Pt, names, "E2"),
                 "train 8.3k 스펙트럼 (고장/정상)": cnt2("spec83"), "train 스펙트럼 파일": lst(Pt, names, "spec83"),
                 "train RES (고장/정상)": cnt2("res83"), "train RES 파일": lst(Pt, names, "res83"),
                 "train 8.3k 합 파일": lst(Pt, names, "rule83"),
                 "val E2": lst(Pv, np.array(nv), "E2"), "val 스펙트럼": lst(Pv, np.array(nv), "spec83"), "val RES": lst(Pv, np.array(nv), "res83"),
                 "val RES 최대": round(float(Pv.RES.max()), 3),
                 "val 고장 판정": f"{int(F['vfin'].sum())}/40", "결합 계수 (ET · 박자 · AE)": np.round(F["m"].c_.coef_[0], 3).tolist(),
                 "결합 문턱": round(F["m"].tc_, 3), "E2 문턱": round(F["m"].t18_, 3)}
    print(nm, fires[nm], flush=True)
ex.note("규칙 켜짐 · 결합 계수 (train 150 전체 학습)", fires)
ex.save_md("규칙_켜짐", "train 150 전체 학습 모델. 8.3k = 스펙트럼(P83 · W83 · K83) OR 공진(RES ≥ 0.7). E2 문턱 = 학습 정상 H18 99 %\n\n"
           + md_table(pd.DataFrame(fires).T.reset_index().rename(columns={"index": "구성"})))

# %% 오류 분석 표 (가로): 파일별 고장 판정 횟수 (seed 5번 중)
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in NM}
files = [names[i] for i in range(n) if any((C[nm][i] > 0) != (y[i] == 1) or (y[i] == 1 and C[nm][i] < 5) for nm in C)]
M4 = NM[4]; ix = lambda f: list(names).index(f)
extra = {f"{nm} (5번 중 고장)": {f: int(C[nm][ix(f)]) for f in files} for nm in NM[:4]}
extra["4 · E2 / 8.3k (전체 학습)"] = {f: ("E2 " if FIT[M4]["Pt"].E2.values[ix(f)] else "") + ("8.3k" if FIT[M4]["Pt"].rule83.values[ix(f)] else "") or "—" for f in files}
ex.save_md("오류분석_표", f"모델 판단 행 = {M4}. 아래 행 = 다른 구성의 고장 판정 횟수. 열 = 어느 구성에서든 한 번이라도 틀린 파일\n\n"
           + error_md_table(names, y, C[M4], files, n_seeds=5, extra=extra))

# %% val 판정 (라벨 없음, 제출 아님)
k_ = lambda nm: nm.split()[0]
V = pd.DataFrame({"파일": [v[-3:] for v in nv], **{k_(nm): np.where(FIT[nm]["vfin"], "고장", "정상") for nm in NM},
                  **{f"{k_(nm)} 여유": np.round(FIT[nm]["Pv"].margin.values, 2) for nm in NM},
                  **{f"{k_(nm)} 규칙": [("E2 " if e else "") + ("8.3k" if r else "") for e, r in zip(FIT[nm]["Pv"].E2, FIT[nm]["Pv"].rule83)] for nm in NM}})
d = V[V[[k_(nm) for nm in NM]].nunique(axis=1) > 1]
vchg = {nm: " ".join(short(nv[i]) + ("+" if FIT[nm]["vfin"][i] else "−") for i in np.flatnonzero(FIT[nm]["vfin"] != FIT[B]["vfin"])) or "없음" for nm in NM[1:]}
ex.note("val 판정 수", {nm: int(FIT[nm]["vfin"].sum()) for nm in NM}); ex.note("val 기준 대비 바뀐 파일 (+ 고장으로 · − 정상으로)", vchg)
ex.save_md("val_판정", "val 고장 판정 수: " + " · ".join(f"{k_(nm)} {int(FIT[nm]['vfin'].sum())}/40" for nm in NM)
           + "\n\n기준(0) 대비 바뀐 파일 (+ 고장으로 · − 정상으로): " + " · ".join(f"{k_(nm)} {v}" for nm, v in vchg.items())
           + "\n\n여유 = 결합 점수 − 문턱. 규칙 = E2 · 8.3k 켜짐\n\n구성 사이에 판정이 다른 파일\n\n" + (md_table(d) if len(d) else "없음") + "\n\n전체\n\n" + md_table(V))

# %% 핵심 그림: logistic_plane 기준(0) vs 셋 다(4) (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정). 규칙만으로 잡힌 고장은 뺌
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
P2 = [B, M4]
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 틀린 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준(0) vs R1 + R2 + R3(4)의 결합 경계", key=True)
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습). 굵은 글씨 = 기준(0)과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "기준 vs R1 + R2 + R3의 val 판정")

# %% t-SNE 기본 (핵심 그림, 기본 공간 = 기준 피처). 빨간 원 = 구성 4 오경보 (5번 중 3회+)
fp4 = np.flatnonzero((y == 0) & (RES[M4]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fp4, mark_label="구성 4 오경보 (5번 중 3회+)", title="EXP-063 (기준 피처 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10, 기준 피처). 빨간 원 = 구성 4 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
