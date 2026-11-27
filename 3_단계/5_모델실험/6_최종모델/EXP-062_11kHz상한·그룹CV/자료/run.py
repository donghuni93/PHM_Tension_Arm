# %% [markdown]
# EXP-062 11 kHz 상한 · 그룹 CV (11-25 사용자 결정, 파이프라인점검 S1 · P1–P3)
# - 바꾼 것: 기준 J-축소 + K83(src/final.py JShrink)에 (2) CV 그룹 097 · 101 · 120 한 그룹 → (3) 11 kHz 상한 변형(j_tables(fmax=11000))
#   11 kHz 변형 = 충격 검출 · 울림 · H18 입력 1–11 kHz 대역통과(4차, 검출 상수 그대로), 옥타브 마지막 칸 8–11 kHz(H5 첨도 · H10 울림 · H15 몫),
#   파워 몫 분모 1–11 kHz, 봉우리 log PSD 1–11 kHz, AE 입력 11 kHz 이하 bin. 8.3k 규칙 재료(P83 · W83 · K83 · RES)는 그대로
# - 구성 (모두 K83 규칙, 이상치 정상 059 · 065 · 116 학습 포함 = 11-25 결정, FinalJ와 같은 조건)
#   1 기준: 파일당 1그룹 (EXP-050 J-축소 0.977과 같은 조건 + K83)
#   2 기준 + 그룹 CV · 3 11 kHz + 그룹 CV — 2와 3은 같은 분할 (1은 그룹이 달라 분할이 다르다)
# - 모델 코드 = 제출용 src/final.py JShrink 그대로(실험용 TwoStack은 AE가 파일 번호 캐시라 11 kHz 입력을 못 받음).
#   두 코드가 같다는 것은 3_단계/7_최종·제출/검증_동등성.py. JShrink 안쪽 4겹 OOF(결합 입력)는 그룹을 쓰지 않는다(FinalJ와 같음)
# - val은 라벨이 없다. 판정 수와 구성 사이에 바뀐 파일만 본다 (val로 고르지 않는다)
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
FMAX = 11000                                             # 11-25 사용자 결정 (실행 전 고정)
SESSION = ("nor_097", "nor_101", "nor_120")              # 같은 세션 후보 → CV 한 그룹 (결정.md 11-25)
ex = Experiment(__file__, "EXP-062", "11 kHz 상한 · 그룹 CV", [],
                member="이동훈", model="J-축소 + K83 (src/final.py JShrink) · 11 kHz 상한 변형",
                changes="CV 그룹(097 · 101 · 120 한 그룹) → 전처리 11 kHz 상한 + 검출기 1–11 kHz 대역통과. 이상치 학습 포함, full",
                key_changes=["CV 그룹: 097 · 101 · 120 한 그룹 (나머지 파일당 1그룹)", "11 kHz 상한: 검출 · 울림 · H18 입력 1–11 kHz 대역통과, 옥타브 마지막 칸 8–11 kHz, 몫 분모 · 봉우리 · AE 입력 ≤ 11 kHz",
                             "이상치 정상 059 · 065 · 116 학습 포함 (11-25)", "모델 코드 = 제출용 JShrink (8.3k 규칙 K83 포함)"])
ns = NS(); y, names = ns.y, np.array(ns.names); n = ns.n; nv = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
OUT = outlier_idx(names)


def jt(tag, X, fmax):
    """j_tables 캐시 4_outputs/cache/jtab_{tag}.npz (새 태그, 기존 캐시는 건드리지 않음)"""
    p = os.path.join(CACHE, f"jtab_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: (float(z[k]) if k == "df" else z[k]) for k in z.files}
    T = j_tables(X, fmax); np.savez(p, **T); return T


t0 = time.time()
TB = {"base": (jt("train", ns.X, None), jt("val", ns.Xv, None)), "11k": (jt("train_11k", ns.X, FMAX), jt("val_11k", ns.Xv, FMAX))}
print(f"표 {time.time() - t0:.0f}s", flush=True)
g_file = np.arange(n); g_sess = g_file.copy(); g_sess[np.isin(names, SESSION)] = list(names).index(SESSION[0])
CFG = [("1 기준 (파일 그룹)", "base", g_file), ("2 기준 + 그룹 CV", "base", g_sess), ("3 11 kHz + 그룹 CV", "11k", g_sess)]


def sub(T, i):
    return {k: (v if np.ndim(v) == 0 else v[i]) for k, v in T.items()}


class JIdx:
    """nested_cv용 껍데기: 입력 행 = [파일 번호]. JShrink를 그 파일들의 표로 학습, 점수 = 최종 판정(결합 OR E2 OR 8.3k, 0/1)"""
    def __init__(self, T): self.T = T
    def fit(self, F, y): self.m_ = JShrink().fit(sub(self.T, np.asarray(F)[:, 0].astype(int)), y); return self
    def decision_function(self, F): return self.m_.parts(sub(self.T, np.asarray(F)[:, 0].astype(int)))["pred"].values.astype(float)


# %% 중첩 CV (full, 2 · 3은 같은 분할)
RES = {}
for nm, tb, g in CFG:
    t0 = time.time(); T = TB[tb][0]
    RES[nm] = nested_cv(g_file[:, None].astype(float), y, g, lambda T=T: JIdx(T), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=nm.startswith("3"))
    print(f"  {time.time() - t0:.0f}s", flush=True)

# %% 전체 학습 모델 (train 150) → train 규칙 켜짐 · val 판정
FIT = {}
for nm, tb, _ in CFG:
    if nm in FIT: continue
    Ttr, Tv = TB[tb]; m = JShrink().fit(Ttr, y); Pt, Pv = m.parts(Ttr), m.parts(Tv)
    comb_oof = m.c_.decision_function(m.Z_) >= m.tc_                                   # 학습 파일의 결합 판정은 안쪽 OOF 입력으로
    FIT[nm] = dict(m=m, Pt=Pt, Pv=Pv, comb=comb_oof, fin=comb_oof | Pt.E2.values | Pt.rule83.values, vfin=Pv.pred.values.astype(bool))
fires = {}
for nm, F in FIT.items():
    Pt, Pv = F["Pt"], F["Pv"]
    lst = lambda P, nn, col: " ".join(short(nn[i]) for i in np.flatnonzero(P[col].values))
    fires[nm] = {"train E2 (고장/정상)": f"{int((Pt.E2 & (y == 1)).sum())}/{int((Pt.E2 & (y == 0)).sum())}", "train E2 파일": lst(Pt, names, "E2"),
                 "train 8.3k (고장/정상)": f"{int((Pt.rule83 & (y == 1)).sum())}/{int((Pt.rule83 & (y == 0)).sum())}", "train 8.3k 파일": lst(Pt, names, "rule83"),
                 "val E2": lst(Pv, np.array(nv), "E2") or "—", "val 8.3k": lst(Pv, np.array(nv), "rule83") or "—",
                 "val 고장 판정": f"{int(F['vfin'].sum())}/40", "결합 계수 (ET · 박자 · AE)": np.round(F["m"].c_.coef_[0], 3).tolist(),
                 "결합 문턱": round(F["m"].tc_, 3), "E2 문턱": round(F["m"].t18_, 3)}
    print(nm, fires[nm], flush=True)
ex.note("규칙 켜짐 · 결합 계수 (train 150 전체 학습)", fires)
ex.save_md("규칙_켜짐", "train 150 전체 학습 모델. 8.3k 규칙은 학습 없음(11 kHz 변형과 무관한 같은 입력), E2 문턱 = 학습 정상 H18 99 %\n\n"
           + md_table(pd.DataFrame(fires).T.reset_index().rename(columns={"index": "구성"})))

# %% 오류 분석 표 (가로): 파일별 고장 판정 횟수 (seed 5번 중)
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm, _, _ in CFG}
files = [names[i] for i in range(n) if any((C[nm][i] > 0) != (y[i] == 1) for nm in C)]
M3 = CFG[2][0]
extra = {f"{nm} (5번 중 고장)": {f: int(C[nm][list(names).index(f)]) for f in files} for nm, _, _ in CFG[:2]}
extra["3 · E2 / 8.3k (전체 학습)"] = {f: ("E2 " if FIT[M3]["Pt"].E2.values[list(names).index(f)] else "") + ("8.3k" if FIT[M3]["Pt"].rule83.values[list(names).index(f)] else "") or "—" for f in files}
ex.save_md("오류분석_표", f"모델 판단 행 = {M3}. 아래 행 = 다른 구성의 고장 판정 횟수\n\n" + error_md_table(names, y, C[M3], files, n_seeds=5, extra=extra))

# %% val 판정 (라벨 없음, 제출 아님)
V = pd.DataFrame({"파일": [v[-3:] for v in nv], **{nm.split()[0]: np.where(FIT[nm]["vfin"], "고장", "정상") for nm, _, _ in CFG},
                  **{f"{nm.split()[0]} 여유": np.round(FIT[nm]["Pv"].margin.values, 2) for nm, _, _ in CFG},
                  **{f"{nm.split()[0]} 규칙": [("E2 " if e else "") + ("8.3k" if r else "") for e, r in zip(FIT[nm]["Pv"].E2, FIT[nm]["Pv"].rule83)] for nm, _, _ in CFG}})
d = V[V[["1", "2", "3"]].nunique(axis=1) > 1]
ex.note("val 판정 수", {nm: int(FIT[nm]["vfin"].sum()) for nm, _, _ in CFG})
ex.save_md("val_판정", "val 고장 판정 수: " + " · ".join(f"{nm} {int(FIT[nm]['vfin'].sum())}/40" for nm, _, _ in CFG)
           + "\n\n여유 = 결합 점수 − 문턱. 규칙 = E2 · 8.3k 켜짐\n\n구성 사이에 판정이 다른 파일\n\n" + (md_table(d) if len(d) else "없음") + "\n\n전체\n\n" + md_table(V))

# %% 핵심 그림: logistic_plane (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정). 규칙만으로 잡힌 고장은 뺌
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
    col = np.where(fin, ABN, NOR); ref = FIT[CFG[1][0]]
    if val:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in np.flatnonzero(fin | (fin != ref["vfin"])):
            ax.annotate(short(nv[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold" if fin[i] != ref["vfin"][i] else "normal")
    else:
        o = np.zeros(len(x), bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & ((fin != (y == 1)) | np.isin(names, SESSION))):
            ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
P2 = [CFG[1][0], CFG[2][0]]                              # 1과 2는 전체 학습 모델이 같다(그룹은 평가 분할에만) → 2 · 3만 그림
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 틀린 파일 · 097 · 101 · 120", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준(2) vs 11 kHz 상한(3)의 결합 경계 (1은 전체 학습 모델이 2와 같음)", key=True)
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, P2): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습). 굵은 글씨 = 기준(2)과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "기준 vs 11 kHz 상한의 val 판정")

# %% t-SNE 기본 (핵심 그림, 기본 공간 = 기준 피처). 빨간 원 = 11 kHz 구성 오경보 (5번 중 3회+)
fp3 = np.flatnonzero((y == 0) & (RES[M3]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fp3, mark_label="11 kHz 구성 오경보 (5번 중 3회+)", title="EXP-062 (기준 피처 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10, 기준 피처). 빨간 원 = 11 kHz 구성 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
