# %% [markdown]
# EXP-059 팀원 피처(소린 32 계열)로 부품 교체 — val 갈림이 피처에서 오나 구조에서 오나 (4-12 비교계획 1 · 2단계)
# - 질문: val 판정이 J-축소(9)와 팀원(효섭 14 · 소린 17)에서 갈린다. J 구조에 팀원 피처를 넣으면 val이 팀원 쪽으로 가나
# - 구성 (같은 바깥 분할, 이상치 학습 제외 없음 — B가 제출 J-축소(EXP-050, 제외 없음)와 같아야 하므로)
#   B  J-축소 (제출, 8.3k 규칙 K83 > 3 포함 = rule83_flags(k_gate=True))
#   C  J 구조(로지스틱 + 박자 + AE + E2 + 8.3k) + ET 입력 = 소린 10개 + H1 · H18(E2가 씀), 봉우리 없음 — 피처 교체
#   D  J-축소 + 소린 10개 추가
# - 팀원 모델 자체는 다시 돌리지 않는다(사용자 11-23). 팀원 val = 효섭 제출 14(0단계 재현 40/40) · 소린 M1 html 17, 팀원 학습 오류 = 팀원 보고(4-12 문서)
# - 소린 10개 값 = 4-12 run_소린.py 표(소린 노트북 코드 그대로). F10은 150개 전체로 고른 목록, AR_resid는 학습 정상 120 전체 AR → 폴드 밖이라 C · D의 F2는 약간 낙관적
# 실행: conda run -n qe-phm python run.py  (quick 기본, 기록용 QE_FULL=1)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import best_f2_threshold
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN, tsne_default, j_mats, j_shrink
from src.models2 import TwoStack, RingShrink
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
F10 = ["b25_5_segk", "D_segk", "A_segk", "R_frac_2_5k", "R_ring_25_58", "AR_resid", "R_ring_811_58", "SK_200_3k", "b25_5_kurt", "I_density"]
HS_VAL14 = {1, 4, 9, 12, 16, 27, 29, 31, 32, 34, 35, 36, 38, 39}          # 효섭 제출(11-19) 고장, 0단계 재현 40/40
M1_VAL17 = HS_VAL14 | {6, 8, 26}                                          # 소린 M1 html(11-23) 고장

ex = Experiment(__file__, "EXP-059", "팀원 피처로 부품 교체 (val 갈림 원인)", ["H1", "H2"],
                member="이동훈", model="B J-축소 · C J 구조 + 소린 10 · D J-축소 + 소린 10",
                changes="J 구조는 두고 ET 입력을 소린 피처로 교체(C) · 추가(D)해 팀원과의 val 갈림이 피처에서 오는지 분리. 같은 분할",
                key_changes=["C = J 구조 + ET 입력 소린 10개(+ H1 · H18, E2용), 봉우리 없음", "D = J-축소 + 소린 10개",
                             "팀원 모델은 다시 돌리지 않음(팀원 val · 보고 오류를 기준으로 씀)", "이상치 학습 제외 없음(제출 J-축소와 같은 조건)"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
vnum = np.array([int(n[-3:]) for n in nv]); TEAM = np.isin(vnum, list(HS_VAL14)); M1 = np.isin(vnum, list(M1_VAL17))

# %% 소린 10개 열 (4-12 표에서 파일 이름으로 맞춤)
ST = pd.read_csv(os.path.join(ROOT, "3_단계", "4_피처", "4-12_1123_팀원피처비교", "figs", "소린_파일표.csv"), encoding="utf-8-sig")
ST = ST.set_index("file"); assert set(names) <= set(ST.index) and set(nv) <= set(ST.index)
SCOL = [f"S_{c}" for c in F10]
ns.T = pd.concat([ns.T.reset_index(drop=True), pd.DataFrame(ST.loc[names, F10].values, columns=SCOL)], axis=1)
ns.Tv = pd.concat([ns.Tv.reset_index(drop=True), pd.DataFrame(ST.loc[nv, F10].values, columns=SCOL)], axis=1)

# %% J 쪽 공통: 8.3k(K83 포함) · AE
spec, res = rule83_flags(ns.X, k_gate=True); sv, rv = rule83_flags(ns.Xv, k_gate=True); r83, r83v = spec | res, sv | rv
# 팀원도 같은 8.3k 스펙트럼 규칙(동훈 규칙)을 썼다 → 11-23 갱신(K83 > 3)을 팀원 판정에도 적용 (사용자 11-23).
# 팀원 val에서 규칙만으로 고장인 파일 = 001(효섭 확률 0.000). 규칙만 파일은 갱신된 스펙트럼 규칙(sv)으로 다시 판정
RULE_ONLY = np.isin(vnum, [1])
TEAM = (TEAM & ~RULE_ONLY) | (RULE_ONLY & sv); M1 = (M1 & ~RULE_ONLY) | (RULE_ONLY & sv)
ex.note("팀원 판정에 K83 적용", dict(val_001_스펙트럼규칙_K83포함=bool(sv[vnum == 1][0]), 효섭_고장=int(TEAM.sum()), 소린M1_고장=int(M1.sum()),
                                 학습_팀원규칙_고장_K83통과=bool(spec[np.isin(names, ["abn_002", "abn_006", "abn_010", "abn_029", "abn_030"])].all())))
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
FC = SCOL + ["H1_nimp", "H18_hfl"]; FD = list(ns.FIXED) + SCOL
def build_C(): return lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(FC), nf=len(FC), df=ns.df, peak=(), extra=(), use_ae=True))
CFG = [("B J-축소", list(ns.FIXED), lambda: j_shrink(ns)), ("C J 구조 + 소린 10", FC, build_C), ("D J-축소 + 소린 10", FD, lambda: j_shrink(ns, FD))]
CN = [c[0] for c in CFG]; TAG = dict(zip(CN, ["J-축소(제출)", "J 구조 + 소린 피처(교체)", "J-축소 + 소린 피처(추가)"]))
def ts_of2(top): return (top.m_.m_, top._tf) if isinstance(top, RingShrink) else (top.m_, lambda F: np.asarray(F, float))

# %% 중첩 CV (같은 분할)
RES = {}
for nm, fx, bl in CFG:
    M, _ = j_mats(ns, r83, fx); RES[nm] = nested_cv(M, y, g, bl(), n_jobs=-1); ex.record(nm, RES[nm], names, main=nm.startswith("B"), mid=not nm.startswith("B"))

# %% 전체 학습 모델 → val
FIT, VAL = {}, {}
for nm, fx, bl in CFG:
    M, Mv = j_mats(ns, r83, fx); top = bl()().fit(M, y); ts, tf = ts_of2(top)
    ft = top.fires(M); fv = ts.fires_raw(tf(Mv), RAW); comb = ts.c_.decision_function(ts.Z_) >= ts.tc_
    FIT[nm] = dict(ts=ts, comb=comb, fin=comb | ft["E2"] | r83, fv=fv); VAL[nm] = fv["main"] | fv["E2"] | r83v
ex.note("결합 계수 [ET, 박자, AE] · 문턱", {nm: dict(계수=np.round(FIT[nm]["ts"].c_.coef_[0], 3).tolist(), 문턱=round(float(FIT[nm]["ts"].tc_), 3)) for nm in CN})
ex.note("ET 중요도 (전체 학습 MDI)", {nm: dict(zip(fx + list(FIT[nm]["ts"].peak), np.round(FIT[nm]["ts"].m_.feature_importances_, 3).tolist())) for nm, fx, _ in CFG})

# %% 오류 분석 표 (학습, 가로)
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in CN}; NSD = len(RES[CN[0]]["f2"])
files = [names[i] for i in range(ns.n) if any((C[nm][i] > 0) if y[i] == 0 else (C[nm][i] < NSD) for nm in CN)]
TY = pd.read_csv(os.path.join(ROOT, "3_단계", "2_EDA", "2-1_고장유형분류", "자료", "figs", "types.csv"), index_col=0, encoding="utf-8-sig")["type"]
ex.save_md("오류분석_표", f"모델 판단 행 = B(J-축소), 아래 행 = C · D의 고장 판정 횟수 / seed {NSD}. 유형 = 2-1 고장 유형 분류\n\n"
           + error_md_table(names, y, C["B J-축소"], files, n_seeds=NSD,
                            extra={**{f"{TAG[nm]} (고장 판정)": {f: int(C[nm][list(names).index(f)]) for f in files} for nm in CN[1:]},
                                   "유형(2-1)": {f: TY.get(f, "-") for f in files}}))

# %% val 판정 표 + 일치 수
COLS = {**{nm: VAL[nm] for nm in CN}, "효섭": TEAM, "소린": M1}
agree = {f"{TAG.get(a, a)}–{TAG.get(b, b)}": int((COLS[a] == COLS[b]).sum()) for i, a in enumerate(COLS) for b in list(COLS)[i + 1:]}
V = pd.DataFrame({"val": [n[-3:] for n in nv], **{TAG.get(k, k): np.where(v, "고장", "정상") for k, v in COLS.items()},
                  **{f"{TAG[nm]} 결합−문턱": np.round(FIT[nm]["fv"]["score"] - FIT[nm]["ts"].tc_, 2) for nm in CN}})
lab = [TAG.get(k, k) for k in COLS]
# 묶음(사용자 11-23): 같은 구성들이 고장이라고 본 파일끼리 → 고장이라고 본 구성이 많은 묶음부터, 같은 수면 앞 구성(B, C, …)이 고장인 묶음부터, 묶음 안은 번호순
Mx = np.array([COLS[k] for k in COLS]).astype(int)
def group_name(p): return "모두 고장" if p.all() else "모두 정상" if not p.any() else "고장: " + " · ".join(l for l, b in zip(lab, p) if b)
GRP = [group_name(Mx[:, i]) for i in range(40)]
order = np.array(sorted(range(40), key=lambda i: (-Mx[:, i].sum(), tuple(-Mx[:, i]), vnum[i])))
V.insert(1, "묶음 (고장이라고 본 구성)", GRP); V = V.iloc[order].reset_index(drop=True)
split = V[V[lab].nunique(axis=1) > 1]
ex.note("val 일치 수 (40개 중)", agree); ex.note("val 고장 수", {k: int(v.sum()) for k, v in COLS.items()})
ex.save_md("val_판정", "val 고장 수: " + " · ".join(f"{TAG.get(k, k)} {int(v.sum())}" for k, v in COLS.items()) + "\n\n일치 수(40개 중): "
           + " · ".join(f"{k} {v}" for k, v in agree.items())
           + "\n\n표는 '묶음'(같은 구성들이 고장이라고 본 파일끼리)으로 정렬했다. 고장이라고 본 구성이 많은 묶음부터\n\n"
           + f"판정이 갈린 val\n\n{md_table(split)}\n\n전체\n\n{md_table(V)}")
print(agree, flush=True)

# %% 핵심 그림 1: 판정이 갈린 val (행 = 파일, 열 = 모델, 같은 판정 묶음끼리)
FULL = ["J-축소\n(제출)", "J 구조 +\n소린 피처\n(교체)", "J-축소 +\n소린 피처\n(추가)", "효섭\n(RF 배깅)", "소린\n(RF + ET)"]
CAUSE = {(1, 0, 1, 1, 1): ("공통 고장", "소린 피처로 교체한 구성만 놓침"),
         (0, 1, 0, 1, 1): ("소린 피처 때문", "J 구조에 소린 피처만 넣어도 고장"),
         (1, 0, 1, 0, 0): ("J 피처 때문", "H 피처가 있는 구성만 고장"),
         (0, 0, 0, 1, 1): ("효섭 · 소린 모델 / 임계값 때문", "같은 소린 피처라도 J 구조에서는 정상"),
         (1, 0, 0, 0, 0): ("J-축소만 고장", "소린 피처를 더하면 정상으로 감"),
         (0, 0, 0, 0, 1): ("소린만 고장", "소린 임계값 0.41이 가장 낮음")}
sp = [o for o in order if 0 < Mx[:, o].sum() < Mx.shape[0]]
allf = [vnum[o] for o in order if Mx[:, o].all()]; nonef = [o for o in order if not Mx[:, o].any()]
fig, ax = plt.subplots(figsize=(13, .52 * len(sp) + 2.6))
for r, o in enumerate(sp):
    for c in range(Mx.shape[0]):
        f = Mx[c, o] == 1
        ax.add_patch(plt.Rectangle((c, r), .94, .86, color=ABN if f else "#eef1f5", lw=0))
        ax.text(c + .47, r + .43, "고장" if f else "정상", ha="center", va="center", fontsize=10, color="white" if f else "#9aa3ae", weight="bold" if f else "normal")
    ax.text(-.12, r + .43, f"val_{vnum[o]:03d}", ha="right", va="center", fontsize=10, color=INK)
r0 = 0
for r in range(1, len(sp) + 1):
    if r == len(sp) or tuple(Mx[:, sp[r]]) != tuple(Mx[:, sp[r - 1]]):
        key = tuple(int(v) for v in Mx[:, sp[r0]]); t1, t2 = CAUSE.get(key, ("", ""))
        ax.text(Mx.shape[0] + .15, (r0 + r - .14) / 2, t1, ha="left", va="bottom", fontsize=10.5, weight="bold", color=INK)
        ax.text(Mx.shape[0] + .15, (r0 + r - .14) / 2 + .05, t2, ha="left", va="top", fontsize=9, color=MUTED)
        if r < len(sp): ax.axhline(r - .07, color=INK, lw=1.2, xmin=0, xmax=1)
        r0 = r
for c, nm in enumerate(FULL): ax.text(c + .47, -.25, nm, ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")
ax.set_xlim(-1.3, Mx.shape[0] + 4.2); ax.set_ylim(len(sp) + .1, -1.9); ax.axis("off")
ax.text(-1.3, -2.6, f"판정이 갈린 val {len(sp)}개 — 같은 판정끼리 묶음", fontsize=13, weight="bold", color=INK)
ax.text(-1.3, -2.05, f"나머지 {40 - len(sp)}개는 다섯 모델이 모두 같음: 모두 고장 {len(allf)}개 ({' · '.join(f'{v:03d}' for v in allf)}) · 모두 정상 {len(nonef)}개.  "
        "효섭 · 소린 판정은 갱신된 8.3k 규칙(K83 > 3) 반영", fontsize=9, color=MUTED)
fig.tight_layout()
ex.save_fig(fig, "val_판정지도", "판정이 갈린 val만, 같은 판정끼리 묶음. 오른쪽 = 묶음의 뜻", key=True)

# %% 핵심 그림 2: logistic_plane (B · C · D, 학습 + val)
def plane(ax, nm, val=False):
    F = FIT[nm]; ts = F["ts"]; c, b, tc = ts.c_.coef_[0], ts.c_.intercept_[0], ts.tc_; Z = ts.Z_; t_et = float(best_f2_threshold(y, Z[:, 0]))
    if val: fv = F["fv"]; et, zb, y3, fin = fv["a"], fv["z"], fv["ae"], VAL[nm]
    else: et, zb, y3, fin = Z[:, 0], Z[:, 1], Z[:, 2], F["fin"]
    x = et + c[1] / c[0] * zb
    x0, x1 = min(x.min(), t_et) - .6, max(x.max(), t_et) + .6; y0, y1 = y3.min() - .6, y3.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    ax.contourf(gx, gy, (c[0] * gx + c[2] * gy + b >= tc).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    col = np.where(fin, ABN, NOR)
    if val:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in range(len(x)):
            if VAL[nm][i] != TEAM[i]: ax.annotate(short(nv[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    else:
        show = ~((y == 1) & ~F["comb"] & fin)                                   # 규칙으로만 잡힌 고장은 뺌
        for msk, mkr, s_ in ((show & (y == 0), "s", 22), (show & (y == 1), "o", 40)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & (fin != (y == 1))): ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40 (굵은 이름 = 효섭과 다름)" if val else " · 학습 150"), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
fig, axs = plt.subplots(2, 3, figsize=(24, 13))
for k, nm in enumerate(CN): plane(axs[0, k], nm); plane(axs[1, k], nm, val=True)
fig.suptitle("logistic_plane — 위 = 학습 150(안쪽 OOF 입력), 아래 = val 40. 배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 색 = 최종 판정. 규칙만으로 잡힌 고장은 뺌", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "logistic_plane", "B · C · D의 결합 경계 (학습 · val)", key=True)

# %% t-SNE 기본 (핵심 그림, J 입력 공간): B와 C가 다르게 틀린 학습 파일 표시
K = NSD // 2 + 1; wrong = {nm: (C[nm] >= K) != (y == 1) for nm in CN}
mk = np.flatnonzero(wrong["B J-축소"] != wrong["C J 구조 + 소린 10"])
fig, ti = tsne_default(ns, mark=mk, mark_label="B · C 중 한쪽만 자주 틀린 파일", title="EXP-059 (J 입력 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = J-축소(B)와 소린 피처 교체(C) 중 한쪽만 자주 틀린 학습 파일", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
