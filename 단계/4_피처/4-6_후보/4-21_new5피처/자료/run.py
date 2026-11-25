# %% 4-21 new_단계 이벤트 5피처를 최종 모델 EXP-062 구성 3(J-축소 + K83 + 11 kHz 상한)에 더할 가치가 있는지 (EDA만, 중첩 CV 없음, 11-25)
# 피처 = new_단계 2-6 정의, 값 = 2-7의 1–11 kHz 대역통과 검출기 입력 버전(events_190_bp11k.csv · feats_bp11k.csv, 읽기만)
#   F1 cnt_low (1.6–4.5 kHz로 운 이벤트 수) · F2 cnt_high (8–11 kHz) · F3 ratio_off = (low + high) ÷ (전체 + 1) · F4 cnt_total
#   F5 off_max_clamped = off 이벤트 중 포락선 ÷ max(바탕, b_floor) 최댓값, b_floor = Train 150 '파일 바탕 중앙값'의 하위 20 %(라벨 없이, 2-6 ①)
# 관문(계산 전 고정, 4-20과 같음): ① AUC 95 % CI 하한 > 0.6 ② J 열(H 13 + H16/H17 + L1_lock)과 max |ρ| < 0.8
#   ③ val 이동 AUC가 '정상 + 고장 0/20/40 % 섞음' 기준 범위 안 ④ 이상치 · EXP-062 오경보 · 경계 정상이 정상 99 % 위로 가지 않음
#   ⑤ 반쪽(앞 1초 vs 뒤 1초): 두 반쪽 모두 고장 > 정상(AUC > 0.5) 이고 정상 내 Spearman ρ ≥ 0.5 (4-20 정의). Train 150 ρ는 참고
# ⑥ 증분 가치(계산 전 고정): EXP-062 구성 3 모델(JShrink, 11 kHz 표)의 바깥 5겹 OOF 결합 여유를 통제
#   - LOO 로그손실 변화 ΔLL = LL(여유 + 피처) − LL(여유), 로지스틱(C=1, 표준화). 전체 150 · 규칙(E2/8.3k) 켜진 파일 뺀 집합 둘 다
#   - 부분 Spearman(여유 통제) · 여유가 작은 고장 · 오경보/경계 정상에서 피처의 정상 내 백분위
#   - '증분 있음' = 두 집합 모두 ΔLL < −0.005 이고 부분 ρ > 0.1. 하나만 = '약함'. 나머지 = '없음'
# OOF 여유는 단일 바깥 5겹(StratifiedGroupKFold, seed 0, 097 · 101 · 120 한 그룹) — 중첩 CV가 아니다(문턱은 JShrink 안쪽 그대로)
# 라벨을 본 탐색이다. 아무 대역 · 문턱도 여기서 고르지 않는다
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.exp2util import NS
from src.final import JShrink, J_FIXED
T0 = time.time()
CACHE = os.path.join(ROOT, "outputs", "cache"); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
B27 = os.path.join(ROOT, "단계", "3_전처리", "3-4_검출기대역통과", "자료", "figs")
LOW, HIGH = (1600, 4500), (8000, 11000)
FEATS = {"F1_cnt_low": "cnt_low", "F2_cnt_high": "cnt_high", "F3_ratio_off": "ratio_off", "F4_cnt_total": "cnt_total", "F5_off_max": "off_max_cl"}
TITLE = {"cnt_low": "cnt_events_low_off (1.6–4.5 kHz 이벤트 수)", "cnt_high": "cnt_events_high_off (8–11 kHz 이벤트 수)",
         "ratio_off": "ratio_off_events (off ÷ (전체+1))", "cnt_total": "cnt_events_total (전체 이벤트 수)", "off_max_cl": "off_max_clamped_amp (off 최대 배수, 하한 보정)"}
OUTL = ["nor_059", "nor_065", "nor_116"]; FP62 = ["nor_016", "nor_055", "nor_059", "nor_065", "nor_116"]   # EXP-062 구성 3 오경보 (result.json)
HARD_A = ["abn_014", "abn_021", "abn_022", "abn_028"]

# %% 피처: 2-7 이벤트 표(1–11 kHz 대역통과)에서 전체 · 앞 1초 · 뒤 1초
ns = NS(); y = ns.y; tr_names = list(ns.names); v_names = [f"val_{v[-3:]}" for v in ns.nv]; names = np.array(tr_names + v_names)
grp = np.array(["normal" if v == 0 else "abnormal" for v in y] + ["val"] * len(v_names))
N, A, V = grp == "normal", grp == "abnormal", grp == "val"; TR = N | A; pos = {f: i for i, f in enumerate(names)}
E = pd.read_csv(os.path.join(B27, "events_190_bp11k.csv")); E["base"] = E.h / E.x
bm = E.groupby("file").base.median().reindex(tr_names).dropna(); FLOOR = float(np.percentile(bm, 20))

def feats(Ev, files):
    G = {f: g for f, g in Ev.groupby("file")}; rows = []
    for f in files:
        e = G.get(f)
        if e is None: rows.append(dict(cnt_low=0, cnt_high=0, cnt_total=0, ratio_off=0.0, off_max=0.0, off_max_cl=0.0)); continue
        lo = e[(e.f >= LOW[0]) & (e.f < LOW[1])]; hi = e[(e.f >= HIGH[0]) & (e.f < HIGH[1])]; off = pd.concat([lo, hi])
        rows.append(dict(cnt_low=len(lo), cnt_high=len(hi), cnt_total=len(e), ratio_off=(len(lo) + len(hi)) / (len(e) + 1),
                         off_max=float(off.x.max()) if len(off) else 0.0,
                         off_max_cl=float((off.h / np.maximum(off.base, FLOOR)).max()) if len(off) else 0.0))
    return pd.DataFrame(rows, index=files)
F = feats(E, list(names)); F1 = feats(E[E.t < 1.0], list(names)); F2h = feats(E[E.t >= 1.0], list(names))
chk = pd.read_csv(os.path.join(B27, "feats_bp11k.csv"), index_col=0).reindex(names)
same = all(np.allclose(F[c].values, chk[c].values) for c in ["cnt_low", "cnt_high", "cnt_total", "ratio_off", "off_max"])
L = [f"4-21 수치 (Train 150 · val 40). 2-7 feats_bp11k.csv와 같은가: {same}. b_floor(Train 바탕 중앙 하위 20 %) = {FLOOR:.4f} g",
     f"이벤트 없는 파일: 정상 {int((F.cnt_total[N] == 0).sum())} · 고장 {int((F.cnt_total[A] == 0).sum())} · val {int((F.cnt_total[V] == 0).sum())}"]

# %% EXP-062 구성 3 모델: 바깥 5겹 OOF 여유(단일 분할) + 전체 학습(val 여유) + J 열
Ttr = dict(np.load(os.path.join(CACHE, "jtab_train_11k.npz"))); Tv = dict(np.load(os.path.join(CACHE, "jtab_val_11k.npz")))
Ttr["df"] = float(Ttr["df"]); Tv["df"] = float(Tv["df"])
sub = lambda T, i: {k: (v if np.ndim(v) == 0 else v[i]) for k, v in T.items()}
gid = np.arange(150); ses = np.isin(np.array(tr_names), ["nor_097", "nor_101", "nor_120"]); gid[ses] = np.flatnonzero(ses)[0]
oof = pd.DataFrame(index=tr_names, columns=["margin", "et_logit", "E2", "rule83", "pred"], dtype=float)
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(150), y, gid):
    P = JShrink().fit(sub(Ttr, a), y[a]).parts(sub(Ttr, b))
    oof.iloc[b] = P[["margin", "et_logit", "E2", "rule83", "pred"]].astype(float).values
mfull = JShrink().fit(Ttr, y); Pv = mfull.parts(Tv)
M = pd.Series(np.r_[oof.margin.values, Pv.margin.values], index=names)                 # 결합 여유 (Train = OOF, val = 전체 학습)
RULE = pd.Series(np.r_[(oof.E2 + oof.rule83).values > 0, (Pv.E2 | Pv.rule83).values], index=names)
PRED = pd.Series(np.r_[oof.pred.values, Pv.pred.values], index=names).astype(int)
f2 = lambda t, p: 5 * ((t == 1) & (p == 1)).sum() / (5 * ((t == 1) & (p == 1)).sum() + 4 * ((t == 1) & (p == 0)).sum() + ((t == 0) & (p == 1)).sum())
L.append(f"\nEXP-062 구성 3 단일 바깥 5겹 OOF (seed 0, 그룹 CV): F2 {f2(y, oof.pred.values.astype(int)):.3f}, 놓친 고장 "
         + (", ".join(np.array(tr_names)[(y == 1) & (oof.pred.values == 0)]) or "없음") + ", 오경보 " + (", ".join(np.array(tr_names)[(y == 0) & (oof.pred.values == 1)]) or "없음"))
L.append("규칙(E2/8.3k)으로 켜진 Train 파일: " + ", ".join(np.array(tr_names)[RULE.values[:150]]))
order_a = oof.margin[y == 1].sort_values(); order_n = oof.margin[y == 0].sort_values(ascending=False)
LOWM_A = [f for f in order_a.index[:6]]; BORDER_N = [f for f in order_n.index[:6]]
L.append("여유 가장 작은 고장 6: " + ", ".join(f"{f} {order_a[f]:+.2f}{' (규칙)' if RULE[f] else ''}" for f in LOWM_A))
L.append("여유 가장 큰 정상 6 (경계 정상): " + ", ".join(f"{f} {order_n[f]:+.2f}" for f in BORDER_N))
J = pd.DataFrame(Ttr["H"], columns=J_FIXED, index=tr_names)
pk = mfull.pr_.transform(Ttr["L"]); J["H16_peak_h"], J["H17_peak_w"] = pk[:, 0], pk[:, 1]; J["L1_lock"] = Ttr["S"]
X65 = np.load(os.path.join(CACHE, "jtab_train_11k_065.npz")); REF = pd.DataFrame({"max_2_4k(4-19)": X65["MAX24"], "max_8_11k(4-19)": X65["MAX811"],
                                                                                "ET_OOF_logit": oof.et_logit.values, "결합_OOF_여유": oof.margin.values}, index=tr_names)
print(f"모델 {time.time() - T0:.0f}s", flush=True)

# %% 관문 ①–⑤ + 증분 ⑥
rng = np.random.default_rng(0)
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
def ci(a, b, B=2000):
    s = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)]; return np.percentile(s, [2.5, 97.5])
def mix_range(v, frac, B=1000):
    n, a = v[N], v[A]; k = int(round(40 * frac)); out = []
    for _ in range(B):
        s = np.r_[rng.choice(n, 40 - k), rng.choice(a, k)] if k else rng.choice(n, 40); out.append(auc(s, n))
    return np.percentile(out, [5, 95])
def tf(c, v): return np.asarray(v, float) if c == "ratio_off" else np.log1p(np.asarray(v, float))
def loo_ll(Xm, yy):
    ll = []
    for i in range(len(yy)):
        k = np.ones(len(yy), bool); k[i] = False
        mu, sd = Xm[k].mean(0), Xm[k].std(0) + 1e-9
        p = LogisticRegression(C=1.0).fit((Xm[k] - mu) / sd, yy[k]).predict_proba(((Xm[i] - mu) / sd)[None])[0, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6); ll.append(-(yy[i] * np.log(p) + (1 - yy[i]) * np.log(1 - p)))
    return float(np.mean(ll))
def pspear(f, m, yy):
    rf, rm, ry = rankdata(f), rankdata(m), rankdata(yy)
    res = lambda r: r - np.polyval(np.polyfit(rm, r, 1), rm)
    return float(np.corrcoef(res(rf), res(ry))[0, 1])
mtr = oof.margin.values; keep = ~RULE.values[:150]
base_all, base_nr = loo_ll(mtr[:, None], y), loo_ll(mtr[keep, None], y[keep])
L.append(f"\n기준 LOO 로그손실 (여유만): 전체 150 {base_all:.4f} · 규칙 켜진 파일 뺀 {keep.sum()}개 {base_nr:.4f}")
pct = lambda c, f: (F[c].values[N] < F.loc[f, c]).mean() * 100 + 0.5 * (F[c].values[N] == F.loc[f, c]).mean() * 100   # 정상 내 중간 백분위(동점 절반)
ROWS = []; WATCH = list(dict.fromkeys(OUTL + FP62 + BORDER_N))
for key, c in FEATS.items():
    v = F[c].values; n, a = v[N], v[A]; lo, hi = ci(a, n); q99 = np.percentile(n, 99)
    rv = auc(v[V], n); m = [mix_range(v, f) for f in (0, .2, .4)]; env = (min(r[0] for r in m), max(r[1] for r in m))
    r = {j: spearmanr(v[TR], J[j].values)[0] for j in J.columns}; jb = max(r, key=lambda k: abs(r[k]) if np.isfinite(r[k]) else -1)
    rr = {j: spearmanr(v[TR], REF[j].values)[0] for j in REF.columns}
    up = [f for f in WATCH if F.loc[f, c] > q99]
    h1, h2 = F1[c].values, F2h[c].values
    a1, a2 = auc(h1[A], h1[N]), auc(h2[A], h2[N]); hs_n = spearmanr(h1[N], h2[N])[0]; hs_t = spearmanr(h1[TR], h2[TR])[0]
    fv = tf(c, v[TR]); d_all = loo_ll(np.column_stack([mtr, fv]), y) - base_all; d_nr = loo_ll(np.column_stack([mtr[keep], fv[keep]]), y[keep]) - base_nr
    pr = pspear(v[TR], mtr, y); co = LogisticRegression(C=1.0).fit(np.column_stack([(mtr - mtr.mean()) / mtr.std(), (fv - fv.mean()) / (fv.std() + 1e-9)]), y).coef_[0]
    inc = "있음" if (d_all < -.005 and d_nr < -.005 and pr > .1) else ("약함" if (d_all < -.005 or d_nr < -.005) else "없음")
    ok = dict(CI=lo > .6, 중복=abs(r[jb]) < .8, val이동=env[0] <= rv <= env[1], 경계정상=not up, 반쪽=(a1 > .5) and (a2 > .5) and (hs_n >= .5))
    ROWS.append(dict(피처=key, 열=c, AUC=round(auc(a, n), 3), CI_lo=round(lo, 3), CI_hi=round(hi, 3),
                     최대중복=f"{jb} {r[jb]:+.2f}", 최대중복_abs=round(abs(r[jb]), 3), rho_ETlogit=round(rr["ET_OOF_logit"], 2), rho_여유=round(rr["결합_OOF_여유"], 2),
                     rho_max24=round(rr["max_2_4k(4-19)"], 2), rho_max811=round(rr["max_8_11k(4-19)"], 2),
                     val_AUC=round(rv, 3), 기준0=f"{m[0][0]:.2f}–{m[0][1]:.2f}", 기준20=f"{m[1][0]:.2f}–{m[1][1]:.2f}", 기준40=f"{m[2][0]:.2f}–{m[2][1]:.2f}",
                     정상99=round(q99, 2), 경계정상_99초과=" ".join(f[-3:] for f in up) or "-", val_정상99초과=" ".join(x[-3:] for x in names[V & (v > q99)]) or "-",
                     반쪽AUC=f"{a1:.2f}/{a2:.2f}", 반쪽rho_정상=round(hs_n, 2), 반쪽rho_Train=round(hs_t, 2),
                     dLL_전체=round(d_all, 4), dLL_규칙뺌=round(d_nr, 4), 부분rho=round(pr, 3), 계수_여유_피처=f"{co[0]:+.2f}/{co[1]:+.2f}",
                     **{f"관문_{k}": bool(t) for k, t in ok.items()}, 관문통과=all(ok.values()), 증분=inc))
G = pd.DataFrame(ROWS).set_index("피처"); G.to_csv(os.path.join(FIG, "관문.csv"), encoding="utf-8-sig")
SHOW = list(dict.fromkeys(HARD_A + LOWM_A)) + WATCH
PCT = pd.DataFrame({c: [round(pct(c, f)) for f in SHOW] for c in FEATS.values()}, index=SHOW)
PCT.insert(0, "OOF여유", [round(M[f], 2) for f in SHOW]); PCT.insert(1, "규칙", ["규칙" if RULE[f] else "" for f in SHOW])
VAL = pd.DataFrame({c: [F.loc[f, c] for f in SHOW] for c in FEATS.values()}, index=SHOW).round(2)
RHO = pd.DataFrame(spearmanr(F.loc[TR, list(FEATS.values())].values)[0], index=FEATS.values(), columns=FEATS.values()).round(2)
L += ["\n관문 표", G.T.to_string(), "\n어려운 파일의 정상 내 백분위 (OOF여유 = 바깥 5겹 결합 여유, 규칙 = OOF E2/8.3k 켜짐)", PCT.to_string(),
      "\n원래 값", VAL.to_string(), "\n5피처끼리 Spearman ρ (Train 150)", RHO.to_string(),
      f"\n참고: off_max 보정 없음 AUC {auc(F.off_max.values[A], F.off_max.values[N]):.3f}"]
for c in FEATS.values():
    L.append(f"{c}: 고장 중 정상 99 % 초과 = " + (", ".join(names[A & (F[c].values > np.percentile(F[c].values[N], 99))]) or "없음")
             + f" | 정상 값 0 비율 {np.mean(F[c].values[N] == 0) * 100:.0f} % · 고장 {np.mean(F[c].values[A] == 0) * 100:.0f} %")
L.append(f"\n걸린 시간 {time.time() - T0:.0f}s")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
F.assign(group=grp, OOF여유=M.values, 규칙=RULE.values, 판정=PRED.values).to_csv(os.path.join(FIG, "files.csv"), encoding="utf-8-sig")

# %% 그림: 피처마다 (왼) 분포 정상 · 고장 · val, ◆ 이상치 (오) EXP-062 OOF 여유 × 피처
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"; C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}
LAB = {"normal": "정상 120", "abnormal": "고장 30", "val": "val 40"}
NAMED = list(dict.fromkeys(HARD_A + LOWM_A[:4] + FP62 + BORDER_N[:3]))
for key, c in FEATS.items():
    D = os.path.join(HERE, key, "figs"); os.makedirs(D, exist_ok=True)
    disc = c.startswith("cnt"); yv = lambda f: tf(c, F.loc[f, c]) if c != "ratio_off" else F.loc[f, c]
    Y = tf(c, F[c].values) if c != "ratio_off" else F[c].values
    jit = rng.uniform(-.25, .25, len(names)); yj = Y + (rng.uniform(-.06, .06, len(names)) if disc else 0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.25]))
    for a_ in ax:
        a_.set_facecolor(SURF); a_.grid(True, color=GRID); a_.set_axisbelow(True)
        for s in ("top", "right"): a_.spines[s].set_visible(False)
    o = np.isin(names, OUTL)
    for k, g in enumerate(("normal", "abnormal", "val")):
        mk = (grp == g) & ~o
        ax[0].scatter(k + jit[mk], yj[mk], s=14, color=C[g], alpha=.6, lw=0)
        ax[0].hlines(np.median(Y[grp == g]), k - .32, k + .32, color=INK, lw=2)
    ax[0].scatter(jit[o], yj[o], marker="D", s=46, color=C["normal"], edgecolor=INK, lw=.8, zorder=3)
    for f in NAMED:
        k = 1 if f.startswith("abn") else 0; i = pos[f]
        ax[0].annotate(f[:1].upper() + f[-3:], (k + jit[i], yj[i]), xytext=(4, 2), textcoords="offset points", fontsize=7.5, color=INK)
    ax[0].axhline(tf(c, np.percentile(F[c].values[N], 99)) if c != "ratio_off" else np.percentile(F[c].values[N], 99), color=C["normal"], ls="--", lw=1)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([LAB[g] for g in ("normal", "abnormal", "val")])
    ax[0].set_ylabel("ln(1 + 값)" if c != "ratio_off" else "값", color=INK2)
    r = G.loc[key]; ax[0].set_title(f"{TITLE[c]}\nAUC {r.AUC:.2f} [{r.CI_lo:.2f}, {r.CI_hi:.2f}] · val AUC {r.val_AUC:.2f} · 관문 {'통과' if r.관문통과 else '미통과'}", loc="left", fontsize=9.5, color=INK)
    # 오른쪽: x = EXP-062 결합 여유 (Train OOF · val 전체 학습), y = 피처
    for g, mk_, s_ in (("normal", "s", 18), ("abnormal", "o", 30), ("val", "^", 30)):
        mk = (grp == g) & ~o
        ax[1].scatter(M.values[mk], yj[mk], marker=mk_, s=s_, color=C[g], alpha=.7, lw=.3, edgecolor="k")
    ax[1].scatter(M.values[o], yj[o], marker="D", s=50, color=C["normal"], edgecolor=INK, lw=.8, zorder=3)
    rl = RULE.values & (grp == "abnormal")
    ax[1].scatter(M.values[rl], yj[rl], s=110, facecolor="none", edgecolor="#7a3db8", lw=1.1, zorder=4)
    for f in NAMED: i = pos[f]; ax[1].annotate(f[:1].upper() + f[-3:], (M.values[i], yj[i]), xytext=(4, 2), textcoords="offset points", fontsize=7.5)
    ax[1].axvline(0, color=INK, ls="--", lw=1)
    ax[1].set_xlabel("EXP-062 구성 3 결합 여유 (Train = 바깥 5겹 OOF, val = 전체 학습)  → 고장 쪽", color=INK2)
    ax[1].set_title(f"여유를 통제한 증분: ΔLL 전체 {r.dLL_전체:+.4f} · 규칙 뺌 {r.dLL_규칙뺌:+.4f} · 부분 ρ {r.부분rho:+.2f} → {r.증분}", loc="left", fontsize=9.5, color=INK)
    ax[1].legend(handles=[Line2D([], [], marker="s", ls="", color=C["normal"], label="정상"), Line2D([], [], marker="o", ls="", color=C["abnormal"], label="고장"),
                          Line2D([], [], marker="^", ls="", color=C["val"], label="val"), Line2D([], [], marker="D", ls="", color=C["normal"], mec=INK, label="◆ 이상치 059·065·116"),
                          Line2D([], [], marker="o", ls="", mfc="none", mec="#7a3db8", label="규칙(E2/8.3k) 켜진 고장")], fontsize=8, loc="upper left")
    fig.text(.01, .01, "검은 선 = 중앙값, 파란 점선 = 정상 99 %. " + ("정수 값은 세로로 조금 흩뜨림. " if disc else "") + "이름: A = 고장(어려운 것 · 여유 작은 것), N = EXP-062 오경보 · 경계 정상. 검출 = 2-7 (1–11 kHz 대역통과)", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(os.path.join(D, "분포.png"), dpi=130, facecolor=SURF); plt.close(fig)
print(f"done {time.time() - T0:.0f}s")
