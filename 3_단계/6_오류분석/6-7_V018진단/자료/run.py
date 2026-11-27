# %% [markdown]
# 6-8 (11-22) V018 진단 — 기준 구성 J-축소(EXP-050)에서 왜 경계(여유 −0.72, 바깥 모델 25개 중 12개 고장)에 있는가. 6-6(V025)과 같은 틀
# - 비교 파일 = 기본 t-SNE 공간(t-SNE 전 거리, 라벨 없는 공간)에서 V018과 가장 가까운 train 정상 2개 · 고장 2개 (라벨은 비교 대상 고르는 데만)
# - ① 결합 분해(ET · 박자 · AE · 절편 − 문턱) ② ET를 민 피처(그 열을 학습 정상 중앙값으로 바꿨을 때 ET logit 변화) ③ 충격 목록 · 간격 · 주기 잠금
#   ④ AE 대역 오차(학습 정상 백분위) ⑤ 가까운 train 5개(기본 t-SNE 공간, t-SNE 전) ⑥ 폴드 모델 5개 + 전체 모델의 여유 범위
# - val 025 = train 150 전체 학습 모델, train 비교 파일 = 자기 빼고 149개로 학습한 모델(OOF와 같은 조건)
# 실행: QE_FULL=1 conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import md_table
from src.evaluate import FULL
from src.exp2util import NS, rule83_flags, ae_map_cols, ae_map_cols_val, tsne_space_val, AE_MAP_COLS, NOR, ABN
from src.models2 import TwoStack, lg
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
from src.impacts import all_impacts
from src.features2 import lock_S
assert FULL, "QE_FULL=1로 실행 (EXP-050과 같은 ET 300)"
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
INK, MUTED, GRID, VAL = "#1d2430", "#5b6575", "#d9dde3", "#8a63d2"
ALPHA, FS = 3.0, 25000
RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
ns = NS(); y, g, n = ns.y, ns.g, ns.n; names = np.array(ns.names); nor = y == 0; nvn = [f"val_{v[-3:]}" for v in ns.nv]; F0 = list(ns.FIXED)
M = np.hstack([ns.T[F0].values, ns.T[["L1_S"]].values, ns.L]); Mv = np.hstack([ns.Tv[F0].values, ns.Tv[["L1_S"]].values, ns.Lv])
Fn = np.column_stack([M, np.arange(n, dtype=float)])
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
TGT = "val_018"; kV = nvn.index(TGT)
_Zt, _Zv, _keep, _ = tsne_space_val(ns); _nk = names[_keep]; _yk = y[_keep]; _dd = np.sqrt(((_Zt - _Zv[kV]) ** 2).sum(1))
CMP = [str(_nk[j]) for j in np.argsort(np.where(_yk == 0, _dd, np.inf))[:2]] + [str(_nk[j]) for j in np.argsort(np.where(_yk == 1, _dd, np.inf))[:2]]
ALL = [TGT] + CMP; print("비교 파일", CMP, flush=True)
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")


class Shrink:
    """EXP-050 J-축소와 같은 축소 + TwoStack(J). 행 = [M | 파일 번호]"""
    def __init__(self, seed=0): self.ri, self.i1, self.seed = [F0.index(c) for c in RING], F0.index("H1_nimp"), seed
    def _tf(self, F):
        F = np.array(F, float); k = np.rint(np.expm1(F[:, self.i1])); has = k > 0
        F[np.ix_(has, self.ri)] = (F[np.ix_(has, self.ri)] * k[has, None] + ALPHA * self.p0_) / (k[has, None] + ALPHA); return F
    def fit(self, F, y):
        F = np.asarray(F, float); k = np.rint(np.expm1(F[:, self.i1])); m = (y == 0) & (k > 0)
        self.p0_ = (F[np.ix_(m, self.ri)] * k[m, None]).sum(0) / k[m].sum()
        self.m_ = TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True, seed=self.seed).fit(self._tf(F), y); return self


def explain(sh, row, ae_raw=None, idx=None):
    """sh = 학습된 Shrink, row = M 한 행(번호 없음). 결합 분해 + ET 피처 기여(그 열을 학습 정상 중앙값으로 → ET logit 변화)"""
    ts = sh.m_; r = sh._tf(row[None, :])
    if idx is None: f = ts.fires_raw(r, {"ae": np.atleast_1d(ae_raw)})
    else: f = ts.fires(np.column_stack([r, [[idx]]]))
    a, z, ae = float(f["a"][0]), float(f["z"][0]), float(f["ae"][0]); c, b, t = ts.c_.coef_[0], ts.c_.intercept_[0], ts.tc_
    H = ts._parts(r)[0]; cols = list(ts.fixed) + list(ts.peak); con = []
    for j, cn in enumerate(cols):
        h = H.copy(); h[0, j] = np.median(ts.H_fit_[ts.y_fit_ == 0, j])
        con.append((a - float(lg(ts.m_.predict_proba(h)[:, 1])[0]), cn, float(H[0, j]), float((ts.H_fit_[ts.y_fit_ == 0, j] < H[0, j]).mean() * 100),
                    float((ts.H_fit_[ts.y_fit_ == 1, j] < H[0, j]).mean() * 100)))
    con.sort(key=lambda q: -abs(q[0]))
    return dict(ET=c[0] * a, 박자=c[1] * z, AE=c[2] * ae, 절편=b, 문턱=t, 여유=c[0] * a + c[1] * z + c[2] * ae + b - t, a=a, z=z, ae=ae,
                E2=bool(f["E2"][0]), con=con, coef=c)


# %% ① ② 결합 분해 · ET 기여
full = Shrink().fit(Fn, y); EX = {TGT: explain(full, Mv[kV], ae_raw=RAW["ae"][kV])}
for nm in CMP:
    i = list(names).index(nm); tr = np.arange(n) != i
    EX[nm] = explain(Shrink().fit(Fn[tr], y[tr]), M[i], idx=i)
    print(nm, round(EX[nm]["여유"], 2), flush=True)

# %% ③ 충격 · 간격 · 주기 잠금
IMv = all_impacts(ns.Xv[[kV]])[0]; IMt = all_impacts(ns.X[[list(names).index(c) for c in CMP]])
IM = {TGT: IMv, **dict(zip(CMP, IMt))}; SIG = {TGT: ns.Xv[kV], **{c: ns.X[list(names).index(c)] for c in CMP}}
def evts(t, gap=.020):
    o = np.sort(t); ev = [[o[0]]] if len(o) else []
    for v in o[1:]: (ev[-1].append(v) if v - ev[-1][-1] < gap else ev.append([v]))
    return np.array([e[0] for e in ev])
LK = {}
for nm in ALL:
    im = IM[nm]; S_, r_, T_, hit = lock_S(SIG[nm], im["t"]); et = evts(im["t"]); iv = np.diff(et)
    LK[nm] = dict(충격=len(im["t"]), 사건=len(et), 사건시점=" · ".join(f"{v:.3f}" for v in et), 간격=" · ".join(f"{v:.3f}" for v in iv),
                  간격CV=round(float(iv.std() / iv.mean()), 3) if len(iv) >= 2 else np.nan, 울림kHz=" · ".join(f"{v / 1000:.1f}" for v in np.asarray(im["rings"])[np.argsort(im["t"]), 0]),
                  세기g=round(float(np.max(im["height_g"])), 3), 주기잠금_r=round(r_, 2), 주기_T=round(T_, 3), hit=round(hit, 2), LOCK_S=round(S_, 2))
# 주기 잠금 S: train 정상 · 고장 분포
IMall = all_impacts(ns.X); Sall = np.array([lock_S(ns.X[i], IMall[i]["t"])[0] for i in range(n)])

# %% ④ AE 대역 · ⑤ 이웃 · ⑥ 폴드 모델 여유
At, Av = ae_map_cols(ns), ae_map_cols_val(ns)
Zt, Zv, keep, _ = tsne_space_val(ns); nk = names[keep]; yk = y[keep]
dd = np.sqrt(((Zt - Zv[kV]) ** 2).sum(1)); o5 = np.argsort(dd)[:5]
WOB = []
for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Fn, y, g):
    WOB.append(explain(Shrink().fit(Fn[tr], y[tr]), Mv[kV], ae_raw=RAW["ae"][kV])["여유"])
WOB.append(EX[TGT]["여유"])

# %% ⑦ 축소 효과: 축소하지 않은 J(EXP-044)로 같은 분해 · V018 울림 비율(축소 전 → 축소 뒤) · 촘촘한 울림 주파수(2-29 csv)
class NoShrink(Shrink):
    def _tf(self, F): return np.array(F, float)
    def fit(self, F, y):
        self.m_ = TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True).fit(np.asarray(F, float), y); return self
EJ = explain(NoShrink().fit(Fn, y), Mv[kV], ae_raw=RAW["ae"][kV])
ri = [F0.index(c) for c in RING]; raw_r = Mv[kV, ri]; shr_r = full._tf(Mv[[kV]])[0, ri]
_csv = os.path.join(_d, "3_단계", "2_EDA", "아카이브", "2-29_1122_울림대역경계", "figs", "impacts_fine.csv")
FINE = pd.read_csv(_csv, encoding="utf-8-sig").query("file == @TGT") if os.path.isfile(_csv) else pd.DataFrame()

# %% 표 (md)
rows = []
for nm in ALL:
    e = EX[nm]; rows.append({"파일": short(nm), "라벨": {"val": "val(없음)", "nor": "정상", "abn": "고장"}[nm[:3]], "ET 몫": round(e["ET"], 2), "박자 몫": round(e["박자"], 2),
                             "AE 몫": round(e["AE"], 2), "절편": round(e["절편"], 2), "문턱": round(e["문턱"], 2), "여유": round(e["여유"], 2), "E2": "켜짐" if e["E2"] else "",
                             "8.3k": "켜짐" if (r83v[kV] if nm.startswith("val") else r83[list(names).index(nm)]) else ""})
T1 = pd.DataFrame(rows)
T2 = pd.DataFrame([{"파일": short(nm), **LK[nm]} for nm in ALL])
T3 = pd.DataFrame([{"피처": cn, "ET logit 기여": round(v, 2), "V018 값": round(x, 3), "정상 백분위": int(pn), "고장 백분위": int(pa)} for v, cn, x, pn, pa in EX[TGT]["con"][:8]])
T4 = pd.DataFrame([{"대역": c, "V018": round(float(Av[kV, j]), 2), "학습 정상 백분위": int((At[nor, j] < Av[kV, j]).mean() * 100), "학습 고장 백분위": int((At[~nor, j] < Av[kV, j]).mean() * 100)} for j, c in enumerate(AE_MAP_COLS)])
nb = " · ".join(f"{short(nk[j])}({dd[j]:.1f})" for j in o5)
txt = ["# 6-8 V018 진단 표 (J-축소, EXP-050과 같은 설정)\n", "## ① 결합 분해 (여유 = ET + 박자 + AE + 절편 − 문턱, 0 이상 = 결합 고장)\n", md_table(T1),
       f"\nJ-축소 전체 모델 결합 계수 ET {EX[TGT]['coef'][0]:.2f} · 박자 {EX[TGT]['coef'][1]:.2f} · AE {EX[TGT]['coef'][2]:.2f}. train 비교 파일은 자기 빼고 149개로 학습한 모델\n",
       "## ③ 충격 · 간격 · 주기 잠금\n", md_table(T2),
       f"\n주기 잠금 S(= r × hit) 분포: train 정상 최대 {Sall[nor].max():.2f}(상위 3개 {', '.join(f'{short(names[i])} {Sall[i]:.2f}' for i in np.argsort(-np.where(nor, Sall, -1))[:3])}) · train 고장 중앙 {np.median(Sall[~nor]):.2f}, 고장 최대 {Sall[~nor].max():.2f}({short(names[np.argmax(np.where(~nor, Sall, -1))])})\n",
       "## ② V018 ET를 민 피처 (그 열을 학습 정상 중앙값으로 바꿨을 때 ET logit이 줄어드는 양, 큰 순 8개)\n", md_table(T3),
       "\n## ④ AE 대역 오차 (log10, 학습 정상 · 고장 백분위)\n", md_table(T4),
       f"\n## ⑤ 가까운 train 5개 (기본 t-SNE 공간, t-SNE 전 거리)\n\n{nb} → 고장 {int(yk[o5].sum())}/5\n",
       f"\n## ⑥ 여유 범위 (바깥 5폴드 모델 + 전체 모델)\n\n" + " · ".join(f"{v:+.2f}" for v in WOB) + f" → 고장 판정 {int((np.array(WOB) >= 0).sum())}/6\n",
       "\n## ⑦ 축소 효과 (J → J-축소)\n",
       f"J(축소 없음): ET {EJ['ET']:+.2f} · 박자 {EJ['박자']:+.2f} · AE {EJ['AE']:+.2f} · 절편 {EJ['절편']:+.2f} − 문턱 {EJ['문턱']:.2f} = 여유 {EJ['여유']:+.2f} → J-축소 여유 {EX[TGT]['여유']:+.2f}\n",
       "울림 비율 H7 1–2k · H8 2–4k · H9 4–8k · H10 8–12.5k: 축소 전 " + " · ".join(f"{v:.2f}" for v in raw_r) + " → 축소 뒤 " + " · ".join(f"{v:.2f}" for v in shr_r) + "\n",
       ("촘촘한 울림 주파수(2-29, 50 Hz 맞춤 · FFT 봉우리): " + " · ".join(f"{a:.0f}/{b:.0f}" for a, b in zip(FINE.f50, FINE.ffft)) + " Hz\n") if len(FINE) else ""]
open(os.path.join(HERE, "표_V018.md"), "w", encoding="utf-8").write("\n".join(txt))

# %% 그림: 위 = 파형(200 Hz 고역) + 충격 표시 · 아래 왼쪽 = 결합 분해 · 아래 가운데 = PSD · 아래 오른쪽 = ET 기여
sos = butter(4, 200, btype="highpass", fs=FS, output="sos"); tt = np.arange(50000) / FS
fig = plt.figure(figsize=(17, 13.5)); gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.25], hspace=.6, wspace=.32)
for j, nm in enumerate(ALL):
    ax = fig.add_subplot(gs[0, :] if j == 0 else gs[1 + (j - 1) // 2, ((j - 1) % 2) * 2:((j - 1) % 2) * 2 + 2]); h = sosfiltfilt(sos, SIG[nm] - SIG[nm].mean())
    col = VAL if nm.startswith("val") else (NOR if nm.startswith("nor") else ABN)
    ax.plot(tt, h, color=col, lw=.4); im = IM[nm]
    for t_, r_ in zip(im["t"], np.asarray(im["rings"])[:, 0]):
        ax.axvline(t_, color=INK, lw=.6, ls=":", alpha=.6); ax.text(t_, ax.get_ylim()[1] * .92, f"{r_ / 1000:.1f}k", fontsize=7, ha="center", color=INK)
    lk = LK[nm]; e = EX[nm]
    ax.set_title(f"{short(nm)} ({ {'val': 'val, 라벨 없음', 'nor': '정상', 'abn': '고장'}[nm[:3]] }) — 충격 {lk['충격']}개 · 사건 간격 {lk['간격']} s · 여유 {e['여유']:+.2f}", loc="left", fontsize=9.5)
    ax.set_xlim(0, 2); ax.tick_params(labelsize=7, colors=MUTED); ax.grid(alpha=.2)
    if j >= 3: ax.set_xlabel("시간 (s), 200 Hz 고역통과, 점선 = 충격 · 숫자 = 울림 주파수", fontsize=8, color=MUTED)
ax = fig.add_subplot(gs[3, 0]); parts = ["ET", "박자", "AE", "절편"]; xs = np.arange(len(ALL)); bot = np.zeros(len(ALL))
cols_ = {"ET": "#d4561f", "박자": "#8a63d2", "AE": "#2a78d6", "절편": "#9aa3b0"}
for p in parts:
    v = np.array([EX[nm][p] for nm in ALL]); ax.bar(xs, v, bottom=np.where(v >= 0, np.maximum(bot, 0), np.minimum(bot, 0)), color=cols_[p], width=.6, label=p)
    bot = bot + v
ax.scatter(xs, [EX[nm]["문턱"] for nm in ALL], marker="_", s=600, color=INK, lw=2.5, zorder=4, label="문턱")
ax.scatter(xs, [EX[nm]["ET"] + EX[nm]["박자"] + EX[nm]["AE"] + EX[nm]["절편"] for nm in ALL], marker="o", s=40, color="white", edgecolor=INK, zorder=5, label="결합 점수")
ax.set_xticks(xs); ax.set_xticklabels([short(a) for a in ALL]); ax.axhline(0, color=GRID); ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_title("① 결합 점수의 몫 (점이 막대 위 선보다 위면 결합 고장)", loc="left", fontsize=9); ax.tick_params(labelsize=8)
ax = fig.add_subplot(gs[3, 1:3]); fb = np.arange(1025) * FS / 2048; Lt = LP_TR.mean(1); Lv = LPv.mean(1)
ax.fill_between(fb, np.percentile(Lt[nor], 5, 0), np.percentile(Lt[nor], 95, 0), color=NOR, alpha=.18, label="train 정상 5–95 %")
ax.plot(fb, np.median(Lt[~nor], 0), color=ABN, lw=1, label="train 고장 중앙")
ax.plot(fb, np.median(Lv, 0), color=MUTED, lw=1, ls="--", label="val 40 중앙")
for nm, ls in zip(CMP, (":", "-.", ":", "-.")):
    ax.plot(fb, Lt[list(names).index(nm)], color=NOR if nm.startswith("nor") else ABN, lw=.9, ls=ls, label=short(nm))
ax.plot(fb, Lv[kV], color=VAL, lw=1.6, label="V018")
ax.set_xlim(0, 12500); ax.set_xlabel("주파수 (Hz)", fontsize=8, color=MUTED); ax.set_ylabel("log PSD (구간 평균)", fontsize=8, color=MUTED)
ax.legend(fontsize=7, ncol=2); ax.tick_params(labelsize=7); ax.grid(alpha=.2); ax.set_title("④ 스펙트럼 (AE가 보는 것)", loc="left", fontsize=9)
ax = fig.add_subplot(gs[3, 3]); con = EX[TGT]["con"][:8][::-1]
ax.barh([c[1].split("_")[0] + " " + "_".join(c[1].split("_")[1:])[:10] for c in con], [c[0] for c in con], color=[ABN if c[0] > 0 else NOR for c in con])
ax.axvline(0, color=INK, lw=.8); ax.tick_params(labelsize=7); ax.set_title("② V018의 ET를 민 피처 (+ = 고장 쪽)", loc="left", fontsize=9)
fig.suptitle(f"V018 진단 (J-축소) — V018과, 기본 t-SNE 공간(t-SNE 전 거리)에서 가장 가까운 train 정상 2개 · 고장 2개 ({' · '.join(short(c) for c in CMP)})", x=.01, ha="left", fontsize=11)
fig.savefig(os.path.join(FIG, "V018_진단.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("done", flush=True)
