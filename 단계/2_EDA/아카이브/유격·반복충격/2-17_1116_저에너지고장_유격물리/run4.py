# %% [markdown]
# 2-17 4차 — 되튐(chatter)을 직접 잰다
# 3차 결론: 같은 하중 사건 안의 타격을 따로 셀수록 고장이 잘 갈림 → "한 번 닫힐 때 여러 번 되튄다"가 지문일 가능성
# 물리 (반발): 반발계수 e < 1 → 되튈 때마다 속도가 e배 → 다음 충격까지 간격 e배, 충격 세기(≈ 충돌 속도) e배
#   → 연속 충격 3개에서 "간격 감소 + 높이 감소"가 함께 나타난다 (간격이 줄며 잦아드는 따닥따닥)
#   → 진짜 되튐이면 간격 비 Δ2/Δ1 과 높이 비 A2/A1 이 같은 e로 비슷해야 한다 (라벨 없는 물리 검증)
# 정의 (라벨 보기 전에 고정)
#   σ    = 200 Hz 고역 h의 바탕 MAD (충격 ±10 ms 가림, 2·3차와 같음)
#   포락선 = |hilbert(h)|, 0.5 ms 이동평균
#   충격 = 포락선 봉우리 > kσ, 1 ms 이상 떨어짐, 직전 충격과의 사이 골짜기보다 ρ배 이상 다시 솟음 (아니면 같은 충격의 울림)
#          → 10 ms로 묶지 않는다 (되튐 간격이 수 ms일 수 있으므로)
#   삼중 = 연속 충격 3개, 첫 간격 Δ1 ≤ Tmax
#   연쇄 = Δ2 < Δ1 이고 A1 > A2 > A3
#   CH   = 연쇄 수 / 삼중 수, 귀무 = 파일 안에서 높이 순서를 섞은 200번의 평균 → CH_ex = CH − 귀무 (우연을 뺀 되튐 비율)
# 견고성: k ∈ {6, 8, 12}, ρ ∈ {1.5, 2, 3}, Tmax ∈ {25, 50, 100} ms
# 실행: conda run -n qe-phm python "단계/2_EDA/아카이브/유격·반복충격/2-17_1116_저에너지고장_유격물리/run4.py"

# %% 설정
import os, sys, itertools
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.features import feature_table_knock
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs4"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = np.array([os.path.splitext(os.path.basename(str(n)))[0] for n in names])
lab = lambda s: s.replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

GAP = int(0.010 * FS)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
ENV = np.abs(hilbert(H, axis=-1))
ker = np.ones(12) / 12
ENV = np.array([np.convolve(e, ker, "same") for e in ENV])
K1 = feature_table_knock(X).K1_second_snr.values

def sigma(h, k=8):
    s0 = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    m = np.ones(len(h), bool)
    for p in np.flatnonzero(np.abs(h) > k * s0):
        m[max(p - GAP, 0):p + GAP] = False
    hb = h[m] if m.sum() > len(h) // 4 else h
    return 1.4826 * np.median(np.abs(hb - np.median(hb))) + 1e-12
SIG = np.array([sigma(h) for h in H])
QUIET = np.zeros(N, bool); QUIET[np.argsort(SIG)[:40]] = True

def impacts(env, sig, k, rho):
    cand, _ = find_peaks(env, height=k * sig, distance=25)
    keep = []
    for p in cand:
        if not keep:
            keep.append(p); continue
        q = keep[-1]
        valley = env[q:p + 1].min()
        if env[p] >= rho * valley and env[q] >= rho * valley:
            keep.append(p)                      # 골짜기 뒤 다시 솟음 = 새 충격
        elif env[p] > env[q]:
            keep[-1] = p                        # 같은 충격, 더 높은 봉우리로
    return np.array(keep, int)

RNG = np.random.default_rng(0)
def chains(pk, A, tmax):
    """연속 삼중 중 연쇄 비율, 귀무(높이 순서 섞기) 평균, 연쇄 목록"""
    if len(pk) < 3:
        return 0, np.nan, np.nan, []
    d = np.diff(pk) / FS
    ok = d[:-1] <= tmax
    if not ok.any():
        return 0, np.nan, np.nan, []
    i = np.flatnonzero(ok)
    dec_t = d[i + 1] < d[i]
    def ch(a):
        return dec_t & (a[i] > a[i + 1]) & (a[i + 1] > a[i + 2])
    c = ch(A)
    nul = np.mean([ch(RNG.permutation(A)).mean() for _ in range(200)])
    lst = [(pk[j], d[j], d[j + 1], A[j], A[j + 1], A[j + 2]) for j in i[c]]
    return int(ok.sum()), float(c.mean()), float(nul), lst

def run(k=8, rho=2, tmax=0.05, keep_list=False):
    rows, CL = [], []
    for n in range(N):
        pk = impacts(ENV[n], SIG[n], k, rho)
        A = ENV[n][pk] / SIG[n]
        nt, ch, nul, lst = chains(pk, A, tmax)
        rows.append(dict(file=short[n], y=int(y[n]), n_imp=len(pk), n_triple=nt, n_chain=len(lst),
                         CH=ch, CH_null=nul, CH_ex=(ch - nul) if nt else np.nan))
        if keep_list:
            CL += [dict(file=short[n], y=int(y[n]), t=a / FS, d1_ms=b * 1e3, d2_ms=c * 1e3, A1=e, A2=f, A3=g) for a, b, c, e, f, g in lst]
    return pd.DataFrame(rows), pd.DataFrame(CL)

def auc(v):
    v = np.asarray(v, float); v = np.where(np.isnan(v), np.nanmin(v) - 1 if np.isfinite(np.nanmin(v)) else 0, v)
    return roc_auc_score(y, v)

# %% 기본
D, C = run(keep_list=True)
D["K1"] = K1; D["quiet"] = QUIET; D["log_chain"] = np.log1p(D.n_chain)
D.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")
C.to_csv(os.path.join(FIG, "chains.csv"), index=False, encoding="utf-8-sig")
say("== 기본 (k=8, ρ=2, Tmax=50 ms) ==")
for g, gn in [(0, "정상"), (1, "고장")]:
    d = D[D.y == g]
    say(f"{gn}: 충격 중앙 {d.n_imp.median():.0f} · 삼중 있는 파일 {int((d.n_triple > 0).sum())}/{len(d)} · 연쇄 ≥1 파일 {int((d.n_chain > 0).sum())}"
        f" · 연쇄 수 합 {int(d.n_chain.sum())} · CH 중앙 {d.CH.median():.3f} · 귀무 중앙 {d.CH_null.median():.3f} · CH_ex 중앙 {d.CH_ex.median():.3f}")
say(f"AUC: n_chain {auc(D.n_chain):.3f} · CH {auc(D.CH):.3f} · CH_ex {auc(D.CH_ex):.3f} · K1(현행) {auc(K1):.3f}")
q = QUIET
say(f"조용한 40 AUC: n_chain {roc_auc_score(y[q], D.n_chain[q]):.3f} · K1 {roc_auc_score(y[q], K1[q]):.3f}")

# 물리 검증: 간격 비 vs 높이 비 (반발계수가 같다면 비슷)
say("\n== 물리 검증: 연쇄에서 간격 비 Δ2/Δ1 과 높이 비 A2/A1 (같은 e라면 상관 +, 값 비슷) ==")
for g, gn in [(0, "정상"), (1, "고장")]:
    c = C[C.y == g]
    if len(c) >= 5:
        et, ea = c.d2_ms / c.d1_ms, c.A2 / c.A1
        r = spearmanr(et, ea)[0]
        say(f"{gn}: 연쇄 {len(c)} · e(간격) 중앙 {et.median():.2f} · e(높이) 중앙 {ea.median():.2f} · 순위상관 {r:.2f} · Δ1 중앙 {c.d1_ms.median():.1f} ms")

say("\n== 연쇄 수 상위 30 파일 ==")
say(D.sort_values(["n_chain", "CH_ex"], ascending=False).head(30).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
say("\n== 저에너지 고장 (조용한 40 안) ==")
say(D[QUIET & (y == 1)].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
nm = D[D.y == 0].n_chain
say(f"\n정상 연쇄 수 분위: 50% {nm.quantile(.5):.0f} · 90% {nm.quantile(.9):.0f} · 95% {nm.quantile(.95):.0f} · 최대 {nm.max():.0f}")
say("고장 중 연쇄 수 > 정상 최대: " + " ".join(lab(s) for s in D[(D.y == 1) & (D.n_chain > nm.max())].file))
say("고장 중 연쇄 수 > 정상 95%: " + " ".join(lab(s) for s in D[(D.y == 1) & (D.n_chain > nm.quantile(.95))].file))

# %% 견고성
rows = []
for k, rho, tm in itertools.product([6, 8, 12], [1.5, 2, 3], [0.025, 0.05, 0.1]):
    d, _ = run(k, rho, tm)
    nmax = d[d.y == 0].n_chain.max()
    rows.append(dict(k=k, rho=rho, Tmax_ms=tm * 1e3, auc_n_chain=auc(d.n_chain), auc_CH_ex=auc(d.CH_ex),
                     nor_chain_max=int(nmax), abn_over_nor_max=int(((d.y == 1) & (d.n_chain > nmax)).sum()),
                     abn_chain_ge1=int(((d.y == 1) & (d.n_chain > 0)).sum()), nor_chain_ge1=int(((d.y == 0) & (d.n_chain > 0)).sum())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(FIG, "robust.csv"), index=False, encoding="utf-8-sig")
say("\n== 견고성 27가지 ==")
say(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% 그림 1: 연쇄 예시 (고장·정상 각각 연쇄 많은 파일 3개, 첫 연쇄 주변 60 ms)
ex = list(D[D.y == 1].sort_values("n_chain", ascending=False).file[:3]) + list(D[D.y == 0].sort_values("n_chain", ascending=False).file[:3])
fig, axs = plt.subplots(2, 3, figsize=(16, 7))
for ax, f in zip(axs.ravel(), ex):
    n = int(np.flatnonzero(short == f)[0]); c = C[C.file == f]
    if not len(c):
        ax.set_title(f"{lab(f)} 연쇄 없음"); continue
    t0 = c.t.iloc[0]; a, b = int((t0 - .01) * FS), int((t0 + .05) * FS)
    tt = np.arange(a, b) / FS * 1e3
    ax.plot(tt, H[n, a:b] / SIG[n], color="0.7", lw=.5)
    ax.plot(tt, ENV[n, a:b] / SIG[n], color=ABN if y[n] else NOR, lw=1.2)
    pk = impacts(ENV[n], SIG[n], 8, 2); pk = pk[(pk >= a) & (pk < b)]
    ax.scatter(pk / FS * 1e3, ENV[n, pk] / SIG[n], color="k", zorder=3)
    ax.set_title(f"{lab(f)} 연쇄 {len(c)} · 첫 연쇄 Δ {c.d1_ms.iloc[0]:.1f}→{c.d2_ms.iloc[0]:.1f} ms", color=ABN if y[n] else NOR)
    ax.set_xlabel("ms"); ax.set_ylabel("/ σ")
fig.suptitle("되튐 연쇄 예시 — 회색 = 200 Hz 고역, 색 = 포락선, 점 = 충격 (위 고장 · 아래 정상, 연쇄 많은 순)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_연쇄예시.png"), dpi=80); plt.close(fig)

# %% 그림 2: 연쇄의 간격 비 vs 높이 비
fig, ax = plt.subplots(figsize=(6, 5.5))
for g, c_ in [(0, NOR), (1, ABN)]:
    c = C[C.y == g]
    ax.scatter(c.d2_ms / c.d1_ms, c.A2 / c.A1, s=12, c=c_, alpha=.6, label=f"{'정상' if g == 0 else '고장'} {len(c)}")
ax.plot([0, 1], [0, 1], color="k", ls=":"); ax.set_xlabel("간격 비 Δ2/Δ1"); ax.set_ylabel("높이 비 A2/A1")
ax.set_title("반발이라면 두 비가 같은 e → 대각선 근처"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_간격비_높이비.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
