# %% [markdown]
# 2-15 고역통과 충격 수·Peak 오탐 줄이기 — L3 고장처럼 보이는 정상과 L3 고장을 사건 단위로 비교
# - 대상: L3 고장 12개(2-0) vs "L3처럼 보이는 정상" = 고역통과 충격 수 > 정상 95% 또는 고역통과 Peak > 정상 95%인 정상
# - 사건 정의는 뷰어·2-12와 같음: 200 Hz 고역통과 h, σ = 1.4826·MAD, 사건 = |h| > 8σ, 10 ms 안은 한 사건 (라벨 무관)
# - 사건 지표는 물리 가설에서 정함 (H1: 간극 타격은 하중 주기마다 비슷한 크기로 반복 / 정상 큰 사건 = 노면 한 번)
#   D1 두 번째 사건 높이(σ) · R 2위/1위 · N50 최대의 50% 넘는 사건 수 · SPREAD 사건이 있는 100 ms 창 비율
#   · GAPCV 사건 간격 변동계수 · FC 사건 주파수 중심(상위 5개 사건 ±2 ms) · ARM 2–3.5 kHz 몫(1-3) · DECAY 사건 감쇠 시간
#   · PRE 최고점 전/후 RMS (그림 3을 본 뒤 추가, 사후)
# - 라벨은 색·비교에만 쓴다. 지표 정의는 라벨을 보기 전에 고정했지만, 어떤 지표가 가르는지는 라벨을 보고 판단한다(설명용)
# 실행: conda run -n qe-phm python "단계/2_EDA/아카이브/개관/2-15_1116_고역통과충격수오탐/2-15_run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = next(_p for _p in (os.path.abspath(os.path.join(HERE, *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src")))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert
from src.exp import load_train
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

FS = 25000
X, y, names = load_train(); y = np.asarray(y); nor = y == 0; N = len(y)
T0 = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "2-1_고장유형분류", "figs", "types.csv"), index_col=0, encoding="utf-8-sig").loc[names]
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
GAP, HW, WIN = int(0.010 * FS), int(0.002 * FS), int(0.100 * FS)

# %% 사건 지표
def events(h):
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if not len(idx):
        return sig, np.array([], int), np.array([])
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > GAP) + 1)
    pk = np.array([g[np.argmax(np.abs(h[g]))] for g in gs])
    return sig, pk, np.abs(h[pk]) / sig


rows = []
for i in range(N):
    h = H[i]; sig, pk, a = events(h)
    o = dict(n_imp=len(pk), peak_g=np.abs(h).max(), peak_sig=np.abs(h).max() / sig)
    s = np.sort(a)[::-1]
    o["D1"] = s[1] if len(s) > 1 else 0.0                                   # 두 번째 사건 높이 (σ)
    o["R"] = s[1] / s[0] if len(s) > 1 else 0.0                             # 2위/1위
    o["N50"] = int((a >= 0.5 * a.max()).sum()) if len(a) else 0             # 최대의 절반 넘는 사건 수
    o["SPREAD"] = len(np.unique(pk // WIN)) / (len(h) // WIN) if len(pk) else 0.0
    o["GAPCV"] = np.diff(np.sort(pk)).std() / np.diff(np.sort(pk)).mean() if len(pk) >= 3 else np.nan
    fc, dc, arm = [], [], []
    env = np.abs(hilbert(h))
    for p in pk[np.argsort(-a)][:5]:                                        # 상위 5개 사건
        seg = h[max(p - HW, 0): p + HW] * np.hanning(len(h[max(p - HW, 0): p + HW]))
        P = np.abs(np.fft.rfft(seg, 256)) ** 2; f = np.fft.rfftfreq(256, 1 / FS)
        fc.append((P * f).sum() / P.sum())
        arm.append(P[(f >= 2000) & (f < 3500)].sum() / P.sum())            # 1-3에 미리 적힌 암 고유진동 대역 몫
        e = env[p: p + int(0.02 * FS)]                                      # 20 ms 안에서 최대의 1/e까지 걸린 시간
        below = np.flatnonzero(e < e[0] / np.e)
        dc.append((below[0] if len(below) else len(e)) / FS * 1000)
    # PRE (그림 3을 본 뒤 추가, 사후): 가장 큰 사건의 최고점 전 0.5–3 ms RMS / 후 0.5–3 ms RMS
    #   충돌 = 조용하다 갑자기 시작(작음) / 공진이 커졌다 줄어드는 방추형 = 앞에도 진폭이 있음(1에 가까움)
    p0 = int(np.argmax(np.abs(h))); a0, a1 = int(0.0005 * FS), int(0.003 * FS)
    pre, post = h[max(p0 - a1, 0): max(p0 - a0, 1)], h[p0 + a0: p0 + a1]
    o["PRE"] = np.sqrt((pre ** 2).mean()) / (np.sqrt((post ** 2).mean()) + 1e-12) if len(pre) and len(post) else np.nan
    o["FC"] = np.median(fc) if fc else np.nan
    o["ARM"] = np.median(arm) if arm else np.nan
    o["DECAY"] = np.median(dc) if dc else np.nan
    rows.append(o)
E = pd.DataFrame(rows, index=names)

# 그룹: L3 고장 / L3처럼 보이는 정상 (충격 수 또는 Peak > 정상 95%) / 기타
q95 = lambda k: np.percentile(E.loc[nor, k], 95)
TH_N, TH_P = q95("n_imp"), q95("peak_g")
like = nor & ((E.n_imp > TH_N) | (E.peak_g > TH_P)).values
l3f = (~nor) & (T0.type == "L3").values
G = np.where(l3f, "L3 고장", np.where(like, "L3처럼 보이는 정상", np.where(nor, "기타 정상", "기타 고장")))
E.insert(0, "group", G); E.insert(0, "y", y); E["type2_0"] = T0.type.values; E["fp_exp025"] = T0.fp.values; E["miss_exp025"] = T0.miss.values
E.round(4).to_csv(os.path.join(FIG, "events.csv"), encoding="utf-8-sig")

# %% 그림 1: 지표별 네 그룹 분포 (L3 고장 vs L3처럼 보이는 정상에 이름)
FEATS = [("n_imp", "충격 수", "symlog"), ("peak_g", "Peak (g)", "log"), ("D1", "두 번째 사건 높이 (σ)", "log"), ("R", "2위/1위 사건 높이", None),
         ("N50", "최대의 50% 넘는 사건 수", "symlog"), ("SPREAD", "사건이 있는 100 ms 창 비율", None), ("GAPCV", "사건 간격 변동계수", None),
         ("FC", "사건 주파수 중심 (Hz)", None), ("ARM", "사건 에너지 중 2–3.5 kHz 몫 (1-3 암 고유진동)", None),
         ("DECAY", "사건 감쇠 시간 (ms)", "log"), ("PRE", "최고점 전/후 RMS (사후 추가)", None)]
ORDER = [("L3 고장", "#1baf7a", "s"), ("L3처럼 보이는 정상", "#eb6834", "o"), ("기타 고장", "#9fd9c3", "s"), ("기타 정상", "#c9c8c2", "o")]
fig, axs = plt.subplots(2, 6, figsize=(26, 9)); axs.flat[-1].axis("off")
rng = np.random.default_rng(0)
for ax, (k, lab, sc) in zip(axs.flat, FEATS):
    for j, (g, col, mk) in enumerate(ORDER):
        m = E.group.values == g; v = E.loc[m, k].values
        xj = j + rng.uniform(-.18, .18, m.sum())
        ax.scatter(xj, v, s=(34 if j < 2 else 14), c=col, marker=mk, lw=.6, edgecolors="white", zorder=3 if j < 2 else 2)
        if j < 2:
            for xx, vv, nm in zip(xj, v, E.index[m]):
                if np.isfinite(vv):
                    ax.annotate(nm.split("_")[1], (xx, vv), fontsize=6.5, xytext=(3, 0), textcoords="offset points", color="#3d3d3a", va="center")
    if sc == "log":
        ax.set_yscale("log")
    elif sc == "symlog":
        ax.set_yscale("symlog", linthresh=1)
    if sc:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(range(4)); ax.set_xticklabels([g for g, *_ in ORDER], fontsize=8)
    ax.set_title(lab, fontsize=10, loc="left"); ax.grid(axis="y", alpha=.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle(f"L3 고장 {l3f.sum()}개 vs L3처럼 보이는 정상 {like.sum()}개 (충격 수 > {TH_N:.0f} 또는 Peak > {TH_P:.2f} g) — 사건 지표", fontsize=12, x=.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_event_strips.png"), dpi=120); plt.close(fig)

# %% 요약: 두 그룹을 가르는 정도 (AUC, 설명용) + 값 표
from sklearn.metrics import roc_auc_score
sub = E[(E.group == "L3 고장") | (E.group == "L3처럼 보이는 정상")]
yy = (sub.group == "L3 고장").astype(int)
L = [f"L3처럼 보이는 정상 문턱: 충격 수 > {TH_N:.1f}, Peak > {TH_P:.3f} g", f"L3 고장 {l3f.sum()} / 닮은 정상 {like.sum()}", ""]
for k, lab, _ in FEATS:
    v = sub[k].fillna(sub[k].median())
    L.append(f"{k:7s} AUC {roc_auc_score(yy, v):.2f}  고장 중앙 {sub.loc[yy == 1, k].median():.3g}  정상 중앙 {sub.loc[yy == 0, k].median():.3g}  ({lab})")
L += ["", sub.drop(columns=["y"]).sort_values(["group", "R"]).round(3).to_string()]

# 전체 150개에서 FC·Peak 분포 (설명용: L3 비교에서 본 차이가 다른 파일에도 있나)
L += ["", "전체 파일 — 사건 주파수 중심 FC (사건 있는 파일만)"]
for g, *_ in ORDER:
    v = E.loc[E.group == g, "FC"].dropna()
    L.append(f"  {g}: n={len(v)} 중앙 {v.median():.0f} Hz, 5–95% {v.quantile(.05):.0f}–{v.quantile(.95):.0f}, FC < 4.9 kHz 비율 {(v < 4900).mean():.2f}")
    a = E.loc[E.group == g, "ARM"].dropna()
    L.append(f"      ARM 중앙 {a.median():.3f}, 5–95% {a.quantile(.05):.3f}–{a.quantile(.95):.3f}")
# 후보 설명 (라벨을 본 뒤 정한 문턱 → 사후, 검증 안 됨): 고장 쪽 = FC < 4.9 kHz 또는 (R ≥ 0.7 이고 Peak ≥ 0.7 g)
cand = (E.FC < 4900) | ((E.R >= 0.7) & (E.peak_g >= 0.7))
E["cand"] = cand
L += ["", "사후 후보 '고장 쪽 = FC < 4.9 kHz 또는 (R ≥ 0.7 & Peak ≥ 0.7 g)' 켜지는 수"]
for g, *_ in ORDER:
    m = E.group == g
    L.append(f"  {g}: {int((cand & m).sum())}/{int(m.sum())}  " + " ".join(n.split('_')[1] for n in E.index[cand & m])[:400])
open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림 2: 사건 주파수 중심 × 2위/1위, 점 크기 = Peak, 전체 150개
fig, ax = plt.subplots(figsize=(10.5, 6.6))
for g, col, mk in ORDER[::-1]:
    m = (E.group == g).values & E.FC.notna().values
    ax.scatter(E.FC[m], E.R[m], s=12 + 60 * np.clip(np.log10(E.peak_g[m] / 0.05), 0, 2.5), c=col, marker=mk,
               lw=.6, edgecolors="white", zorder=3 if g.startswith("L3") else 2, label=f"{g} ({m.sum()})")
    if g.startswith("L3"):
        for i in np.flatnonzero(m):
            ax.annotate(f"{names[i].split('_')[1]} {E.peak_g.iloc[i]:.2g}g", (E.FC.iloc[i], E.R.iloc[i]), fontsize=7, xytext=(4, 2),
                        textcoords="offset points", color="#3d3d3a")
ax.axvline(4900, color="#6b6a64", ls="--", lw=1); ax.axhline(0.7, color="#6b6a64", ls=":", lw=1)
ax.text(4900, 0.45, " 4.9 kHz (사후)", fontsize=7.5, color="#6b6a64", transform=ax.get_xaxis_transform())
ax.text(0.99, 0.7, "R 0.7 (2-12 유격 규칙)", fontsize=7.5, color="#6b6a64", ha="right", va="bottom", transform=ax.get_yaxis_transform())
ax.set_xlabel("사건 주파수 중심 FC (Hz, 상위 5개 사건 ±2 ms)"); ax.set_ylabel("2위/1위 사건 높이 R")
ax.set_title("사건이 낮은 주파수(암 고유진동 2–3.5 kHz 근처)로 울리거나, 비슷한 크기로 반복되면 고장 쪽 — 점 크기 = 고역통과 Peak", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.grid(alpha=.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_fc_vs_R.png"), dpi=130); plt.close(fig)

# %% 그림 3: 가장 큰 사건 파형(±5 ms)과 스펙트럼 — L3 강타 고장 vs 강타 정상(EXP-025 오경보 포함)
SHOW = [("abn_009", "#1baf7a"), ("abn_012", "#1baf7a"), ("abn_030", "#1baf7a"), ("abn_024", "#1baf7a"),
        ("nor_025", "#eb6834"), ("nor_116", "#eb6834"), ("nor_055", "#eb6834"), ("nor_095", "#eb6834")]
fig, axs = plt.subplots(2, len(SHOW), figsize=(18, 5.6))
t5 = int(0.005 * FS)
for j, (nm, col) in enumerate(SHOW):
    i = names.index(nm); h = H[i]; p = int(np.argmax(np.abs(h)))
    seg = h[max(p - t5, 0): p + t5]; tt = (np.arange(len(seg)) - min(p, t5)) / FS * 1000
    axs[0, j].plot(tt, seg, color=col, lw=.8); axs[0, j].set_title(f"{nm}  Peak {np.abs(h).max():.2g} g\nFC {E.FC[nm]:.0f} Hz · R {E.R[nm]:.2f}", fontsize=8.5, loc="left")
    axs[0, j].set_xlabel("ms", fontsize=8)
    w = h[max(p - 2 * HW, 0): p + 2 * HW]; w = w * np.hanning(len(w))
    P = np.abs(np.fft.rfft(w, 512)) ** 2; f = np.fft.rfftfreq(512, 1 / FS)
    axs[1, j].plot(f / 1000, 10 * np.log10(P / P.max() + 1e-12), color=col, lw=.9)
    axs[1, j].axvspan(2, 3.5, color="#c9c8c2", alpha=.35, lw=0); axs[1, j].set_ylim(-40, 2); axs[1, j].set_xlabel("kHz", fontsize=8)
    for a in axs[:, j]:
        a.tick_params(labelsize=7); a.grid(alpha=.25)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
axs[1, 0].set_ylabel("dB (최대 기준)"); axs[0, 0].set_ylabel("g (고역통과)")
fig.suptitle("가장 큰 사건 ±5 ms (위) · ±4 ms 스펙트럼 (아래, 회색 = 1-3의 암 고유진동 2–3.5 kHz) — 초록 = L3 고장, 주황 = 강타 정상", fontsize=11, x=.01, ha="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_top_event.png"), dpi=120); plt.close(fig)
open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8").write("\n".join(L))
print("done")
