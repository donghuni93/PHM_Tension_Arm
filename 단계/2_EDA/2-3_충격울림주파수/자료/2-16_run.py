# %% [markdown]
# 2-16 충격 울림 주파수 — 무엇이 어디를 쳤나
# 질문 1. 충격 하나하나는 어느 주파수로 울리나. 정상과 고장이 다른가 (신효섭 11-13 보고서 그림 3을 우리 코드로 확인)
# 질문 2. 고장의 "8 kHz대 울림"은 우리 8.3 kHz 모드(2-11, 8.3 kHz 규칙)와 같은 현상인가
#         = 8.3 kHz 규칙이 켜지는 고장은 충격이 실제로 8.2–8.45 kHz로 울리는가 (H2′: 타격이 8.3 kHz 모드를 들뜨게 한다)
# 질문 3. 파일 단위로 보면 울림 축 두 개(2–5 kHz, 8.3 kHz)가 고장 몇 개를 덮고, 우리 문제 파일은 어디 있나
# - 충격 검출·울림 주파수 정의 = 신효섭(1 kHz 고역통과 → 포락선 6×중앙값, 5 ms 간격 → 충격 뒤 8 ms를 감쇠 사인파 사전 180개와 대조)
# - 대역 경계 2–5 kHz(팀원)와 8.20–8.45 kHz(2-9 우리)는 모두 라벨을 본 뒤 정한 값 = 사후. 라벨은 색·집계에만 쓴다
# 실행: conda run -n qe-phm python "단계/2_EDA/2-3_충격울림주파수/자료/2-16_run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from src.exp import load_train
from src.experts import peak83_table
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000

X, y, names = load_train(); y = np.asarray(y); N = len(y); nor = y == 0
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
idx = {s: i for i, s in enumerate(short)}
P83, W83, _ = peak83_table()
RULE83 = (P83 >= 0.75) & (W83 >= 150)                          # EXP-021 규칙 (학습 150개 전체 기준, 설명용)
FOCUS = {"nor_025": "오경보", "nor_055": "오경보", "nor_059": "오경보", "nor_116": "오경보",
         "abn_022": "미탐", "abn_005": "경계", "abn_014": "유격 규칙", "abn_017": "유격 규칙"}
FOCUS = {k: v for k, v in FOCUS.items() if k in idx}
COL = {"오경보": "#d62728", "미탐": "#9467bd", "경계": "#8c564b", "유격 규칙": "#17becf", "8.3k 규칙": "#2ca02c"}
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# %% 1. 충격 표 (파일 · 시각 · 포락선 높이 · 울림 주파수 · 감쇠비)
X1k = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
SEG = int(0.008 * FS); t = np.arange(SEG) / FS
fgrid = np.arange(1000, 12001, 250); zgrid = [0.01, 0.02, 0.04, 0.08]
atoms, par = [], []
for f in fgrid:
    for z in zgrid:
        w = 2 * np.pi * f; wd = w * np.sqrt(1 - z ** 2); a = np.exp(-z * w * t)
        Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1))
        atoms.append(Q); par.append((f, z))
atoms = np.stack(atoms); par = np.array(par)

rows = []
for i in range(N):
    x = X1k[i]
    env = sosfiltfilt(lp500, np.abs(hilbert(x)))
    base = np.median(env)
    pk, _ = find_peaks(env, height=6 * base, distance=int(0.005 * FS))
    pk = pk[pk + SEG <= len(x)]
    if not len(pk):
        continue
    S = np.stack([x[p:p + SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2)
    b = fit.argmax(1)
    for p, bb, ft in zip(pk, b, fit.max(1)):
        rows.append(dict(file=short[i], y=int(y[i]), t=p / FS, height=env[p] / base, f=par[bb, 0], zeta=par[bb, 1], fit=ft))
IM = pd.DataFrame(rows)
IM.to_csv(os.path.join(FIG, "impacts.csv"), index=False, encoding="utf-8-sig")
say(f"충격 {len(IM)}개 (정상 {int((IM.y == 0).sum())} · 고장 {int((IM.y == 1).sum())}), 충격 0개 파일: 정상 {int(nor.sum() - IM[IM.y == 0].file.nunique())} · 고장 {int((~nor).sum() - IM[IM.y == 1].file.nunique())}")

# 파일별 요약
B25 = (IM.f >= 2000) & (IM.f < 5000); B58 = (IM.f >= 5000) & (IM.f < 8000)
B83 = (IM.f >= 8200) & (IM.f < 8450); B8p = IM.f >= 8000
g = IM.assign(b25=B25, b58=B58, b83=B83, b8p=B8p).groupby("file")
F = pd.DataFrame({"n_imp": g.size(), "ring_2_5k": g.b25.mean(), "ring_5_8k": g.b58.mean(),
                  "ring_83": g.b83.mean(), "ring_8p": g.b8p.mean()}).reindex(short).fillna(0)
F["y"] = y; F["P83"] = P83; F["W83"] = W83; F["rule83"] = RULE83
F.to_csv(os.path.join(FIG, "files.csv"), encoding="utf-8-sig")

# %% 2. 그림 1 — 충격별 울림 주파수 분포 (250 Hz 칸 = 사전 격자)
edges = np.r_[fgrid - 125, fgrid[-1] + 125]
fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw={"height_ratios": [1.3, 1]})
for k, (lab, c, nm) in enumerate(((0, "#4472c4", "정상"), (1, "#ed7d31", "고장"))):
    v = IM.f[IM.y == lab]
    ax[0].hist(v, edges, weights=np.full(len(v), 1 / len(v)), alpha=0.6, color=c, label=f"{nm} 충격 {len(v)}개 ({IM[IM.y == lab].file.nunique()}파일)")
ax[0].axvspan(2000, 5000, color="#d62728", alpha=0.07); ax[0].axvspan(8200, 8450, color="#2ca02c", alpha=0.25)
ax[0].text(2100, ax[0].get_ylim()[1] * 0.92, "2–5 kHz (팀원)"); ax[0].text(8500, ax[0].get_ylim()[1] * 0.92, "8.20–8.45 kHz\n(우리 8.3 kHz 모드, 2-11)", color="#2ca02c")
ax[0].set_ylabel("충격 비율"); ax[0].legend(loc="upper left", bbox_to_anchor=(0.3, 1)); ax[0].set_xlabel("울림 주파수 (Hz)")
ax[0].set_title("① 충격 하나하나가 울리는 주파수 — 정상은 6.5–7.5 kHz 한 곳, 고장은 2–4 kHz와 8 kHz대로 갈라진다", loc="left")
# 파일 단위로도 같은가: 파일마다 같은 무게(충격이 많은 파일이 덮지 않게)
for lab, c, nm in ((0, "#4472c4", "정상"), (1, "#ed7d31", "고장")):
    sub = IM[IM.y == lab]; w = 1 / sub.groupby("file").f.transform("size") / sub.file.nunique()
    ax[1].hist(sub.f, edges, weights=w, alpha=0.6, color=c, label=nm)
ax[1].axvspan(8200, 8450, color="#2ca02c", alpha=0.25); ax[1].axvspan(2000, 5000, color="#d62728", alpha=0.07)
ax[1].set_title("② 같은 분포를 파일마다 같은 무게로 (충격이 많은 파일이 덮지 않게)", loc="left")
ax[1].set_xlabel("울림 주파수 (Hz)"); ax[1].set_ylabel("파일 평균 비율"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-16_1_울림분포.png"), dpi=130); plt.close(fig)

# %% 3. 그림 2 — 울림 지도: 파일(행) × 울림 주파수(열), 칸 = 그 파일 충격 중 비율
H = np.zeros((N, len(fgrid)))
for i, s in enumerate(short):
    v = IM.f[IM.file == s].values
    if len(v):
        H[i] = np.histogram(v, edges)[0] / len(v)
def order(mask):
    ii = np.flatnonzero(mask)
    c = (H[ii] * fgrid).sum(1) / np.maximum(H[ii].sum(1), 1e-9)          # 울림 무게중심 순
    return ii[np.argsort(c)]
fig, axs = plt.subplots(1, 2, figsize=(16, 11), gridspec_kw={"width_ratios": [4, 1.3]})
for ax, mask, nm in ((axs[0], nor, "정상 120"), (axs[1], ~nor, "고장 30")):
    ii = order(mask)
    ax.imshow(H[ii], aspect="auto", cmap="magma_r", vmin=0, vmax=0.8, extent=(edges[0], edges[-1], len(ii), 0))
    ax.axvline(8200, c="#2ca02c", lw=0.8); ax.axvline(8450, c="#2ca02c", lw=0.8)
    ax.axvline(2000, c="#d62728", lw=0.6, ls=":"); ax.axvline(5000, c="#d62728", lw=0.6, ls=":")
    ax.set_title(f"{nm} (울림 무게중심 순)", loc="left"); ax.set_xlabel("울림 주파수 (Hz)")
    if mask is not nor:
        ax.set_yticks(np.arange(len(ii)) + 0.5)
        lab = []
        for i in ii:
            s = short[i]; tag = (" ◆8.3k" if RULE83[i] else "") + (f" [{FOCUS[s]}]" if s in FOCUS else "")
            lab.append(s[-3:] + tag)
        ax.set_yticklabels(lab, fontsize=8)
    else:
        ax.set_yticks([k + 0.5 for k, i in enumerate(ii) if short[i] in FOCUS])
        ax.set_yticklabels([f"{short[i][-3:]} [{FOCUS[short[i]]}]" for i in ii if short[i] in FOCUS], fontsize=8, color="#d62728")
fig.suptitle("② 울림 지도 — 칸 색 = 그 파일 충격 중 해당 주파수로 울린 비율 (빈 행 = 충격 없음). 초록 선 = 8.20–8.45 kHz, 빨간 점선 = 2–5 kHz. ◆ = 8.3 kHz 규칙",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(FIG, "2-16_2_울림지도.png"), dpi=120); plt.close(fig)

# %% 4. 그림 3 — 8.3 kHz 모드와 같은 현상인가: 스펙트럼 봉우리(P83) × 충격이 8.2–8.45 kHz로 울린 비율
fig, ax = plt.subplots(figsize=(10, 7))
for lab, c, nm in ((0, "#4472c4", "정상"), (1, "#ed7d31", "고장")):
    m = y == lab
    ax.scatter(F.ring_83[m], P83[m], s=np.where(F.n_imp[m] > 0, 18 + 3 * np.sqrt(F.n_imp[m]), 10), c=c, alpha=0.55,
               marker="o", label=f"{nm} (점 크기 = 충격 수)")
ax.scatter(F.ring_83[RULE83], P83[RULE83], s=160, facecolors="none", edgecolors="#2ca02c", lw=2, label="8.3 kHz 규칙 켜짐")
for i in np.flatnonzero(RULE83 | ((~nor) & (F.ring_83.values > 0.1))):
    ax.annotate(short[i], (F.ring_83.iloc[i], P83[i]), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.axhline(0.75, ls="--", c="#2ca02c", lw=0.8); ax.text(ax.get_xlim()[1] * 0.98, 0.77, "규칙 돌출 문턱 0.75", ha="right", color="#2ca02c", fontsize=8)
ax.set_xlabel("충격 중 8.20–8.45 kHz로 울린 비율 (울림, 시간 영역)"); ax.set_ylabel("8.20–8.45 kHz 스펙트럼 돌출 P83 (dec, 2-11)")
ax.set_title("③ 8.3 kHz 봉우리(스펙트럼)와 8.3 kHz 울림(충격)은 같은 현상인가", loc="left"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-16_3_8k모드연결.png"), dpi=130); plt.close(fig)

# %% 5. 그림 4 — 파일 지도: 2–5 kHz 울림 × 8 kHz대 울림
q25, q83 = np.percentile(F.ring_2_5k[nor], 95), np.percentile(F.ring_8p[nor], 95)
fig, ax = plt.subplots(figsize=(10, 7.5))
rng = np.random.default_rng(0); jit = lambda n: rng.uniform(-0.012, 0.012, n)
for lab, c, nm in ((0, "#4472c4", "정상"), (1, "#ed7d31", "고장")):
    m = y == lab
    ax.scatter(F.ring_2_5k[m] + jit(m.sum()), F.ring_8p[m] + jit(m.sum()), s=22, c=c, alpha=0.6, label=nm)
ax.scatter(F.ring_2_5k[RULE83], F.ring_8p[RULE83], s=150, facecolors="none", edgecolors="#2ca02c", lw=2, label="8.3 kHz 규칙 켜짐")
for s, kind in FOCUS.items():
    i = idx[s]; ax.scatter(F.ring_2_5k.iloc[i], F.ring_8p.iloc[i], s=110, facecolors="none", edgecolors=COL[kind], lw=1.8)
    ax.annotate(f"{s} ({kind})", (F.ring_2_5k.iloc[i], F.ring_8p.iloc[i]), xytext=(6, -10), textcoords="offset points", fontsize=8, color=COL[kind])
ax.axvline(q25, ls="--", c="gray", lw=0.8); ax.axhline(q83, ls="--", c="gray", lw=0.8)
ax.text(q25 + 0.01, 0.98, f"정상 95% {q25:.2f}", fontsize=8, color="gray"); ax.text(0.7, q83 + 0.01, f"정상 95% {q83:.2f}", fontsize=8, color="gray")
ax.set_xlabel("ring_2_5k: 충격 중 2–5 kHz로 울린 비율"); ax.set_ylabel("ring_8p: 충격 중 8 kHz 이상으로 울린 비율")
ax.set_title("④ 파일 지도 — 고장은 두 방향(2–5 kHz / 8 kHz대)으로, 정상은 원점 근처(5–8 kHz 울림)", loc="left"); ax.legend(loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2-16_4_파일지도.png"), dpi=130); plt.close(fig)

# %% 6. 집계
ab = ~nor
a25 = F.ring_2_5k.values > q25; a8 = F.ring_8p.values > q83
say(f"정상 95% 선: ring_2_5k {q25:.3f} · ring_8p {q83:.3f} (정상은 5%씩 넘는 것이 정의상 정상)")
say(f"고장 30개 중 ring_2_5k > 선: {int((a25 & ab).sum())} · ring_8p > 선: {int((a8 & ab).sum())} · 둘 중 하나: {int(((a25 | a8) & ab).sum())}")
say(f"정상 120개 중 ring_2_5k > 선: {int((a25 & nor).sum())} · ring_8p > 선: {int((a8 & nor).sum())} · 둘 중 하나: {int(((a25 | a8) & nor).sum())}")
say(f"  ring_2_5k 넘는 정상: {[short[i] for i in np.flatnonzero(a25 & nor)]}")
say(f"  ring_8p 넘는 정상: {[short[i] for i in np.flatnonzero(a8 & nor)]}")
say(f"  둘 다 못 넘는 고장: {[short[i] for i in np.flatnonzero(~(a25 | a8) & ab)]}")
say(f"\n8.3 kHz 규칙 켜짐 {int(RULE83.sum())}개: " + ", ".join(f"{short[i]}(y={y[i]}, 8.3k 울림 {F.ring_83.iloc[i]:.2f}, 8k+ {F.ring_8p.iloc[i]:.2f}, 충격 {int(F.n_imp.iloc[i])})" for i in np.flatnonzero(RULE83)))
say(f"충격이 8.20–8.45 kHz로 울린 비율 > 0.1인 파일: 정상 {int(((F.ring_83 > 0.1) & nor).sum())} · 고장 {int(((F.ring_83 > 0.1) & ab).sum())} → "
    + ", ".join(short[i] for i in np.flatnonzero((F.ring_83.values > 0.1) & ab)))
say(f"8.20–8.45 kHz로 울린 충격: 정상 {int((B83 & (IM.y == 0)).sum())}/{int((IM.y == 0).sum())} · 고장 {int((B83 & (IM.y == 1)).sum())}/{int((IM.y == 1).sum())}")
say(f"충격 울림 주파수 중앙값: 정상 {IM.f[IM.y == 0].median():.0f} Hz · 고장 {IM.f[IM.y == 1].median():.0f} Hz")
say("\n문제 파일")
say(F.loc[list(FOCUS), ["n_imp", "ring_2_5k", "ring_5_8k", "ring_83", "ring_8p", "P83"]].round(2).to_string())
with open(os.path.join(FIG, "2-16_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
