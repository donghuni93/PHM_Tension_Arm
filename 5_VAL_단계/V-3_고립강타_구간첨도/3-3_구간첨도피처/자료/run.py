# %% [markdown]
# V-7 구간 첨도 EDA (11-27)
# - 질문: 효섭 A_segk(0.1초 구간 첨도 평균, train AUC 0.911, V-5)를 우리 전처리로 만들면
#   (1) train에서 고장을 가르는가 (2) 우리 ET 15열(특히 H3 첨도)과 다른 정보인가 (3) train 고립 강타 정상을 올리지 않는가 (4) val 이동은 고장 비율 범위 안인가
# - 가설 H1″(물리가설 H1 연장): 유격 타격은 2초 안 여러 0.1초 구간에 흩어진다. 외부 강타(정상)는 한 구간에 몰린다
# - 열(실행 전 고정, 라벨 보고 고르지 않음). 신호 = 평균 제거 → 1–11 kHz 대역통과(우리 검출기 입력과 같음), 20구간 × 2500점, 첨도(가우스 = 3)
#   segk_mean  구간 첨도 평균(효섭 방식)
#   segk_med   구간 첨도 중앙값
#   segk_trim  가장 큰 구간 1개를 뺀 평균 — 강타 하나에 끌려가지 않는가(가설의 직접 검정)
#   n_seg_hi   첨도가 train 정상 전체 구간의 99 % 분위를 넘는 구간 수(0–20)
#   (참고) A_segk  효섭과 같은 10 Hz 고역통과 신호의 구간 첨도 평균(재현 확인)
# - 관문(4-19와 같음): AUC [95 %] · 정상 99 % 넘는 고장 · 고장 10 % 선 넘는 정상 · val 이동 AUC(고장 0 · 20 · 40 % 섞은 기준) · ρ(하중) · ET 15열과의 최대 |ρ|
# - val 그룹(다른 조 결과로 추론)은 그림 표시에만 쓰고 열 정의 · 문턱에 쓰지 않는다
#   conda run -n qe-phm python 5_VAL_단계/V-3_고립강타_구간첨도/3-3_구간첨도피처/자료/run.py

# %% 설정 · 데이터
import os, sys, json
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib
from scipy import stats
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr
from src.data import load_dataset, load_folder
from src.exp import outlier_idx

FS, L = 25000, 2500
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
Xtr, y, ntr = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
Xva, nva = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
X = np.vstack([Xtr, Xva]); names = np.array(list(ntr) + list(nva))
grp = np.array(["normal" if v == 0 else "abnormal" for v in y] + ["val"] * len(nva))
N, A, V = grp == "normal", grp == "abnormal", grp == "val"

# %% 구간 첨도
X0 = X - X.mean(1, keepdims=True)
S1 = sosfiltfilt(butter(4, [1000, 11000], "bandpass", fs=FS, output="sos"), X0, axis=1)
S0 = sosfiltfilt(butter(4, 10, "highpass", fs=FS, output="sos"), X0, axis=1)
seg = lambda S: stats.kurtosis(S[:, :S.shape[1] // L * L].reshape(len(S), -1, L), axis=2, fisher=False)   # (파일, 20)
K1, K0 = seg(S1), seg(S0)
thr = np.percentile(K1[N].ravel(), 99)                       # train 정상 구간 전체의 99 % 분위
Ks = np.sort(K1, 1)
F = pd.DataFrame(index=names)
F["segk_mean"] = np.log10(K1.mean(1))
F["segk_med"] = np.log10(np.median(K1, 1))
F["segk_trim"] = np.log10(Ks[:, :-1].mean(1))
F["n_seg_hi"] = (K1 > thr).sum(1)
F["(참고) A_segk"] = np.log10(K0.mean(1))

# 백색화 열 (V-6 결과를 받아 추가, 정의는 효섭 그대로: AR(40), 앞 20000점으로 계수, 양끝 500점 자름)
from scipy import linalg
from scipy.signal import lfilter, fftconvolve
def ar_whiten(x, p=40):
    r = fftconvolve(x[:20000], x[:20000][::-1], "full")[19999:19999 + p + 1]   # np.correlate와 같은 값, FFT로 빠르게
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1]); return lfilter(np.r_[1, -a], [1], x)[p:]
W1 = np.stack([ar_whiten(s)[500:-500] for s in S1])          # 우리 1–11 kHz 신호를 백색화
W0 = np.stack([ar_whiten(s)[500:-500] for s in S0])          # 효섭 D_ 와 같은 10 Hz 고역통과 신호
KW1, KW0 = seg(W1), seg(W0)                                  # 48,960점 → 19구간
thr_w = np.percentile(KW1[N].ravel(), 99)
F["w_segk_mean"] = np.log10(KW1.mean(1))
F["w_n_seg_hi"] = (KW1 > thr_w).sum(1)
F["(참고) D_segk"] = np.log10(KW0.mean(1))
cols = list(F.columns)

# %% ET 15열(제출 모델 EXP-062 구성 3, V-1 캐시)과 하중
SUB = os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062")
m = joblib.load(os.path.join(SUB, "모델", "final_model_062.joblib")); core = m.core_
Ttr, Tva, _, _, _ = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
ecols = list(core.cols) + list(core.peak)
Etr, _ = core._et_input(Ttr); Eva, _ = core._et_input(Tva)
E = pd.DataFrame(np.vstack([Etr, Eva]), columns=ecols, index=names)
oof = np.r_[core.Z_[:, 0], np.full(len(nva), np.nan)]
p = np.clip(core.et_.predict_proba(Eva)[:, 1], 1 / 600, 1 - 1 / 600); lv = np.r_[np.full(len(ntr), np.nan), np.log(p / (1 - p))]
load = sosfiltfilt(butter(4, 20, "lowpass", fs=FS, output="sos"), X0, axis=1).std(1)

# %% 관문
rng = np.random.default_rng(0)
auc = lambda a, b: mannwhitneyu(a, b).statistic / (len(a) * len(b))
def ci(a, b, B=2000):
    return np.percentile([auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(B)], [2.5, 97.5])
def mix_range(v, frac, B=1000):
    n, a = v[N], v[A]; k = int(round(40 * frac)); out = []
    for _ in range(B):
        s = np.r_[rng.choice(n, 40 - k), rng.choice(a, k)] if k else rng.choice(n, 40); out.append(auc(s, n))
    return np.percentile(out, [5, 95])
T = N | A
out = [f"구간 첨도 문턱(n_seg_hi) = train 정상 구간 2400개의 99 % 분위 = {thr:.2f} · 백색화(w_n_seg_hi) = {thr_w:.2f}", ""]
G = []
for c in cols:
    v = F[c].values.astype(float); n, a = v[N], v[A]; lo, hi = ci(a, n); q99 = np.percentile(n, 99); q10a = np.percentile(a, 10)
    r = {j: spearmanr(v[T], E[j].values[T])[0] for j in ecols}; jb = max(r, key=lambda k: abs(r[k]))
    m0, m20, m40 = mix_range(v, 0), mix_range(v, .2), mix_range(v, .4)
    G.append(dict(열=c, AUC=f"{auc(a, n):.3f} [{lo:.2f}, {hi:.2f}]", 정상99넘는고장=f"{int((A & (v > q99)).sum())}: " + " ".join(n_[4:] for n_ in names[A & (v > q99)]),
                  고장10선넘는정상=f"{int((N & (v > q10a)).sum())}/120", val이동=f"{auc(v[V], n):.2f} (0 %: {m0[0]:.2f}–{m0[1]:.2f} · 20 %: {m20[0]:.2f}–{m20[1]:.2f} · 40 %: {m40[0]:.2f}–{m40[1]:.2f})",
                  ρ하중=f"{spearmanr(n, load[N])[0]:+.2f}", ρH3=f"{r['H3_kurt_1-2k']:+.2f}", 최대겹침=f"{jb} {r[jb]:+.2f}"))
Gt = pd.DataFrame(G)
out += ["== 관문 ==", Gt.to_string(index=False), ""]
out.append("후보끼리 ρ (train 150): " + " · ".join(f"{a}~{b} {spearmanr(F[a][T], F[b][T])[0]:+.2f}" for i, a in enumerate(cols) for b in cols[i + 1:]))

# H3를 같이 쓸 때 더해지는 정보: H3 순위로 설명되지 않는 부분(잔차)의 AUC, H3 ≤ 고장 중앙값 파일 안에서의 AUC
h3 = E["H3_kurt_1-2k"].values
out.append("\n== H3와 겹치지 않는 정보 ==")
for c in cols:
    v = F[c].values.astype(float)
    rk = lambda z: stats.rankdata(z) / len(z)
    b = np.polyfit(rk(h3[T]), rk(v[T]), 1); res = rk(v[T]) - np.polyval(b, rk(h3[T]))
    yy = A[T]; lowH3 = h3[T] <= np.median(h3[A])
    out.append(f"{c:14s} H3 순위 잔차 AUC {auc(res[yy], res[~yy]):.3f} | H3 ≤ 고장 중앙값 파일(정상 {int((lowH3 & ~yy).sum())} · 고장 {int((lowH3 & yy).sum())}) 안 AUC {auc(v[T][lowH3 & yy], v[T][lowH3 & ~yy]):.3f}")

# %% 볼 파일: train 고립 강타 정상(V-2 · V-5) · 오경보 이상치 · val 후보
pct = lambda c, f: (F[c].values[N] < F.loc[f, c]).mean() * 100
SHOW = {"train 고립 강타 정상": ["nor_046", "nor_060", "nor_006", "nor_038", "nor_055", "nor_089", "nor_054", "nor_032"],
        "이상치 정상(◆)": [ntr[i] for i in outlier_idx(ntr)],
        "train 고장 예": ["abn_021", "abn_028", "abn_002", "abn_014", "abn_017"],
        "val 고립 강타(V-2)": ["val_012", "val_032", "val_034", "val_036"],
        "val 8조만 · 경계": ["val_010", "val_018", "val_037", "val_009", "val_021"],
        "val 확실한 고장": ["val_039", "val_016", "val_004"]}
rows = []
for g, fs in SHOW.items():
    for f in fs:
        rows.append(dict(묶음=g, file=f, **{c: f"{(F.loc[f, c] if "n_seg" in c else 10 ** F.loc[f, c]):.1f} ({pct(c, f):.0f})" for c in cols},
                         최대구간첨도=f"{K1[list(names).index(f)].max():.1f}"))
out += ["\n== 파일별: 값 (train 정상 안 백분위). segk 열은 원래 첨도 값 ==", pd.DataFrame(rows).to_string(index=False)]
for c in cols:
    q = np.percentile(F[c][N], 99)
    out.append(f"val 중 {c} > 정상 99 %: " + (", ".join(names[V & (F[c].values > q)]) or "없음"))
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))
F.assign(group=grp, H3=h3, et_logit=np.where(V, lv, oof)).to_csv(os.path.join(HERE, "파일표.csv"), encoding="utf-8-sig")
json.dump({"thr_seg99": float(thr), "gates": G}, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# %% 그림 1: 네 열 분포 (정상 · 고장 · val), 고립 강타 정상 · val 후보 표시
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src import viz  # noqa: F401
C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}
LAB = {"normal": "train 정상 120", "abnormal": "train 고장 30", "val": "val 40"}
ISO_N, ISO_V = SHOW["train 고립 강타 정상"], SHOW["val 고립 강타(V-2)"]
TTL = {"segk_mean": "구간 첨도 평균", "segk_med": "구간 첨도 중앙값", "segk_trim": "가장 큰 구간을 뺀 평균", "n_seg_hi": "뾰족한 구간 수 (정상 99 % 넘는 구간)",
       "w_segk_mean": "백색화 구간 첨도 평균 (1–11 kHz)", "w_n_seg_hi": "백색화 뾰족한 구간 수 (1–11 kHz)", "(참고) D_segk": "(참고) 효섭 D_segk (10 Hz 고역)"}
PAN = ["segk_mean", "segk_trim", "n_seg_hi", "w_segk_mean", "w_n_seg_hi", "(참고) D_segk"]
fig, ax = plt.subplots(2, 3, figsize=(16, 10)); ax = ax.ravel()
for a_, c in zip(ax, PAN):
    a_.grid(axis="y", alpha=.3)
    for k, g in enumerate(("normal", "abnormal", "val")):
        v = F[c][grp == g].values.astype(float); v = v if "n_seg" in c else 10 ** v
        jx = k + np.random.default_rng(k).uniform(-.25, .25, len(v))
        a_.scatter(jx, v, s=14, color=C[g], alpha=.55, lw=0); a_.hlines(np.median(v), k - .32, k + .32, color="k", lw=2)
        nm_g = names[grp == g]
        for j, f in enumerate(nm_g):
            if f in ISO_N or f in ISO_V:
                a_.scatter(jx[j], v[j], s=60, facecolor="none", edgecolor="#8e44ad" if f in ISO_V else "k", lw=1.4)
                a_.annotate(f[-3:], (jx[j], v[j]), xytext=(4, 2), textcoords="offset points", fontsize=7, color="#8e44ad" if f in ISO_V else "k")
    q = np.percentile(F[c][N], 99); a_.axhline(q if "n_seg" in c else 10 ** q, color=C["normal"], ls="--", lw=1)
    if "n_seg" not in c: a_.set_yscale("log")
    a_.set_xticks(range(3)); a_.set_xticklabels([LAB[g] for g in ("normal", "abnormal", "val")], fontsize=9)
    a_.set_title(f"{TTL[c]}\nAUC {Gt.set_index('열').loc[c, 'AUC']}", loc="left", fontsize=10)
fig.text(.01, .01, "검은 선 = 중앙값 · 파란 점선 = train 정상 99 % · 검은 원 = train 고립 강타 정상 · 보라 원 = val 고립 강타(V-2) · 신호 1–11 kHz, 0.1초 × 20구간", fontsize=9, color="#555")
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(os.path.join(FIG, "구간첨도_분포.png"), dpi=130)

# %% 그림 2: H3(ET 1–2 kHz 첨도) 대 segk_trim — 새 정보가 있는가
fig, a_ = plt.subplots(figsize=(8, 6.5))
for g in ("normal", "abnormal", "val"):
    k = grp == g; a_.scatter(h3[k], 10 ** F["segk_trim"][k], s=22 if g != "val" else 34, color=C[g], alpha=.6, lw=0,
                             marker="o" if g != "val" else "^", label=LAB[g])
for f in ISO_N + ISO_V:
    i = list(names).index(f); a_.annotate(f[-3:] if f.startswith("val") else f.replace("nor_", "N"), (h3[i], 10 ** F["segk_trim"].iloc[i]),
                                          xytext=(4, 3), textcoords="offset points", fontsize=8, color="#8e44ad" if f in ISO_V else "k")
a_.set_yscale("log"); a_.set_xlabel("ET 입력 H3_kurt_1-2k (1–2 kHz 첨도, 제출 모델 값)"); a_.set_ylabel("segk_trim (가장 큰 구간을 뺀 구간 첨도 평균)")
a_.axhline(10 ** np.percentile(F["segk_trim"][N], 99), color=C["normal"], ls="--", lw=1)
a_.legend(fontsize=9); a_.grid(alpha=.3)
a_.set_title(f"H3와 segk_trim · train ρ {spearmanr(h3[T], F['segk_trim'][T])[0]:+.2f}", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "H3_대_구간첨도.png"), dpi=130)

# %% 그림 3: 구간별 첨도 지도 — 사건이 흩어졌나, 몰렸나
pick = ISO_N[:6] + ["nor_116", "abn_021", "abn_014", "abn_002", "val_039", "val_016"] + ISO_V + ["val_010", "val_018", "val_037"]
M = np.log10(K1[[list(names).index(f) for f in pick]])
fig, a_ = plt.subplots(figsize=(11, 8))
im = a_.imshow(M, aspect="auto", cmap="magma_r", vmin=np.log10(3), vmax=np.log10(60))
a_.set_yticks(range(len(pick))); a_.set_yticklabels([f"{f}  (뾰족 {int(F.loc[f, 'n_seg_hi'])})" for f in pick], fontsize=8)
for t, f in zip(a_.get_yticklabels(), pick): t.set_color(C["val"] if f.startswith("val") else C["abnormal"] if f.startswith("abn") else C["normal"])
a_.set_xticks(range(20)); a_.set_xticklabels([f"{i/10:.1f}" for i in range(20)], fontsize=7); a_.set_xlabel("구간 시작 (초), 0.1초 × 20")
for r_, f in enumerate(pick):
    for c_ in np.where(K1[list(names).index(f)] > thr)[0]: a_.text(c_, r_, "•", ha="center", va="center", color="cyan", fontsize=9)
cb = fig.colorbar(im, ax=a_); cb.set_label("구간 첨도 (log10, 3 = 가우스)")
a_.set_title(f"구간별 첨도 · • = train 정상 구간 99 % 분위({thr:.1f}) 초과 · 이름 색 = 정상 / 고장 / val", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "구간첨도_지도.png"), dpi=130)
print("done")
