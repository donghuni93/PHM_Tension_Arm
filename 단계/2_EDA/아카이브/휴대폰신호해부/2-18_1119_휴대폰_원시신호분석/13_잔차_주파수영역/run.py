# %% [markdown]
# 13. 주파수 영역 잔차: ΔPSD · AE bin별 복원 오차 (4개 코호트 분리 + 에너지 정규화 병행)
# 질문: 오탐 경계 정상(116·059·055·106)이 튀는 이유는 본체 모드 가진 세기(전체 에너지)인가, 아니면 실제로 다른 대역의 이탈인가?
#       미탐 경계 고장(022·014·017·002)은 2–4 kHz, 8.25 kHz 대역에서 분리되는가?
# 코호트: ① 일반 정상(회색 5–95% 띠) ② 오탐 경계 정상(파랑) ③ 미탐 경계 고장(빨강) ④ 일반 고장 중앙값(주황)
# 행: 1 ΔPSD 원본 / 2 ΔPSD 전체 파워 정규화 / 3 ΔPSD 본체 대역(5.5–7.75 kHz) 정규화 / 4 AE bin 오차 / 5 AE bin 오차(모양 AE = 게인 제거, H3)
# ΔPSD = 파일의 평균 log PSD(dB) − 일반 정상 중앙값. AE는 5-fold OOF(파일이 자기 학습에 안 들어감).
# 보기용이다. 대역(2–4 / 5.5–7.75 / 8.0–8.7 kHz)은 2-16 울림 기록에서 사전 고정한 값이다.
# 실행(PC): conda run -n qe-phm python "단계/2_EDA/아카이브/휴대폰신호해부/2-18_1119_휴대폰_원시신호분석/13_잔차_주파수영역/run.py"

# %% 설정
import os, sys
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from src.experts import X, y, names, LP, F_PSD, ae_segerr     # LP: (150, 8구간, 1025) log10 PSD, 200 Hz 고역통과
from matplotlib import font_manager as _fm  # src.viz가 글꼴을 바꾸므로 임포트 뒤에 설정
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):      # 한글 글꼴 (있는 첫 번째)
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

FP_NORMAL = [116, 59, 55, 106]
FN_FAULT = [22, 14, 17, 2]
def find(num, label):
    for i, (n, l) in enumerate(zip(names, y)):
        if l == label and int("".join(c for c in n if c.isdigit()) or -1) == num:
            return i
    raise KeyError(f"{num} (label {label}) 없음. names 예: {names[:3]}")
IDX_FP = [find(n, 0) for n in FP_NORMAL]
IDX_FN = [find(n, 1) for n in FN_FAULT]
GEN_N = np.array([i for i in np.where(y == 0)[0] if i not in IDX_FP])      # 일반 정상
OTH_F = np.array([i for i in np.where(y == 1)[0] if i not in IDX_FN])      # 일반 고장
f = F_PSD
BANDS = {"2–4k": (2000, 4000), "5.5–7.75k": (5500, 7750), "8.0–8.7k": (8000, 8700),
         "9–10k(사후 추가)": (9000, 10000)}   # 9–10 kHz는 첫 실행의 그림을 본 뒤 추가한 탐색용 대역 (2-16에서 9.5–10 kHz 울림은 고장 근거 아님으로 기록됨)

# %% 행렬 5개 만들기
Lm = LP.mean(1)                                    # (150, 1025) log10 PSD (구간 평균)
Pw = 10 ** Lm
body = (f >= 5500) & (f <= 7750)
tot_db = 10 * np.log10(Pw.sum(1))[:, None]         # 전체 파워 (dB)
body_db = 10 * np.log10(Pw[:, body].sum(1))[:, None]
raw_db = 10 * Lm
def delta(M):                                       # 일반 정상 중앙값을 뺌
    return M - np.median(M[GEN_N], axis=0)
M_raw = delta(raw_db)
M_tot = delta(raw_db - tot_db)                      # 전체 에너지로 나눔 → 모양만
M_body = delta(raw_db - body_db)                    # 본체 모드 대역 에너지로 나눔

def oof_bins(shape):
    out = np.zeros((len(y), len(f)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        out[te] = np.log10(ae_segerr(tr, te, shape=shape).mean(1) + 1e-12)
    return out
M_ae = oof_bins(False)
M_aes = oof_bins(True)

ROWS = [("ΔPSD 원본 (dB)", M_raw), ("ΔPSD 전체 파워 정규화 (dB)", M_tot), ("ΔPSD 본체 대역 정규화 (dB)", M_body),
        ("AE bin 복원 오차 log10", M_ae), ("AE bin 복원 오차 (모양 AE) log10", M_aes)]

# %% 코호트 오버레이 그림
fig, ax = plt.subplots(len(ROWS), 1, figsize=(14, 3.3 * len(ROWS)), sharex=True)
for a, (title, M) in zip(ax, ROWS):
    lo, hi = np.percentile(M[GEN_N], [5, 95], axis=0)
    a.fill_between(f / 1000, lo, hi, color="gray", alpha=0.35, label="일반 정상 5–95%")
    a.plot(f / 1000, np.median(M[GEN_N], 0), color="black", lw=0.8)
    for i in IDX_FP: a.plot(f / 1000, M[i], color="tab:blue", lw=0.9, label=f"오탐 경계 정상 {names[i]}")
    for i in IDX_FN: a.plot(f / 1000, M[i], color="tab:red", lw=0.9, label=f"미탐 고장 {names[i]}")
    a.plot(f / 1000, np.median(M[OTH_F], 0), color="darkorange", lw=2, label="일반 고장 중앙값")
    for lo_, hi_ in BANDS.values(): a.axvspan(lo_ / 1000, hi_ / 1000, color="gold", alpha=0.12)
    a.set_title(title, fontsize=10, loc="left"); a.set_xlim(0, 12.5)
ax[0].legend(fontsize=7, ncol=3, loc="upper right"); ax[-1].set_xlabel("주파수 (kHz). 노란 띠 = 사전 고정 대역 3개")
fig.savefig(os.path.join(FIG, "cohort_overlay.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 대역별 z 점수 표: 파일이 일반 정상 분포에서 몇 σ 벗어나는가 (행렬 종류 × 대역)
rows = []
for tag, M in (("ΔPSD원본", M_raw), ("ΔPSD전체정규화", M_tot), ("ΔPSD본체정규화", M_body), ("AE", M_ae), ("AE모양", M_aes)):
    for bn, (lo_, hi_) in BANDS.items():
        m = (f >= lo_) & (f <= hi_)
        v = M[:, m].mean(1)
        z = (v - v[GEN_N].mean()) / (v[GEN_N].std() + 1e-12)
        rows.append({"측정": tag, "대역": bn, **{f"오탐정상 {names[i]}": z[i] for i in IDX_FP},
                     **{f"미탐고장 {names[i]}": z[i] for i in IDX_FN}, "일반고장 중앙값": np.median(z[OTH_F])})
T = pd.DataFrame(rows).round(2)
T.to_csv(os.path.join(HERE, "band_zscore.csv"), index=False, encoding="utf-8-sig")
print(T.to_string(index=False))

# %% z 점수 히트맵 (한눈에: 어느 측정·대역에서 파일이 갈리는가)
cols = [c for c in T.columns if c not in ("측정", "대역")]
Z = T[cols].values
fig, a = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.45 * len(T) + 2))
im = a.imshow(Z, cmap="coolwarm", vmin=-6, vmax=6, aspect="auto")
a.set_xticks(range(len(cols))); a.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
a.set_yticks(range(len(T))); a.set_yticklabels([f"{r['측정']} {r['대역']}" for _, r in T.iterrows()], fontsize=8)
for r in range(Z.shape[0]):
    for c in range(Z.shape[1]): a.text(c, r, f"{Z[r, c]:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, label="일반 정상 대비 z (σ)")
a.set_title("대역별 z: 오탐 정상이 '정규화하면 0 근처로 돌아가는가' / 고장이 '정규화해도 남는가'")
fig.savefig(os.path.join(FIG, "band_zscore_heatmap.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("저장:", FIG)

# %% 합본 (휴대폰용): cohort_overlay + band_zscore_heatmap 세로로
import matplotlib.image as mpimg
imgs = [mpimg.imread(os.path.join(FIG, p))[..., :3] for p in ("cohort_overlay.png", "band_zscore_heatmap.png")]
Wd = max(im.shape[1] for im in imgs)
big = np.concatenate([np.pad(im, ((0, 0), ((Wd - im.shape[1]) // 2, Wd - im.shape[1] - (Wd - im.shape[1]) // 2), (0, 0)), constant_values=1.0) for im in imgs])
plt.imsave(os.path.join(FIG, "freq_residual_all.png"), big)
print("합본:", os.path.join(FIG, "freq_residual_all.png"))
