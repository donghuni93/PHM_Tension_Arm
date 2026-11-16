# %% [markdown]
# 2-14 보충 (11-16): 정상은 왜 큰 타격이 생기나 — 정상·고장의 "바탕보다 큰 타격" 개수·크기·시점 비교
# 사건 정의는 run.py와 같다: 200 Hz 고역 |h| > 8σ (σ = 1.4826·MAD), 10 ms 안은 한 사건, 높이 = 사건 최대 |h| / σ
# "큰 타격" 문턱 15·20·30·50σ는 라벨과 무관하게 미리 정한 격자
# 실행: conda run -n qe-phm python "단계/2_EDA/유의미/2-14_1114_불안정고장_005_020/run2_정상타격.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); names = [str(n) for n in names]
FS = 25000; X = X - X.mean(1, keepdims=True)
HP = sosfiltfilt(butter(4, 200, "high", fs=FS, output="sos"), X, axis=1)
LP = sosfiltfilt(butter(4, 20, "low", fs=FS, output="sos"), X, axis=1)
SIG = 1.4826 * np.median(np.abs(HP - np.median(HP, 1, keepdims=True)), 1)


def events(i, k=8, merge=0.010):
    h = HP[i]; s = SIG[i]; idx = np.flatnonzero(np.abs(h) > k * s)
    if not len(idx): return np.array([]), np.array([])
    g = np.split(idx, np.flatnonzero(np.diff(idx) > int(merge * FS)) + 1)
    p = np.array([gg[np.argmax(np.abs(h[gg]))] for gg in g])
    return p / FS, np.abs(h[p]) / s


# %% 파일별 표
rows = []
for i, n in enumerate(names):
    t, hh = events(i); o = np.argsort(hh)[::-1]; t, hh = t[o], hh[o]
    big = t[hh >= 20]
    lp = np.abs(LP[i] - np.median(LP[i]))
    rows.append(dict(f=n, y=y[i], n8=len(hh), n20=(hh >= 20).sum(),
                     h1=hh[0] if len(hh) else 0, h2=hh[1] if len(hh) > 1 else 0, h4=hh[3] if len(hh) > 3 else 0,
                     near03=(np.abs(big - t[0]) <= 0.3).mean() if len(big) else np.nan,      # 큰 타격 중 1위 ±0.3 s 안 비율
                     span=(big.max() - big.min()) if len(big) > 1 else 0,                     # 큰 타격이 퍼진 시간
                     lp_pct=np.mean([(lp < lp[int(tt * FS)]).mean() for tt in big]) if len(big) else np.nan,  # 타격 때 출렁임 백분위
                     lp_rms=LP[i].std(), t_big=";".join(f"{x:.2f}" for x in np.sort(big))))
T = pd.DataFrame(rows).set_index("f"); T.to_csv(os.path.join(FIG, "hits.csv"))
H = [np.sort(events(i)[1])[::-1] for i in range(len(y))]

# %% 요약
out = ["## 1. 큰 타격 개수 (문턱 격자)"]
for th in (15, 20, 30, 50):
    n = np.array([(h >= th).sum() for h in H])
    for lab, m in (("정상", y == 0), ("고장", y == 1)):
        c = n[m]; out.append(f"{th}σ {lab}: 0개 {(c==0).sum():3d} 1개 {(c==1).sum():3d} 2–3개 {((c>=2)&(c<=3)).sum():3d} 4개+ {(c>=4).sum():3d}")
out.append("\n## 2. k번째 큰 타격 높이 (/σ)")
for k in (1, 2, 3, 4, 5):
    v = np.array([h[k-1] if len(h) >= k else 0 for h in H]); vn = v[y == 0]; va = v[y == 1]
    out.append(f"k={k}: 정상 중앙 {np.median(vn):6.1f} 95% {np.percentile(vn,95):6.1f} 최대 {vn.max():6.1f} | 고장 중앙 {np.median(va):6.1f} | "
               f"정상 95% 넘는 고장 {(va>np.percentile(vn,95)).sum()}/30, 정상 최대 넘는 고장 {(va>vn.max()).sum()}/30")
out.append("\n## 3. 타격 시점 (20σ 이상 2개 넘는 파일)")
for lab, m in (("정상", T.y == 0), ("고장", T.y == 1)):
    d = T[m]; d2 = d[d.n20 >= 2]
    out.append(f"{lab}: 1위 ±0.3 s 안 비율 중앙 {d2.near03.median():.2f} | 퍼진 시간 중앙 {d2.span.median():.2f} s | "
               f"타격 때 출렁임 백분위 중앙 {d.lp_pct.median():.2f} | 저주파 RMS–큰 타격 수 스피어만 {spearmanr(d.lp_rms, d.n20)[0]:.2f}")
out.append("\n## 4. 큰 타격(20σ) 4개 이상 정상")
out.append(T[(T.y == 0) & (T.n20 >= 4)].sort_values("h4", ascending=False).round(2).to_string())
open(os.path.join(FIG, "normal_hits.txt"), "w", encoding="utf-8").write("\n".join(out))
