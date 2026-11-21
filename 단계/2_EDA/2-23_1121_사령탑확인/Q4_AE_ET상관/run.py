# %% [markdown]
# Q4 AE z x ET(ext150) z 상관 (사령탑 확인 요청, 11-21)
# - train OOF: 4등분해서 3으로 학습, 1을 채점 (src/final.py Core.fit과 같은 방식). 모델·src는 고치지 않고 Core의 내부 함수를 읽어 쓴다
# - 구성 두 가지: (a) ET 입력 = ext150 (ring=False, 요청한 것) / (b) ET 입력 = ext150 + 울림 2열 (ring=True, 현재 최종 모델)
# - 분할 seed 0·1·2 각각의 OOF 상관과 부트스트랩 95% 구간(파일 다시 뽑기 1000회)을 낸다. 라벨은 그룹 나누기에만 쓴다
# - z = (점수 - 정상 중앙값) / (정상 90% 선 - 정상 중앙값). 상관은 z 선형 변환에 안 변해서 원 점수와 같음
# 실행: conda run -n qe-phm python run.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import StratifiedKFold
from src.exp import load_train
from src.final import signal_tables, Core, take
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
NOR, ABN = "#2a78d6", "#eb6834"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
T, cols = signal_tables(X)
n = len(y)

def oof_z(ring, seed):
    core = Core(ring=ring)
    raw = np.empty((n, 2))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=seed).split(np.arange(n), y):
        raw[b] = core._raw(core._experts(take(T, a), y[a]), take(T, b))
    nz = raw[y == 0]; med = np.median(nz, 0)
    d = np.abs(np.percentile(nz, 90, 0) - med)
    sc = np.where(d > 1e-9, d, nz.std(0)) + 1e-6
    return (raw - med) / sc, raw

rng = np.random.default_rng(0)
def boot(a, b, k=1000):
    r = []
    for _ in range(k):
        i = rng.integers(0, len(a), len(a))
        r.append(spearmanr(a[i], b[i])[0])
    return np.percentile(r, [2.5, 97.5])

out = ["상관 = AE z x ET z (train OOF). Pearson / Spearman [Spearman 95% 부트스트랩 구간]", ""]
res = {}
for ring, tag in ((False, "ext150"), (True, "ext150+울림2열(최종)")):
    for seed in (0, 1, 2):
        Z, raw = oof_z(ring, seed); res[(tag, seed)] = (Z, raw)
        row = []
        for g, mk in (("전체", np.ones(n, bool)), ("정상만", y == 0), ("고장만", y == 1)):
            pr = pearsonr(Z[mk, 0], Z[mk, 1])[0]; sr = spearmanr(Z[mk, 0], Z[mk, 1])[0]; lo, hi = boot(Z[mk, 0], Z[mk, 1], 300 if seed else 1000)
            row.append(f"{g} P {pr:+.2f} / S {sr:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        out.append(f"{tag} seed {seed}: " + " | ".join(row))
    out.append("")

# seed 평균 요약 + 고장 중 규칙 안 켜진 것은 따로 안 나눔(규칙 판정은 학습 전체 기준이라 OOF와 섞임). 대신 고장 중 ET 점수가 낮은 파일(놓친 고장) 확인
for tag in ("ext150", "ext150+울림2열(최종)"):
    S = {g: np.mean([spearmanr(res[(tag, s)][0][mk, 0], res[(tag, s)][0][mk, 1])[0] for s in (0, 1, 2)]) for g, mk in (("전체", np.ones(n, bool)), ("정상만", y == 0), ("고장만", y == 1))}
    out.append(f"{tag} seed 평균 Spearman: " + ", ".join(f"{g} {v:+.2f}" for g, v in S.items()))
out.append("참고(연결 파일 2절): AE x ET(D+) 상관 0.81, 정상만 0.67, 고장만 0.49")

# 두 전문가가 서로 다른 고장을 잡는가: seed 0 기준, z>=1(정상 90% 선) 한쪽만 넘는 고장
for tag in ("ext150", "ext150+울림2열(최종)"):
    Z = res[(tag, 0)][0]; ab = y == 1
    both = ((Z[:, 0] >= 1) & (Z[:, 1] >= 1) & ab).sum(); ae_only = ((Z[:, 0] >= 1) & (Z[:, 1] < 1) & ab).sum()
    et_only = ((Z[:, 0] < 1) & (Z[:, 1] >= 1) & ab).sum(); none = ((Z[:, 0] < 1) & (Z[:, 1] < 1) & ab).sum()
    out.append(f"{tag} seed 0, 고장 30개를 z 1 기준으로: 둘 다 {both} · AE만 {ae_only} · ET만 {et_only} · 둘 다 아님 {none}")
    out.append("   AE만: " + ", ".join(np.array(names)[(Z[:, 0] >= 1) & (Z[:, 1] < 1) & ab]) + " | ET만: " + ", ".join(np.array(names)[(Z[:, 0] < 1) & (Z[:, 1] >= 1) & ab]) + " | 둘 다 아님: " + ", ".join(np.array(names)[(Z[:, 0] < 1) & (Z[:, 1] < 1) & ab]))
    nn = (y == 0)
    out.append(f"   정상 120개: 둘 다 {((Z[:,0]>=1)&(Z[:,1]>=1)&nn).sum()} · AE만 {((Z[:,0]>=1)&(Z[:,1]<1)&nn).sum()} · ET만 {((Z[:,0]<1)&(Z[:,1]>=1)&nn).sum()}")
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))
rows = []
for (tag, seed), (Z, raw) in res.items():
    for i in range(n): rows.append((tag, seed, names[i], y[i], Z[i, 0], Z[i, 1]))
pd.DataFrame(rows, columns=["config", "seed", "file", "y", "ae_z", "et_z"]).to_csv(os.path.join(HERE, "oof_z.csv"), index=False, encoding="utf-8-sig")

# %% 그림: 산점도 (seed 0) 두 구성 + 상관 막대
fig, axs = plt.subplots(1, 3, figsize=(17, 5.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for ax, tag in zip(axs[:2], ("ext150", "ext150+울림2열(최종)")):
    Z = res[(tag, 0)][0]
    for k, col, lab in ((0, NOR, "정상 120"), (1, ABN, "고장 30")):
        mk = y == k
        ax.scatter(Z[mk, 0], Z[mk, 1], s=18 if k == 0 else 40, c=col, alpha=.55 if k == 0 else .95, label=lab)
        c = np.polyfit(Z[mk, 0], Z[mk, 1], 1); xs = np.linspace(Z[mk, 0].min(), Z[mk, 0].max(), 20)
        ax.plot(xs, np.polyval(c, xs), color=col, lw=1.4)
    ax.axvline(1, color="gray", lw=.6, ls="--"); ax.axhline(1, color="gray", lw=.6, ls="--")
    sr = [spearmanr(Z[y == k, 0], Z[y == k, 1])[0] for k in (0, 1)]
    ax.set_title(f"{tag}\nSpearman 정상 {sr[0]:+.2f} · 고장 {sr[1]:+.2f} (seed 0, 직선 = 그룹별 회귀)", fontsize=10)
    ax.set_xlabel("AE z"); ax.set_ylabel("ET z"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax = axs[2]; xl = []; w = .25
for j, tag in enumerate(("ext150", "ext150+울림2열(최종)")):
    for gi, (g, mk) in enumerate((("정상만", y == 0), ("고장만", y == 1))):
        v = [spearmanr(res[(tag, s)][0][mk, 0], res[(tag, s)][0][mk, 1])[0] for s in (0, 1, 2)]
        ax.bar(gi + (j - .5) * w * 1.1, np.mean(v), w, color=[NOR, ABN][gi], alpha=.55 if j == 0 else 1, edgecolor="k")
        ax.errorbar(gi + (j - .5) * w * 1.1, np.mean(v), yerr=[[np.mean(v) - min(v)], [max(v) - np.mean(v)]], color="k", capsize=3)
ax.set_xticks([0, 1], ["정상만", "고장만"]); ax.set_ylabel("Spearman (seed 3개 평균, 막대 = 최소~최대)"); ax.set_ylim(0, 1)
ax.set_title("연한 막대 = ext150, 진한 막대 = +울림 2열", fontsize=10); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q4_corr.png"), dpi=90); plt.close(fig)
print("done")
