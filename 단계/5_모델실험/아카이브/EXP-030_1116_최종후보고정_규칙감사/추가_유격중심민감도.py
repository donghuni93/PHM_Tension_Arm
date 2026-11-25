# %% [markdown]
# EXP-030 추가 확인 (11-16, run.py 기록 뒤): 유격 반복 규칙의 '정상 중심'은 어떤 정상으로 잡느냐에 얼마나 민감한가
# - 계기: 최종 모델 학습(단계/7_최종·제출/1_학습.py)의 임계값용 OOF에서, 일부 분할에서만 정상 001·093·106이 규칙에 켜지고
#   고장 014가 꺼졌다. 규칙의 r은 학습 쪽 정상의 중앙값·MAD로 맞춘 값이라, 정상 표본이 바뀌면 r도 바뀐다
# - 방법 (학습 모델 없음, 라벨은 집계에만): 정상 120개에서 n개를 뽑아 중심·퍼짐을 다시 잡고 150개 전체에 규칙을 적용 → 2,000번 반복
#   n = 72(중첩 CV 안쪽) · 90(최종 모델의 임계값용 4-fold) · 96(중첩 CV 바깥) · 120 복원추출(부트스트랩 = 최종 모델 중심의 추정 오차)
# - run.py 결과는 바꾸지 않는다. 출력: figs/knock_center.txt · figs/knock_center.png
# 실행: conda run -n qe-phm python "단계/5_모델실험/아카이브/EXP-030_1116_최종후보고정_규칙감사/추가_유격중심민감도.py"

# %%
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.features import feature_table_knock
import matplotlib.pyplot as plt

X, y, names = load_train(); names = [str(n) for n in names]
K = cached_table("knock", feature_table_knock, X).values
nor = np.flatnonzero(y == 0)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
R_MIN, RATIO_MIN, B = 2.0, 0.7, 2000


def rule(ref):
    """ref = 중심을 잡는 정상 파일 번호 → (150개의 r, 켜짐 여부, 퍼짐 2개)"""
    Kn = K[ref][:, :2]
    med = np.median(Kn, 0); mad = 1.4826 * np.median(np.abs(Kn - med), 0) + 1e-12
    r = np.sqrt((((K[:, :2] - med) / mad) ** 2).mean(1))
    return r, (r >= R_MIN) & (K[:, 2] >= RATIO_MIN), mad


r0, f0, mad0 = rule(nor)
L = [f"기준 (정상 120개 전체): 퍼짐 K1 {mad0[0]:.3f} · K2 {mad0[1]:.3f} · 켜짐 고장 {int(f0[y == 1].sum())} · 정상 {int(f0[y == 0].sum())}",
     f"K1이 바닥값 log10(8)인 정상(8σ 사건이 2개 미만) = {int((np.abs(K[nor, 0] - np.log10(8)) < 1e-9).sum())}/120 "
     f"→ K1 분포의 한 점에 값이 몰려 있어 중앙값·MAD가 표본에 따라 계단처럼 변한다", ""]
rng = np.random.default_rng(0)
CASES = [("72개 비복원 (중첩 CV 안쪽)", 72, False), ("90개 비복원 (최종 모델 임계값용 4-fold)", 90, False),
         ("96개 비복원 (중첩 CV 바깥)", 96, False), ("120개 복원 (부트스트랩)", 120, True)]
FREQ, MADS = {}, {}
for lab, n, rep in CASES:
    cnt, mads = np.zeros(len(y)), []
    for _ in range(B):
        _, f, mad = rule(rng.choice(nor, n, replace=rep)); cnt += f; mads.append(mad)
    FREQ[lab], MADS[lab] = cnt / B, np.array(mads)
    q = np.percentile(MADS[lab], [5, 50, 95], 0)
    nn = [(FREQ[lab][i], names[i]) for i in nor if FREQ[lab][i] >= 0.01]
    ff = [(FREQ[lab][i], names[i]) for i in np.flatnonzero(y == 1) if 0.01 <= FREQ[lab][i] < 0.99 or (f0[i] and FREQ[lab][i] < 0.99)]
    L += [f"[{lab}] 퍼짐 5–50–95%: K1 {q[0, 0]:.3f}–{q[1, 0]:.3f}–{q[2, 0]:.3f} · K2 {q[0, 1]:.3f}–{q[1, 1]:.3f}–{q[2, 1]:.3f}",
          "  1% 이상 켜지는 정상: " + (" ".join(f"{short(nm)} {v:.0%}" for v, nm in sorted(nn, reverse=True)) or "없음"),
          f"  표본 하나에서 켜지는 정상 수의 기대값: {FREQ[lab][nor].sum():.2f}개 / 120",
          "  켜졌다 꺼졌다 하는 고장: " + (" ".join(f"{short(nm)} {v:.0%}" for v, nm in sorted(ff)) or "없음"), ""]
open(os.path.join(HERE, "figs", "knock_center.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림: 파일별 켜지는 비율 (부트스트랩) — 기준 r 순서
lab = CASES[3][0]; fr = FREQ[lab]
show = np.flatnonzero((fr >= 0.01) | f0)
show = show[np.argsort(-fr[show])]
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(range(len(show)), fr[show], color=["#eb6834" if y[i] else "#2a78d6" for i in show])
ax.set_xticks(range(len(show)), [f"{short(names[i])}\nr {r0[i]:.1f}\n{K[i, 2]:.2f}" for i in show], fontsize=7.5)
ax.axhline(1, color="k", lw=.5); ax.set_ylabel("규칙이 켜지는 비율 (정상 120개를 복원추출해 중심을 다시 잡은 2,000번)")
ax.set_title("유격 반복 규칙: 정상 중심을 다시 추정하면 누가 켜지나 (주황 = 고장, 파랑 = 정상)\n막대 아래 = 파일 · 기준 r · 2위/1위", fontsize=10)
ax.grid(alpha=.3, axis="y"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "knock_center.png"), dpi=80)
