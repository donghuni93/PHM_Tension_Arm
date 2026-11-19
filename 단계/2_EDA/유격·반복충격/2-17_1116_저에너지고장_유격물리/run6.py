# %% [markdown]
# 2-17 6차 — 날카로움(5차 SE·RT)이 유격 규칙의 여유를 넓히는가
# 현행 유격 규칙 (EXP-025): r ≥ 2 그리고 K3(2위/1위) ≥ 0.7 → 고장. r = K1·K2 정상 로버스트 z의 RMS. 2·0.7은 라벨을 본 뒤 정한 값
# 여유 경고 (EXP-030): 문턱 −14%에서 정상 009, −19%에서 001이 켜짐. 정상 중심 복원추출 시 정상 001·093·106이 22–33% 켜짐
# 질문
#   1. 규칙 근처(r ≥ 1.5) 정상과 규칙 고장을 날카로움으로 가를 수 있나
#   2. "날카롭다" 조건(SE > 정상 중앙값, 라벨 무관: 정상만으로 정함)을 더하면 문턱을 낮출 때·중심을 흔들 때 정상이 덜 켜지나
#   3. 라벨로 고른 K3 ≥ 0.7 대신 날카로움을 쓰면 (r ≥ 2 그리고 SE > 정상 중앙값) 어떤가
# 문턱 스캔은 r·K3 문턱을 같은 비율로 낮춘다 (EXP-030과 같은 방식). SE 조건은 고정
# 실행: conda run -n qe-phm python "단계/2_EDA/유격·반복충격/2-17_1116_저에너지고장_유격물리/run6.py"  (5차 figs5/files.csv 필요)

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
from src.exp import load_train
from src.features import feature_table_knock
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs6"); os.makedirs(FIG, exist_ok=True)
NOR, ABN = "#2a78d6", "#eb6834"

X, y, names = load_train(); y = np.asarray(y); N = len(y)
short = np.array([os.path.splitext(os.path.basename(str(n)))[0] for n in names])
lab = lambda s: s.replace("abn_", "A").replace("nor_", "N")
log = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s); log.append(s)

K = feature_table_knock(X).values                      # K1, K2, K3
S = pd.read_csv(os.path.join(HERE, "figs5", "files.csv")).set_index("file").loc[short]
SE, RT = S.SE.values, S.RT.values
nor = y == 0

def r_of(Kk, ref):
    med = np.median(Kk[ref][:, :2], 0); mad = 1.4826 * np.median(np.abs(Kk[ref][:, :2] - med), 0) + 1e-12
    Z = (Kk[:, :2] - med) / mad
    return np.sqrt((Z ** 2).mean(1))
r = r_of(K, nor); K3 = K[:, 2]
SE_MED, RT_MED = np.nanmedian(SE[nor]), np.nanmedian(RT[nor])
sharp = np.nan_to_num(SE, nan=0) > SE_MED                # 날카롭다 (정상 중앙보다)
sharp_rt = np.nan_to_num(RT, nan=9) < RT_MED             # 반상승이 정상 중앙보다 짧다
say(f"정상 중앙: SE {SE_MED:.3f} · RT {RT_MED:.3f} ms  (라벨 무관 문턱 = 정상만으로)")

D = pd.DataFrame(dict(file=short, y=y, r=r, K3=K3, SE=SE, RT=RT, sharp_SE=sharp, sharp_RT=sharp_rt))
D.to_csv(os.path.join(FIG, "files.csv"), index=False, encoding="utf-8-sig")

# %% 1. 규칙 근처
say("\n== 1. 규칙 근처 (r ≥ 1.5) — r 순 ==")
say(D[D.r >= 1.5].sort_values("r", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% 2. 규칙 정의 비교: 문턱 스캔
RULES = {
    "현행 r≥2 & K3≥0.7": lambda f, rr: (rr >= 2 * f) & (K3 >= 0.7 * f),
    "현행 + SE 날카로움": lambda f, rr: (rr >= 2 * f) & (K3 >= 0.7 * f) & sharp,
    "현행 + RT 날카로움": lambda f, rr: (rr >= 2 * f) & (K3 >= 0.7 * f) & sharp_rt,
    "r≥2 & SE 날카로움 (K3 없음)": lambda f, rr: (rr >= 2 * f) & sharp,
    "r≥2 & SE·RT 둘 다 (K3 없음)": lambda f, rr: (rr >= 2 * f) & sharp & sharp_rt,
}
say("\n== 2. 문턱 스캔 (r·K3 문턱 × f, 날카로움 조건 고정) — 고장 수 / 정상 수 [켜진 정상] ==")
FS_ = [1.0, 0.9, 0.8, 0.7, 0.6]
for nm, fn in RULES.items():
    parts = []
    for f in FS_:
        on = fn(f, r)
        parts.append(f"×{f}: {int((on & ~nor).sum())}/{int((on & nor).sum())}")
    say(f"  {nm:28} " + " · ".join(parts))
    on = fn(1.0, r)
    say(f"      ×1.0 고장: {' '.join(lab(s) for s in short[on & ~nor])}")
    # 첫 정상이 켜지는 f
    fz = next((f for f in np.arange(1.0, 0.39, -0.01) if (fn(f, r) & nor).any()), None)
    if fz is not None:
        say(f"      첫 정상 켜짐: ×{fz:.2f} ({' '.join(lab(s) for s in short[fn(fz, r) & nor])}) → 여유 {100 * (1 - fz):.0f}%")

# %% 3. 정상 중심 복원추출 (EXP-030과 같은 방식: 정상 120개를 복원추출해 중앙값·MAD 다시 잡기, 200번)
say("\n== 3. 정상 중심 복원추출 200번 — 규칙이 켜지는 비율 (정상은 >0인 것, 고장은 <100%인 것) ==")
rng = np.random.default_rng(0)
ni = np.flatnonzero(nor)
B = 200
hits = {nm: np.zeros(N) for nm in RULES}
for _ in range(B):
    ref = np.zeros(N, bool); ref_idx = rng.choice(ni, len(ni), replace=True)
    Kb = K.copy()
    med = np.median(K[ref_idx][:, :2], 0); mad = 1.4826 * np.median(np.abs(K[ref_idx][:, :2] - med), 0) + 1e-12
    rb = np.sqrt((((K[:, :2] - med) / mad) ** 2).mean(1))
    for nm, fn in RULES.items():
        hits[nm] += fn(1.0, rb)
for nm in RULES:
    h = hits[nm] / B
    nz = [(lab(short[i]), h[i]) for i in np.flatnonzero(nor & (h > 0))]
    ab = [(lab(short[i]), h[i]) for i in np.flatnonzero(~nor & (h > 0) & (h < 1))]
    say(f"  {nm:28} 정상 평균 오작동 {h[nor].mean() * 100:.2f}% · 켜진 정상 " + (" ".join(f"{a} {b:.0%}" for a, b in nz) or "없음")
        + " | 흔들리는 고장 " + (" ".join(f"{a} {b:.0%}" for a, b in ab) or "없음"))

# %% 그림: r × K3 평면, 점 크기/테두리 = 날카로움
fig, ax = plt.subplots(figsize=(9, 6.5))
for g, c in [(0, NOR), (1, ABN)]:
    s = y == g
    ax.scatter(r[s & ~sharp], K3[s & ~sharp], s=25, facecolors="none", edgecolors=c, label=f"{'정상' if g == 0 else '고장'} · 무딤")
    ax.scatter(r[s & sharp], K3[s & sharp], s=40, c=c, label=f"{'정상' if g == 0 else '고장'} · 날카로움 (SE > 정상 중앙)")
for i in np.flatnonzero(r >= 1.5):
    ax.annotate(lab(short[i]), (r[i], K3[i]), fontsize=7)
ax.axvline(2, color="k", ls="--"); ax.axhline(0.7, color="k", ls="--")
ax.axvline(2 * .8, color="0.6", ls=":"); ax.axhline(0.7 * .8, color="0.6", ls=":")
ax.set_xscale("log"); ax.set_xlabel("r (정상 중심 거리, K1·K2)"); ax.set_ylabel("K3 = 2위/1위 사건 높이")
ax.set_title("유격 규칙 평면 — 채운 점 = 날카로운 충격, 빈 점 = 무딤. 점선 = 현행 문턱, 회색 = −20%")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_규칙평면_날카로움.png"), dpi=80); plt.close(fig)

with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
