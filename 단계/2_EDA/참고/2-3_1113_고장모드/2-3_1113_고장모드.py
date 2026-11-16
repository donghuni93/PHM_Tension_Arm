# %% [markdown]
# # 2-3 고장 모드 찾기 (고장 30개를 신호 특징으로 묶기)
# 목적: 고장 30개가 한 종류인지 여러 종류인지 본다 → 단계/1_기준·기록/개념_도메인·평가.md의 결함 메커니즘(경화·박리·유격 …)에 연결
# 방법
# - 해석 가능한 기술 지표 11개 (대역은 물리 기준 고정: 저 <300 Hz 구조·차체, 중 300–5k 조립부 고차 모드, 고 5k–12.5k)
# - 각 지표를 **정상 120개 기준 로버스트 z** = (x − 정상 중앙값) / (정상 IQR / 1.349) 로 바꾼다 → 0 = 정상 한가운데, ±2 ≈ 정상 바깥
# - 고장 30개만 Ward 계층 군집 → 히트맵 + 덴드로그램
# - 설명용 EDA. 이 결과로 모델 피처 대역을 고르지 않는다 (작업 원칙 3)

# %%
import sys
from pathlib import Path
ROOT_DIR = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
                if (p / "src" / "exp.py").exists())
sys.path.insert(0, str(ROOT_DIR))
from src.exp import *
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import kurtosis
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = ROOT_DIR / "단계" / "2_EDA" / "참고" / "2-3_1113_고장모드"; FIGS = HERE / "figs"
FS = 25000
X, y, names = load_train()
Xc = X - X.mean(1, keepdims=True)

# %% 기술 지표
f, P = welch(Xc, fs=FS, nperseg=8192)
df = f[1] - f[0]
band_rms = lambda lo, hi: np.sqrt(P[:, (f >= lo) & (f < hi)].sum(1) * df)
hp = sosfiltfilt(butter(4, 300, "highpass", fs=FS, output="sos"), Xc, axis=1)    # 300 Hz 이상 (충격·공진 성분)
win = int(0.02 * FS)
wrms = np.sqrt((hp[:, : hp.shape[1] // win * win].reshape(len(X), -1, win) ** 2).mean(2))   # 20 ms 창 RMS

m_hi = (f >= 300)
Ph = P[:, m_hi]; fh = f[m_hi]
m5 = (f >= 5000)
D = pd.DataFrame({
    "name": names, "y": y,
    "평균 g": X.mean(1),
    "저대역 RMS": np.log10(band_rms(2, 300)),
    "중대역 RMS": np.log10(band_rms(300, 5000)),
    "고대역 RMS": np.log10(band_rms(5000, 12500)),
    "고/중 비율": np.log10(band_rms(5000, 12500) / band_rms(300, 5000)),
    "무게중심 Hz": (Ph * fh).sum(1) / Ph.sum(1),                      # 300 Hz 이상 스펙트럼 무게중심
    "고대역 피크 Hz": f[m5][P[:, m5].argmax(1)],
    "피크 돌출도": np.log10(P[:, m_hi].max(1) / np.median(P[:, m_hi], 1)),   # 가장 큰 피크 / 대역 중앙값
    "첨도(>300Hz)": np.log10(kurtosis(hp, axis=1) + 3),               # 정규 = log10(3)
    "파고율(>300Hz)": np.abs(hp).max(1) / hp.std(1),
    "간헐성(창RMS CV)": wrms.std(1) / wrms.mean(1),                   # 충격이 몰려 있으면 큼
})
COLS = [c for c in D.columns if c not in ("name", "y")]
nor = D[D.y == 0][COLS]
Z = (D[COLS] - nor.median()) / ((nor.quantile(.75) - nor.quantile(.25)) / 1.349)
Z.insert(0, "name", names); Z.insert(1, "y", y)
D.to_csv(FIGS / "2-3_descriptors.csv", index=False, encoding="utf-8-sig")
Z.to_csv(FIGS / "2-3_robust_z.csv", index=False, encoding="utf-8-sig")

# %% 고장 30개 군집
Za = Z[Z.y == 1].set_index("name")[COLS]
Zc = Za.clip(-6, 6)                                   # 극단값 하나가 군집을 지배하지 않게
CL = [c for c in COLS if c != "평균 g"]                 # 평균(중력·장착 방향)은 결함 물리가 아니므로 군집 기준에서 뺌, 표시만
L = linkage(Zc[CL].values, "ward")
out = []
for k in (3, 4, 5, 6):
    lab = fcluster(L, k, "maxclust")
    out.append(f"k={k}: " + " | ".join(f"[{c}] " + ",".join(n[4:] for n in Za.index[lab == c]) for c in np.unique(lab)))
K = 5
lab = fcluster(L, K, "maxclust")
Za_out = Za.assign(군집=lab)
summ = Za_out.groupby("군집").median().round(1)
summ.insert(0, "파일", Za_out.groupby("군집").apply(lambda g: ",".join(n[4:] for n in g.index)))

# 각 군집의 중앙 프로파일과 가장 가까운 정상 파일 (오경보 위험)
Zn = Z[Z.y == 0].set_index("name")[COLS].clip(-6, 6)
near = {c: ", ".join(Zn.sub(Zc[lab == c].median(), axis=1).pow(2).sum(axis=1).nsmallest(3).index) for c in np.unique(lab)}
outside = (Zc[CL].abs() > 2).sum(axis=1)                       # 정상 바깥(|z|>2) 지표 수

with open(FIGS / "2-3_fault_modes.txt", "w", encoding="utf-8") as fh_:
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    fh_.write("== 군집 수별 구성\n" + "\n".join(out) + "\n\n")
    fh_.write(f"== k={K} 군집별 로버스트 z 중앙값\n" + summ.to_string() + "\n\n")
    fh_.write("== 군집 프로파일과 가장 닮은 정상 3개\n" + "\n".join(f"[{c}] {v}" for c, v in near.items()) + "\n\n")
    fh_.write("== 파일별 |z|>2 지표 수 (0 = 모든 지표가 정상 범위)\n" + outside.sort_values().to_string() + "\n\n")
    fh_.write("== 정상 120개에서 |z|>2 지표 수 분포\n" + (Zn[CL].abs() > 2).sum(axis=1).value_counts().sort_index().to_string() + "\n")

# %% 그림: 덴드로그램 + 히트맵 (행 정렬 맞춤)
n = len(Za)
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[30, 1], wspace=.02, hspace=.35)
ad = fig.add_subplot(gs[0, 0])
dn = dendrogram(L, orientation="left", no_labels=True, ax=ad, color_threshold=L[-(K - 1), 2],
                above_threshold_color="#999999")
ad.set_ylim(10 * n, 0); ad.set_axis_off()          # 잎 i는 y = 5 + 10i → 위에서부터 order 순
order = dn["leaves"]
ah = fig.add_subplot(gs[0, 1])
M = Zc.values[order]
im = ah.imshow(M, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto", extent=(-.5, len(COLS) - .5, n - .5, -.5))
ah.set_xticks(range(len(COLS))); ah.set_xticklabels([c + (" *" if c not in CL else "") for c in COLS],
                                                     rotation=35, ha="right", fontsize=9)
ah.set_yticks(range(n)); ah.set_yticklabels([f"{Za.index[i]}  [{lab[i]}]" for i in order], fontsize=8)
ah.yaxis.tick_right()
for r in range(n):
    for c in range(len(COLS)):
        ah.text(c, r, f"{M[r, c]:.0f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(M[r, c]) > 3.5 else "#222222")
ah.axvline(.5, color="k", lw=.8)
cax = fig.add_subplot(gs[1, 1])
fig.colorbar(im, cax=cax, orientation="horizontal", label="정상 120개 기준 로버스트 z (±6에서 자름, |z|>2 = 대략 정상 바깥)")
fig.suptitle(f"고장 30개 기술 지표와 Ward 군집 {K}개 ([ ] = 군집 번호, * = 군집 기준에서 제외)", fontsize=11, x=.01, ha="left")
fig.savefig(FIGS / "2-3_fault_modes_heatmap.png", dpi=120, bbox_inches="tight")
