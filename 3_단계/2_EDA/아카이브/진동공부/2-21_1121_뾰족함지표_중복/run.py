# %% [markdown]
# 2-21 뾰족함 지표 중복 — 첨도·파고율·충격 수는 사실상 같은 것(충격이 있었나)을 여러 번 재는가
# - 배운 것: 시간 그래프 모양(Stage 1). 첨도 = 큰 값이 드물게 튀는 정도, 파고율 = 최고값 ÷ 평균 크기
# - 대상 (라벨 무관하게 이름으로 고름): ET 입력 150개 중 첨도·파고율·충격계수·여유율·TKEO·창 최대/평균 계열 전부 +
#   충격 수(울림 검출) · 큰 타격 수(바탕의 20배 이상) · 두 번째 타격 크기(유격 규칙 K1) · 1 kHz 고역통과 첨도
# - 같은 묶음 = 순위상관 |ρ| ≥ 0.8 (4-3 중복 지도와 같은 기준), 완전 연결 군집
# - AUC는 "사후" 관찰. 묶음마다 하나만 남겨도 정보가 거의 그대로인지 보는 것이 목적
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.dirname(HERE))
from _common import *
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from src.exp import cached_table
from src.features import feature_table_ext, feature_table_hits, feature_table_knock
from src.final import ring_table

X, y, names, log, save, flush = setup(HERE)
nor, abn = y == 0, y == 1

# %% 표 모으기
E = cached_table("ext", feature_table_ext, X)
pat = r"kurt|crest|impulse|clearance|max_to_mean|tkeo_max|tkeo_kurt"
cols = [c for c in E.columns if pd.Series(c).str.contains(pat).item()]
T = E[cols].copy()
Hh = cached_table("hits", feature_table_hits, X)
Kk = cached_table("knock", feature_table_knock, X)
_, _, NIMP = ring_table(X)
Xc = X - X.mean(1, keepdims=True)
H1 = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
T["충격수"] = NIMP
T["큰타격수_n20"] = Hh["HIT_n20"].values
T["두번째타격_K1"] = Kk["K1_second_snr"].values
T["첨도_1k고역"] = kurtosis(H1, axis=1, fisher=False)
log(f"뾰족함 계열 {T.shape[1]}개 (ET {len(cols)} + 충격 수·큰 타격 수·K1·1 kHz 고역통과 첨도)")

# %% 순위상관 · 묶음
rho = T.rank().corr().values
D = 1 - np.abs(rho); np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), "complete")
grp = fcluster(Z, 0.2, "distance")
order = leaves_list(Z)
auc = np.array([roc_auc_score(y, T[c]) for c in T.columns])
sep = np.abs(auc - .5) + .5
log(f"|ρ| ≥ 0.8 묶음 수: {grp.max()} (지표 {T.shape[1]}개)")
G = pd.DataFrame(dict(col=T.columns, grp=grp, auc=auc, sep=sep))
for g, d in G.sort_values("sep", ascending=False).groupby("grp", sort=False):
    best = d.iloc[0]
    log(f"  묶음 {g:2d} ({len(d):2d}개) 대표 {best.col} AUC {best.auc:.2f} | " + ", ".join(d.col.tolist()[:8]) + (" …" if len(d) > 8 else ""))
for a, b in (("충격수", "첨도_1k고역"), ("충격수", "T07_kurtosis"), ("T07_kurtosis", "T09_crest"), ("큰타격수_n20", "충격수"), ("두번째타격_K1", "T09_crest")):
    log(f"ρ({a}, {b}) = {spearmanr(T[a], T[b]).correlation:.2f}")

# %% 그림 1: 상관 열지도 (군집 순)
lab = [c.replace("_kurtosis", "_첨도").replace("_crest", "_파고율") for c in T.columns[order]]
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(np.abs(rho[np.ix_(order, order)]), cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=90, fontsize=6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{l}  [{grp[o]}]" for l, o in zip(lab, order)], fontsize=6)
fig.colorbar(im, ax=ax, label="|순위상관|", shrink=.7)
ax.set_title(f"뾰족함 지표 {T.shape[1]}개의 서로 닮은 정도 — 밝은 사각형 = 같은 것을 재는 묶음 ([ ] = 묶음 번호, |ρ| ≥ 0.8 → {grp.max()}묶음)")
save(fig, "1_상관열지도_묶음")

# %% 그림 2: 대표 셋 산점도
pairs = (("충격수", "첨도_1k고역"), ("T07_kurtosis", "T09_crest"), ("충격수", "두번째타격_K1"))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (a, b) in zip(axs, pairs):
    for m, c, l in ((nor, NOR, "정상"), (abn, ABN, "고장")):
        ax.scatter(T[a][m] + (np.random.default_rng(0).uniform(-.3, .3, m.sum()) if a == "충격수" else 0), T[b][m], s=14, color=c, alpha=.7, label=l)
    for n_ in ("abn_002", "abn_014", "abn_017", "abn_022", "nor_116"):
        i = names.index(n_); ax.annotate(n_.replace("abn_", "A").replace("nor_", "N"), (T[a][i], T[b][i]), fontsize=8)
    ax.set_yscale("log"); plain_log(ax)
    if a != "충격수":
        ax.set_xscale("log"); plain_log(ax, "x")
    ax.set_xlabel(a); ax.set_ylabel(b); ax.grid(alpha=.3, which="both")
    ax.set_title(f"순위상관 {spearmanr(T[a], T[b]).correlation:.2f}")
axs[0].legend()
fig.suptitle("뾰족함 지표끼리: 한 줄로 늘어서면 같은 정보", y=1.02)
save(fig, "2_대표지표_산점도")

# %% 그림 3: 묶음별 가르는 힘 (사후)
Gs = G.sort_values(["grp", "sep"], ascending=[True, False])
fig, ax = plt.subplots(figsize=(14, 4.5))
xs = np.arange(len(Gs))
cmap = plt.get_cmap("tab20")
ax.bar(xs, Gs.sep, color=[cmap((g - 1) % 20) for g in Gs.grp])
ax.axhline(.5, color="gray", ls=":")
ax.set_xticks(xs); ax.set_xticklabels([c.replace("_kurtosis", "_첨도").replace("_crest", "_파고율") for c in Gs.col], rotation=90, fontsize=6)
ax.set_ylim(.45, 1); ax.set_ylabel("가르는 힘 (AUC, 0.5 = 못 가름)")
ax.set_title("같은 색 = 같은 묶음. 묶음 안 막대가 비슷하면 하나만 남겨도 된다 (사후)")
save(fig, "3_묶음별_가르는힘")

# %% 첨도·충격 수로 못 잡는 고장
log("\n특이 파일: 정상 백분위 (충격수 · 1 kHz 첨도 · 원신호 첨도 · 파고율)")
for n_ in ("abn_002", "abn_014", "abn_017", "abn_022", "abn_015"):
    i = names.index(n_)
    log(f"  {n_}: " + " · ".join(f"{pct_in_normal(T[c].values, y, i):.0f}%" for c in ("충격수", "첨도_1k고역", "T07_kurtosis", "T09_crest")))
G.to_csv(os.path.join(HERE, "groups.csv"), index=False, encoding="utf-8-sig")
flush()
