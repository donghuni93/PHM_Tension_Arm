# %% [markdown]
# 4-7. 2차 설계 피처 라벨 없는 점검 (11-21)
# 1) 충격 정의·S 재현 확인 (assert)  2) train 150 + val 40 스피어만 상관 → |ρ| >= 0.8 묶음마다 대표 하나 (라벨 안 씀)
# 3) 피처별 train 정상 5–95 % 띠 vs val 분포 (보고용 고장 AUC)  4) t-SNE (최종 피처 + L1 S, 강건 표준화) · 5-NN 숫자 (ext150과 나란히)
# 묶음 규칙(실행 전 고정): |ρ| >= 0.8 그래프의 연결 성분 = 묶음, 묶음 안 평균 |ρ|(다른 구성원과)가 가장 큰 피처 하나만 남김. 동률 = 앞쪽 열
# 결과 최종 피처 목록 -> 최종피처.json (EXP-040이 읽는다, 바꾸지 않는다)
# 실행: conda run -n qe-phm python "단계/4_피처/4-2_H피처목록/자료/run.py"
import os, sys, json, warnings
warnings.filterwarnings("ignore")
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.impacts import detect_impacts
from src.features2 import cached_feat2, PeakRef, robust_z, lock_S, H_FIXED, H_FOLD, L_COLS

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, PUR, GRAY = "#2a78d6", "#eb6834", "#7B1FA2", "#9aa0a6"
R = {}

# %% 데이터 · 피처 (캐시 outputs/cache/feat2_*)
X, y, names = load_train(); names = [str(n) for n in names]
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val")); nv = [str(n) for n in nv]
assert X.shape == (150, 50000) and Xv.shape == (40, 50000)
T, L, fr = cached_feat2("train", X); Tv, Lv, _ = cached_feat2("val", Xv)
nor = y == 0; n = len(y); df = float(fr[1] - fr[0])

# %% 재현 확인 (끝단 제외 전 시점으로 S, 충격 수)
Xa = np.vstack([X, Xv]); na = names + nv
S0 = pd.read_csv(os.path.join(_d, "단계", "4_피처", "4-1_충격정의", "자료", "사령탑_7b-7e", "S_새충격.csv"), encoding="utf-8-sig", index_col=0)
S_pre = np.array([lock_S(x, detect_impacts(x, edge_ms=0)["t"])[0] for x in Xa])
assert np.allclose(S_pre, S0.loc[na, "S"].values, atol=1e-6), "끝단 제외 전 S가 7b-7e와 다르다"
i22, i14 = na.index("abn_022"), na.index("abn_014")
SN = S_pre[:n][nor]
R["repro"] = dict(S_022_pre=float(S_pre[i22]), S_014_pre=float(S_pre[i14]), S_normal_max_pre=float(SN.max()), S_normal_max_file=names[int(np.flatnonzero(nor)[SN.argmax()])],
                  S_val_max_pre=float(S_pre[n:].max()), S_022_post=float(T.L1_S[i22]), S_014_post=float(T.L1_S[i14]), S_normal_max_post=float(T.L1_S[nor].max()),
                  S_val_max_post=float(Tv.L1_S.max()))
cnt = np.concatenate([np.expm1(T.H1_nimp.values), np.expm1(Tv.H1_nimp.values)]).round().astype(int)
grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * 40)
R["repro"]["n_impacts_post"] = {"normal": int(cnt[grp == "정상"].sum()), "val": int(cnt[grp == "val"].sum()), "abnormal": int(cnt[grp == "고장"].sum())}
assert R["repro"]["n_impacts_post"] == {"normal": 854, "val": 247, "abnormal": 402}
R["repro"]["n_impacts_abn_024"] = int(cnt[na.index("abn_024")]); R["repro"]["n_impacts_abn_015"] = int(cnt[na.index("abn_015")])
assert abs(R["repro"]["S_022_pre"] - 0.963) < 1e-3 and abs(R["repro"]["S_normal_max_pre"] - 0.449) < 1e-3 and abs(R["repro"]["S_014_pre"] - 0.546) < 1e-3
print("재현 확인", R["repro"], flush=True)

# %% H16–H17은 정상 전체(train 120) 기준 참고값, 후보 표 = train 150 + val 40
pr = PeakRef(df).fit(L[nor])
ph, phv = pr.transform(L), pr.transform(Lv)
A_tr = T.copy(); A_tr["H16_peak_h"], A_tr["H17_peak_w"] = ph[:, 0], ph[:, 1]
A_va = Tv.copy(); A_va["H16_peak_h"], A_va["H17_peak_w"] = phv[:, 0], phv[:, 1]
CAND = H_FIXED[:-1] + H_FOLD + H_FIXED[-1:] + L_COLS                 # H11은 제외됨: H 블록 17개(H1–H10, H12–H18) + L1
assert len(CAND) == 18
ALL = pd.concat([A_tr[CAND], A_va[CAND]], ignore_index=True)
print("후보", len(CAND), CAND, flush=True)

# %% 스피어만 상관 · 묶음 · 대표
C = ALL.corr(method="spearman"); Ca = C.abs().values; k = len(CAND)
adj = (Ca >= 0.8) & ~np.eye(k, dtype=bool)
comp = -np.ones(k, int); nc = 0
for s in range(k):
    if comp[s] >= 0: continue
    st = [s]; comp[s] = nc
    while st:
        u = st.pop()
        for v in np.flatnonzero(adj[u]):
            if comp[v] < 0: comp[v] = nc; st.append(v)
    nc += 1
groups, keep, drop = [], [], {}
for c in range(nc):
    m = np.flatnonzero(comp == c)
    if len(m) == 1:
        keep.append(CAND[m[0]]); continue
    mean_abs = [np.mean([Ca[i, j] for j in m if j != i]) for i in m]
    rep = int(m[int(np.argmax(mean_abs))])
    groups.append(dict(members=[CAND[i] for i in m], mean_abs={CAND[i]: round(float(a), 3) for i, a in zip(m, mean_abs)}, kept=CAND[rep]))
    keep.append(CAND[rep])
    for i in m:
        if i != rep: drop[CAND[i]] = CAND[rep]
final = [c for c in CAND if c in keep]
R["spearman"] = dict(candidates=CAND, groups=groups, dropped=drop, final=final, n_final=len(final))
pairs = [(CAND[i], CAND[j], round(float(C.values[i, j]), 3)) for i in range(k) for j in range(i + 1, k) if Ca[i, j] >= 0.8]
R["spearman"]["pairs_ge_0.8"] = pairs
json.dump(dict(final=final, hfold=[c for c in final if c in H_FOLD], fixed=[c for c in final if c in H_FIXED], lock=[c for c in final if c in L_COLS],
               dropped=drop, groups=groups), open(os.path.join(HERE, "최종피처.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("묶음", groups, "\n최종", final, flush=True)

fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(k), CAND, rotation=75, ha="right", fontsize=8); ax.set_yticks(range(k), CAND, fontsize=8)
for i in range(k):
    for j in range(k):
        v = C.values[i, j]
        ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("1.00", "1"), ha="center", va="center", fontsize=6.5, color="w" if abs(v) > .6 else "k",
                weight="bold" if abs(v) >= .8 and i != j else "normal")
fig.colorbar(im, shrink=.7); ax.set_title("스피어만 상관 (train 150 + val 40, 라벨 안 씀). 굵은 글씨 = |ρ| ≥ 0.8\nH16·H17은 정상 120 기준 참고값", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "spearman_heatmap.png"), dpi=100); plt.close(fig)

# %% 피처별 train 정상 5–95 % 띠 vs val, 보고용 고장 AUC
rows = []
for c in CAND:
    lo, hi = np.percentile(A_tr[c][nor], [5, 95]); vv = A_va[c].values
    rows.append(dict(feature=c, normal_p5=lo, normal_p95=hi, val_out=float(((vv < lo) | (vv > hi)).mean()), val_median=float(np.median(vv)), normal_median=float(np.median(A_tr[c][nor])),
                     fault_out=float(((A_tr[c][~nor] < lo) | (A_tr[c][~nor] > hi)).mean()), auc=float(roc_auc_score(y, A_tr[c]))))
B = pd.DataFrame(rows); B.round(3).to_csv(os.path.join(HERE, "피처_분포이동_AUC.csv"), index=False, encoding="utf-8-sig")
R["bands"] = B.round(3).to_dict("records")
fig, axs = plt.subplots(3, 6, figsize=(21, 11)); rg = np.random.default_rng(0)
for a, c in zip(axs.ravel(), CAND):
    r = B[B.feature == c].iloc[0]
    a.axhspan(r.normal_p5, r.normal_p95, color=NOR, alpha=.15, lw=0)
    for j, (v, col, s) in enumerate(((A_tr[c][nor], NOR, 14), (A_tr[c][~nor], ABN, 20), (A_va[c], PUR, 22))):
        a.scatter(j + rg.uniform(-.25, .25, len(v)), v, s=s, color=col, alpha=.7, lw=0)
    a.set_xticks([0, 1, 2], ["정상", "고장", "val"], fontsize=8); a.set_title(f"{c}\nval 띠 밖 {r.val_out:.0%} · AUC {r.auc:.2f}", fontsize=8)
for a in axs.ravel()[len(CAND):]: a.axis("off")
fig.suptitle("피처별 train 정상 5–95 % 띠(파란 영역)와 train 고장 · val 분포. val 띠 밖 기대치 = 10 %. AUC = 보고용(라벨 안 씀 선택)", x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(os.path.join(FIG, "band_val.png"), dpi=90); plt.close(fig)

# %% t-SNE · 5-NN 숫자 (강건 표준화: train 정상 중앙값·MAD×1.4826, MAD=0이면 robust_z의 대체 규모)
def standardize(Atr, Ava):
    Z = np.empty((len(Atr), Atr.shape[1])); Zv = np.empty((len(Ava), Atr.shape[1]))
    for j in range(Atr.shape[1]):
        f = robust_z(Atr[nor, j]); Z[:, j] = f(Atr[:, j]); Zv[:, j] = f(Ava[:, j])
    return Z, Zv


def knn_numbers(Z, Zv):
    nb = NearestNeighbors(n_neighbors=6).fit(Z)
    idx = nb.kneighbors(Z[y == 1], return_distance=False)[:, 1:]
    purity = float(y[idx].mean()); dtr = nb.kneighbors(Z)[0][:, 1:].mean(1); dva = nb.kneighbors(Zv, n_neighbors=5)[0].mean(1)
    q = np.percentile(dtr, [50, 90, 95])
    return dict(purity=purity, random=float((y.sum() - 1) / (n - 1)), train_5nn_median=float(q[0]), train_5nn_p95=float(q[2]),
                val_5nn_median=float(np.median(dva)), val_over_p95=int((dva > q[2]).sum()), val_over_p90=int((dva > q[1]).sum()), val_max=float(dva.max()),
                val_top5=[(nv[i][-3:], round(float(dva[i]), 2)) for i in np.argsort(dva)[::-1][:5]])


def tsne_best(A, perp):
    best = None
    for s in range(5):
        t = TSNE(2, perplexity=perp, init="random", random_state=s, learning_rate="auto"); Em = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em, s)
    return best


MARK = ["abn_002", "abn_014", "abn_015", "abn_017", "abn_022", "abn_024", "abn_029"]
S_all = np.concatenate([T.L1_S.values, Tv.L1_S.values])


def tsne_fig(Z, Zv, perp, title, fname):
    A = np.vstack([Z, Zv]); kl, Em, seed = tsne_best(A, perp); Et, Ev = Em[:n], Em[n:]
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.6))
    a = ax[0]; a.scatter(*Et.T, s=26, color=GRAY, alpha=.65, lw=0, marker="o", label="train 150 (라벨 숨김)"); a.scatter(*Ev.T, s=55, color=PUR, marker="^", edgecolor="white", lw=.5, label="val 40")
    a.set_title("(a) 무라벨: train · val 모양만", fontsize=10); a.legend(fontsize=8)
    a = ax[1]; a.scatter(*Et[nor].T, s=22, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[~nor].T, s=40, color=ABN, lw=0, label="train 고장"); a.scatter(*Ev.T, s=50, color=PUR, marker="^", edgecolor="white", lw=.5, label="val")
    for m in MARK:
        i = names.index(m); a.annotate(m[-3:], Et[i], xytext=(3, 3), textcoords="offset points", fontsize=8, color=ABN, weight="bold")
    for i in np.flatnonzero(~nor):
        if names[i] not in MARK: a.annotate(names[i][-3:], Et[i], xytext=(2, 2), textcoords="offset points", fontsize=6, color=ABN)
    a.set_title("(b) 라벨 색 (고장 번호, 굵게 = 002 014 015 017 022 024 029)", fontsize=10); a.legend(fontsize=8)
    a = ax[2]; sc = a.scatter(*np.vstack([Et, Ev]).T, c=np.clip(S_all, 0, 1), cmap="viridis", s=26, lw=0); fig.colorbar(sc, ax=a, shrink=.8, label="주기 잠금 S")
    a.scatter(*Et[~nor].T, s=60, facecolor="none", edgecolor=ABN, lw=.9); a.scatter(*Ev.T, s=60, facecolor="none", edgecolor=PUR, lw=.9, marker="^")
    a.set_title("(c) 박자 S 색 (테두리: 주황 고장 · 보라 val)", fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title} · perplexity {perp} · seed {seed} (5개 중 KL 최소 {kl:.3f}) · 같은 좌표 세 칸. 모양은 perplexity에 따라 바뀌므로 숫자(표)를 본다", x=.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(FIG, fname), dpi=85); plt.close(fig)
    return dict(kl=float(kl), seed=int(seed))


# 새 피처 = 최종 목록 (H16–H17은 정상 전체 참고값) + L1 S (final에 L1이 있으면 중복 안 넣음)
cols_new = final + ([c for c in L_COLS if c not in final])
Z_new, Zv_new = standardize(A_tr[cols_new].values, A_va[cols_new].values)
# ext150 (기존 캐시) — val은 같은 함수로 계산해 캐시, 열은 train에 맞춤
E = cached_table("ext", feature_table_ext, X); Ev_ = cached_table("ext_val", feature_table_ext, Xv).reindex(columns=E.columns, fill_value=0.0)
Z_ext, Zv_ext = standardize(E.values, Ev_.values)
R["knn"] = {"새 피처 (%d열)" % len(cols_new): knn_numbers(Z_new, Zv_new), "ext150 (%d열)" % E.shape[1]: knn_numbers(Z_ext, Zv_ext)}
R["knn_clip10"] = {"새 피처": knn_numbers(np.clip(Z_new, -10, 10), np.clip(Zv_new, -10, 10)), "ext150": knn_numbers(np.clip(Z_ext, -10, 10), np.clip(Zv_ext, -10, 10))}   # 보조: z를 ±10으로 자른 거리 (MAD 0에 가까운 열이 val 거리를 지배하는지 확인)
print(json.dumps(R["knn"], ensure_ascii=False), flush=True); print(json.dumps(R["knn_clip10"], ensure_ascii=False), flush=True)
R["tsne"] = {}
for perp in (15, 30):
    R["tsne"][f"new_{perp}"] = tsne_fig(Z_new, Zv_new, perp, f"새 피처 {len(cols_new)}열 (최종 목록 + L1 S, 강건 표준화)", f"tsne_새피처_perp{perp}.png")
R["tsne"]["ext_30"] = tsne_fig(Z_ext, Zv_ext, 30, "비교 기준 ext150 (같은 강건 표준화)", "tsne_ext150_perp30.png")
R["cols_new"] = cols_new
json.dump(R, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("완료")
