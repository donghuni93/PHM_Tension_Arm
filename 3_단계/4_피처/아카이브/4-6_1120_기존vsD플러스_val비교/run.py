# %% [markdown]
# 4-6. 기존 최종 피처(EXP-034 ET 입력 = ext150 + 울림 2열, 152열) vs D+ 81열 — 같은 틀에서 ET 입력만 바꿔 train + val을 한 그림으로 (11-20)
# 행 = 피처 집합, 열 = (1) 2D 평면 AE z × ET z + 로지스틱 경계 (2) 무라벨 t-SNE (3) ET 근접도 t-SNE
# - (1) AE는 두 행이 같다(스펙트럼 오토인코더). ET z만 피처 집합에 따라 다르다. train 점 = OOF z(그 파일을 학습에 안 쓴 모델), val = 전체 학습 모델로 변환
#       규칙(8.3 kHz · 유격)으로 고장인 파일은 평면에서 뺀다(결정.md 그림 규칙). val 규칙 파일은 제목에 적는다
# - (2) 피처 표준화(train으로 fit) 후 유클리드 거리. 라벨은 색칠에만
# - (3) ET(train 라벨로 학습) 잎 공유 비율 = 근접도. train 점은 라벨을 본 점이라 분리가 과장된다 → 제목에 OOF 이웃 순도를 같이 적는다
# - val을 보고 문턱·구성을 고치지 않는다. 이 그림은 설명·비교용
# 실행: conda run -n qe-phm python "3_단계/4_피처/4-6_1120_기존vsD플러스_val비교/run.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import src.final as fin
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.dplus import DPlus
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
LOG = open(os.path.join(HERE, "실행로그.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

# %% 데이터 · 표 (D+ 변환은 train으로만 fit, val은 변환만)
X, y, names = load_train(); n = len(y)
dp = DPlus().fit(X, cached_table("ext", feature_table_ext, X)); T = dp.T_
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val")); Dv, Tv = dp.transform(Xv)
fin.KEYS = fin.KEYS + ("DP",)                                    # take()가 D+ 표도 나눠 들게
T["DP"], Tv["DP"] = dp.D_, Dv
m = joblib.load(os.path.join(_d, "4_outputs", "final", "final_model.joblib"))
assert list(m.cols_) == list(dp.cols_), "최종 모델과 지금 피처 열이 다르다"

class CoreD(fin.Core):                                           # EXP-034와 같고 ET 입력만 D+ 81열
    def _E(self, T): return T["DP"]
SETUPS = {"기존 EXP-034 (ext150 + 울림2 = 152열)": m.core_, "D+ (81열)": CoreD().fit(T, y)}
THR = m.thr_

# %% 행마다 계산
def tsne(A, pre=False):
    best = None
    for s in range(5):
        t = TSNE(2, perplexity=15, init="random" if pre else "pca", random_state=s, learning_rate="auto",
                 metric="precomputed" if pre else "euclidean"); E = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, E)     # KL 최소 seed (라벨 안 씀)
    return best[1]
def prox(et, A, B):
    La, Lb = et.apply(A), et.apply(B)
    return np.mean([(La[:, t][:, None] == Lb[:, t][None, :]) for t in range(La.shape[1])], axis=0)
def mk_et(): return ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=-1)
def purity_unsup(F):                                             # 고장 파일의 이웃 5개(자기 제외) 중 고장 비율 평균. 거리에 라벨 안 씀
    nb = NearestNeighbors(n_neighbors=6).fit(F).kneighbors(F[y == 1], return_distance=False)[:, 1:]; return y[nb].mean()
def purity_et(F):                                                # in-sample(라벨 본 거리) vs OOF(폴드 밖 ET로 잰 거리)
    P = prox(mk_et().fit(F, y), F, F); np.fill_diagonal(P, -1); ins = y[np.argsort(-P[y == 1], 1)[:, :5]].mean()
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, y):
        Pt = prox(mk_et().fit(F[tr], y[tr]), F[te], F[tr]); nb = np.argsort(-Pt, 1)[:, :5]
        acc += list(y[tr][nb].mean(1)[y[te] == 1])
    return ins, float(np.mean(acc))

R = {}
for name, core in SETUPS.items():
    pt, pv = core.parts(T), core.parts(Tv)
    rule_t = (pt.rule83 | pt.knock).values; rule_v = (pv.rule83 | pv.knock).values
    Zt, Zv = core.oof_z_, pv[["ae_z", "et_z"]].values
    lr_oof = core.lr_.decision_function(Zt)
    Ft, Fv = core._E(T), core._E(Tv); sc = StandardScaler().fit(Ft); A = np.vstack([sc.transform(Ft), sc.transform(Fv)])
    et = core.experts_[1]; P = prox(et, np.vstack([Ft, Fv]), np.vstack([Ft, Fv])); Dm = 1 - P; np.fill_diagonal(Dm, 0)
    pu = purity_unsup(sc.transform(Ft)); pe_in, pe_oof = purity_et(Ft)
    R[name] = dict(Zt=Zt, Zv=Zv, rule_t=rule_t, rule_v=rule_v, lr=core.lr_, pv=pv, E_u=tsne(A), E_e=tsne(Dm, pre=True),
                   pu=pu, pe_in=pe_in, pe_oof=pe_oof, ncol=Ft.shape[1],
                   auc_et=roc_auc_score(y, Zt[:, 1]), ap_et=average_precision_score(y, Zt[:, 1]),
                   auc_lr=roc_auc_score(y, lr_oof), ap_lr=average_precision_score(y, lr_oof),
                   flag_v=(pv.score >= THR).values)
    r = R[name]
    say(f"[{name}] 열 {r['ncol']} | train OOF  ET z: AUC {r['auc_et']:.3f} · PR-AUC {r['ap_et']:.3f} | AE+ET 로지스틱: AUC {r['auc_lr']:.3f} · PR-AUC {r['ap_lr']:.3f}")
    say(f"   이웃 순도(고장의 5-NN 중 고장, 무작위 0.20): 무라벨 {pu:.2f} | ET 근접도 in-sample {pe_in:.2f} → OOF {pe_oof:.2f}")
    say(f"   val 고장 판정(점수 ≥ {THR}) {int(r['flag_v'].sum())}개: {' '.join(f[-3:] for f, k in zip(nv, r['flag_v']) if k)} | 그중 규칙 {int(rule_v.sum())}개: {' '.join(f[-3:] for f, k in zip(nv, rule_v) if k)}")
k0, k1 = list(R)
a, b = R[k0]["flag_v"], R[k1]["flag_v"]
say(f"val 판정 차이: 기존만 고장 {[f[-3:] for f, p, q in zip(nv, a, b) if p and not q]} | D+만 고장 {[f[-3:] for f, p, q in zip(nv, a, b) if q and not p]}")
from scipy.stats import spearmanr
say(f"val ET z 순위 상관(기존 vs D+) {spearmanr(R[k0]['Zv'][:, 1], R[k1]['Zv'][:, 1])[0]:.2f} | 로지스틱 점수 순위 상관 {spearmanr(R[k0]['pv'].lr, R[k1]['pv'].lr)[0]:.2f}")
pd.DataFrame({"file": nv, **{f"{k.split()[0]}_{c}": R[k]["pv"][c].round(3).values for k in R for c in ("ae_z", "et_z", "lr", "score")},
              **{f"{k.split()[0]}_판정": R[k]["flag_v"].astype(int) for k in R}}).to_csv(os.path.join(HERE, "val_두피처_점수.csv"), index=False, encoding="utf-8-sig")

# %% 그림 2행 × 3열
NOR, ABN, PUR, DARK = "#2a78d6", "#eb6834", "#7B1FA2", "#4A148C"
lo, hi = -3, 4.5; xx, yy = np.meshgrid(np.linspace(lo, hi, 300), np.linspace(lo, hi, 300)); G = np.c_[xx.ravel(), yy.ravel()]
fig, ax = plt.subplots(2, 3, figsize=(24, 15.5))
for row, (name, r) in enumerate(R.items()):
    # (1) 평면
    a = ax[row, 0]; Dg = r["lr"].decision_function(G).reshape(xx.shape)
    a.contourf(xx, yy, Dg, levels=[-1e9, THR, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    a.contour(xx, yy, Dg, levels=[THR], colors="k", linewidths=1.6); a.contour(xx, yy, Dg, levels=[0], colors="k", linewidths=.8, linestyles="--")
    Zt, Zv = np.clip(r["Zt"], lo, hi), np.clip(r["Zv"], lo, hi); kt, kv = ~r["rule_t"], ~r["rule_v"]
    a.scatter(*Zt[kt & (y == 0)].T, s=22, color=NOR, marker="s", lw=0, alpha=.8, label="train 정상 (OOF)")
    a.scatter(*Zt[kt & (y == 1)].T, s=30, color=ABN, lw=0, alpha=.9, label="train 고장 (OOF, 규칙 고장 제외)")
    a.scatter(*Zv[kv].T, s=75, color=PUR, edgecolor="white", lw=.7, zorder=5, label="val (규칙 고장 제외)")
    for i in np.flatnonzero(kv & r["flag_v"]): a.annotate(nv[i][-3:], Zv[i], fontsize=8.5, weight="bold", color=DARK, xytext=(4, 3), textcoords="offset points")
    for i in np.flatnonzero(kt & (y == 1) & (r["lr"].decision_function(r["Zt"]) < THR)): a.annotate(names[i].replace("abn_", "a"), Zt[i], fontsize=7.5, color=ABN, xytext=(3, -9), textcoords="offset points")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=8, loc="upper left")
    a.set_xlabel("AE z (스펙트럼 모양 이상, 두 행 같음)"); a.set_ylabel(f"ET z ({name.split(' (')[0]} 피처로 학습한 ET)")
    rv = " ".join(f[-3:] for f, k in zip(nv, r["rule_v"]) if k) or "없음"
    a.set_title(f"[{name}] AE z × ET z 평면 · 실선 = 임계값 {THR:.1f}, 점선 = 0\ntrain OOF: ET z PR-AUC {r['ap_et']:.3f} · 결합 PR-AUC {r['ap_lr']:.3f} | val 고장 판정 {int(r['flag_v'].sum())}개 (번호), 규칙으로 고장 val: {rv}", fontsize=9.5)
    # (2)(3) t-SNE
    for c, (E, tt) in enumerate(((r["E_u"], f"무라벨 t-SNE ({r['ncol']}열 표준화, 유클리드)\n고장의 이웃 5개 중 고장 비율 {r['pu']:.2f} (무작위 0.20)"),
                                (r["E_e"], f"ET 근접도 t-SNE (train 라벨로 학습)\n이웃 순도 in-sample {r['pe_in']:.2f} (과장) → 폴드 밖 OOF {r['pe_oof']:.2f}")), start=1):
        a = ax[row, c]; Et, Ev = E[:n], E[n:]
        a.scatter(*Et[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[y == 1].T, s=36, color=ABN, alpha=.9, lw=0, label="train 고장")
        a.scatter(*Ev.T, s=70, color=PUR, edgecolor="white", lw=.7, zorder=5, label="val 40")
        a.scatter(*Ev[r["rule_v"]].T, s=200, marker="D", facecolor="none", edgecolor="#1a9e5a", lw=1.4, zorder=6, label="val 규칙 고장")
        for i in np.flatnonzero(r["flag_v"]): a.annotate(nv[i][-3:], Ev[i], fontsize=8.5, weight="bold", color=DARK, xytext=(4, 3), textcoords="offset points")
        a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=8, loc="best"); a.set_title(f"[{name.split(' (')[0]}] {tt}", fontsize=10)
fig.suptitle(f"기존 피처 vs D+ — 같은 모델 틀(AE + ET → 로지스틱 + 규칙)에서 ET 입력만 바꿈 · 번호 = 그 구성이 고장 판정한 val (점수 ≥ {THR:.1f})", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(FIG, "기존vsD플러스_평면_tSNE.png"), dpi=100); plt.close(fig)
say("완료"); LOG.close()
