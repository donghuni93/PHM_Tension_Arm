# %% [markdown]
# EXP-014 두 전문가 결합 — 충격은 ET, 스펙트럼 모양은 AE, 경계는 2차원 로지스틱 하나
# - 문제 (EXP-007 ET · EXP-012 AE, seed 5번 중 3번 이상)
#   · ET 놓침 002 022 029 (조용한 고장, 스펙트럼 모양만 다름) / 오경보 008 025 027 050 054 055 073 086 095 116
#   · AE 놓침 005 014 017 022 (충격형, 전체 스펙트럼은 정상 같음) / 오경보 025 039 055 059 062 106 116
#   · 둘 다 놓침 022, 둘 다 오경보 025 055 116. 나머지는 서로 반대 → 한쪽만 이상한 파일은 다른 쪽이 바로잡을 수 있다
# - 가설: H1(충격 → ET), H2·H3(공진·모양 변화 → AE). 고장 유형마다 드러나는 축이 달라 축 하나로는 경계가 흐리다
# - 방법 (모두 폴드 안)
#   1) 학습 파일 안 4-fold OOF로 ET 점수(logit 확률)·AE 점수(log 복원 오차)
#   2) 각 점수를 학습 정상 OOF 분포로 z = (s − 중앙값) / (90% − 중앙값)  → "정상 90%선까지 거리가 1"
#   3) 결합: 로지스틱(L2, C=1 고정, 균형 가중) — 파라미터 3개라 과적합 여지가 작다. 튜닝 안 함
#   4) 전문가는 학습 파일 전체로 다시 학습해 테스트 점수 → 같은 z → 결합
# - 평가: 탐색 비교라 quick_cv(5-fold × seed 3, PR-AUC). 중첩 CV는 최종 후보가 정해진 뒤에만
# - 주 구성은 실행 전에 '로지스틱 결합'으로 정함. 평균(학습 0개)·최댓값(OR)·단독은 비교용
# - 탐색: + 잔향 비(2-6, H5). 그림을 보고 정의한 값이라 주 구성 후보가 아니다
# 실행: conda run -n qe-phm python 단계/5_모델비교/1_첫베이스모델/EXP-014_1114_ET·AE결합/run.py

# %% 설정
import os, sys, warnings
sys.path.insert(0, next(_p for _p in (os.path.abspath(os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."), *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src"))))
from src.exp import *
import matplotlib.pyplot as plt
from joblib import Memory
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from src.evaluate import quick_cv

warnings.filterwarnings("ignore", category=ConvergenceWarning)
FS = 25000
NOR, ABN = "#2a78d6", "#eb6834"

ex = Experiment(__file__, "EXP-014", "두 전문가 결합 — ET(충격) + 스펙트럼 AE(모양) → 2차원 로지스틱 경계", ["H1", "H2", "H3"],
                member="이동훈", model="ET(ext150) + 스펙트럼 AE → 정상 기준 z → L2 로지스틱",
                changes="ET 점수와 AE 복원 오차를 폴드 안 정상 OOF 기준 z로 맞춰 파라미터 3개 로지스틱으로 결합", focus=())
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X).values
g = np.arange(len(y))

ET_MISS = ["abn_002", "abn_022", "abn_029"]
AE_MISS = ["abn_005", "abn_014", "abn_017", "abn_022"]
ET_FP = ["nor_008", "nor_025", "nor_027", "nor_050", "nor_054", "nor_055", "nor_073", "nor_086", "nor_095", "nor_116"]
AE_FP = ["nor_025", "nor_039", "nor_055", "nor_059", "nor_062", "nor_106", "nor_116"]
HARD = sorted(set(ET_MISS + AE_MISS)) + sorted(set(ET_FP + AE_FP))


# %% AE 입력 (EXP-012와 같음: 200 Hz 고역통과 → 1초 구간 8개 → log Welch PSD). 라벨 안 씀
def segment_logpsd(X, seg=FS, nseg=8, nperseg=2048):
    Xc = X - X.mean(1, keepdims=True)
    H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
    starts = np.linspace(0, X.shape[1] - seg, nseg).astype(int)
    S = np.stack([H[:, s:s + seg] for s in starts], 1)
    f, P = welch(S, fs=FS, nperseg=nperseg, axis=-1)
    return f, np.log10(P + 1e-20)


p = os.path.join(CACHE, "seg_logpsd_hp200_8x1s.npz")
if os.path.exists(p):
    LP = np.load(p)["LP"]
else:
    _, LP = segment_logpsd(X); np.savez(p, f=_, LP=LP)

# 잔향 비 (2-6, 탐색용): 가장 큰 충격 뒤 5–20 ms 포락선 중앙값 / 최고값. 파일마다 라벨 없이 계산된 값
dur = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "아카이브", "개관", "2-6_1114_충격지속시간", "figs", "duration_per_file.csv"))
RV = np.log10(dur.set_index("name").loc[names, "big_rv"].values + 1e-4)
print(f"잔향 비 결측 {np.isnan(RV).sum()}개 → 전체 중앙값으로 채움 (라벨 안 씀)", flush=True)
RV = np.where(np.isnan(RV), np.nanmedian(RV), RV)


# %% 전문가 점수 (학습 파일 tr로 학습 → te 점수). 같은 분할은 디스크 캐시로 재사용 (구성끼리 공유)
mem = Memory(os.path.join(CACHE, "joblib_exp014"), verbose=0)


@mem.cache
def ae_scores(tr, te, seed=0):
    """정상만 학습 (EXP-012 구조 1025-128-16-128-1025, 60 epoch). 점수 = log10 평균 복원 제곱오차"""
    tr, te = np.array(tr), np.array(te)
    S = LP[tr[y[tr] == 0]].reshape(-1, LP.shape[-1])
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Zs = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed).fit(Zs, Zs)
    Z = (LP[te] - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return np.log10(((R - Z) ** 2).mean((1, 2)))


@mem.cache
def et_scores(tr, te, seed=0):
    """ext150 ExtraTrees (EXP-007 기준과 같은 설정). 점수 = logit(고장 확률)"""
    tr, te = np.array(tr), np.array(te)
    pr = make_et(seed=seed)().fit(E[tr], y[tr]).predict_proba(E[te])[:, 1]
    pr = np.clip(pr, 1 / 600, 1 - 1 / 600)
    return np.log(pr / (1 - pr))


class TwoExpert(BaseEstimator):
    """F 1열 = 파일 번호. combine: 'et' 'ae' 'mean' 'max' 'lr'. rv=True면 잔향 비를 3번째 입력으로 (lr만)"""
    def __init__(self, combine="lr", rv=False, k=4, seed=0):
        self.combine, self.rv, self.k, self.seed = combine, rv, k, seed

    def _raw(self, tr, te):
        cols = [ae_scores(tr, te), et_scores(tr, te)]
        if self.rv:
            cols.append(RV[list(te)])
        return np.column_stack(cols)

    def _z(self, S):
        return (S - self.med_) / self.sc_

    def fit(self, F, yy):
        idx, yy = np.asarray(F)[:, 0].astype(int), np.asarray(yy)
        self.tr_ = tuple(idx.tolist())
        oof = None
        for a, b in StratifiedKFold(self.k, shuffle=True, random_state=self.seed).split(idx, yy):
            s = self._raw(tuple(idx[a].tolist()), tuple(idx[b].tolist()))
            oof = np.empty((len(idx), s.shape[1])) if oof is None else oof
            oof[b] = s
        n = oof[yy == 0]
        self.med_ = np.median(n, 0)
        self.sc_ = np.abs(np.percentile(n, 90, 0) - self.med_) + 1e-6
        self.oof_z_ = self._z(oof)
        if self.combine == "lr":
            self.lr_ = LogisticRegression(C=1.0, class_weight="balanced").fit(self.oof_z_, yy)
        return self

    def zscores(self, F):
        return self._z(self._raw(self.tr_, tuple(np.asarray(F)[:, 0].astype(int).tolist())))

    def decision_function(self, F):
        Z = self.zscores(F)
        return {"ae": lambda: Z[:, 0], "et": lambda: Z[:, 1], "mean": lambda: Z[:, :2].mean(1),
                "max": lambda: Z[:, :2].max(1), "lr": lambda: self.lr_.decision_function(Z)}[self.combine]()


# %% 탐색 비교 = quick_cv (5-fold × seed 3, PR-AUC, 임계값 없음). 중첩 CV는 최종 후보에서만 (결정 "평가")
# 주 구성은 실행 전에 '로지스틱 결합'으로 정함
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
SEEDS = range(3)

# 캐시 채우기: quick_cv가 부를 (학습, 채점) 분할을 미리 병렬 계산. TwoExpert.fit과 같은 분할을 재현
jobs = []
for s in SEEDS:
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        jobs.append((tuple(tr.tolist()), tuple(te.tolist())))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(tr, y[tr]):
            jobs.append((tuple(tr[a].tolist()), tuple(tr[b].tolist())))
Parallel(n_jobs=-1)(delayed(fn)(a, b) for a, b in jobs for fn in (ae_scores, et_scores))
print(f"캐시 완료: 분할 {len(jobs)}개 × 전문가 2", flush=True)

CONFIGS = [("로지스틱 결합 ET+AE (주)", dict(combine="lr")),
           ("평균 결합 (학습 0개)", dict(combine="mean")),
           ("최댓값 결합 = OR", dict(combine="max")),
           ("ET 단독 (같은 틀)", dict(combine="et")),
           ("AE 단독 (같은 틀)", dict(combine="ae")),
           ("탐색: 로지스틱 + 잔향 비 (H5, 사후 정의)", dict(combine="lr", rv=True))]
R, rows = {}, []
nor = y == 0
for name, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda kw=kw: TwoExpert(**kw), seeds=SEEDS)
    # 파일별 '정상 중 순위' = 정상 점수 중 그 파일보다 낮은 비율 (seed 평균). 고장은 낮으면 놓침, 정상은 높으면 오경보
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:40s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · "
                f"ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음)", "\n".join(rows) +
        "\n기준 비교(같은 방식이 아님): EXP-007 ET 중첩 F2 0.827, EXP-012 AE 0.832")

# 어려운 파일 표 (정상 중 순위 백분위)
tab = ["파일          " + " | ".join(f"{c[0][:10]:10s}" for c in CONFIGS)]
for nm in HARD:
    i = names.index(nm)
    tab.append(f"{nm:12s}  " + " | ".join(f"{R[c[0]]['pct'][i]:10.2f}" for c in CONFIGS))
ex.note("어려운 파일의 정상 중 순위 (고장은 1에 가까울수록, 정상은 0에 가까울수록 좋음)", "\n".join(tab))

main_name = CONFIGS[0][0]
pm = R[main_name]["pct"]
missed = [str(names[i]) for i in np.flatnonzero((y == 1) & (pm < 0.90))]   # 정상 90% 선 아래 고장
fps = [str(names[i]) for i in np.flatnonzero((y == 0) & (pm > 0.95))]      # 정상 중 상위 5%
ex.note("주 구성: 정상 90% 선 아래 고장 · 정상 상위 5% 정상",
        f"고장({len(missed)}개): {' '.join(missed)}\n정상({len(fps)}개): {' '.join(fps)}")


# %% 그림용: quick_cv seed 0과 같은 분할로 150개 파일의 OOF z 좌표 (캐시 재사용)
Zall = np.empty((len(y), 3)); W = []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = TwoExpert("lr", rv=True).fit(g[tr][:, None], y[tr])
    Zall[te] = m.zscores(g[te][:, None])
    W.append(TwoExpert("lr").fit(g[tr][:, None], y[tr]).lr_)
coef = np.mean([w.coef_[0] for w in W], 0); icpt = np.mean([w.intercept_[0] for w in W])
lines = []
for nm, s in (("AE", Zall[:, 0]), ("ET", Zall[:, 1]), ("평균", Zall[:, :2].mean(1)), ("OR", Zall[:, :2].max(1)),
              ("로지스틱(폴드 평균 계수)", Zall[:, :2] @ coef + icpt)):
    lines.append(f"{nm:24s} PR-AUC {average_precision_score(y, s):.3f} · ROC-AUC {roc_auc_score(y, s):.3f}")
ex.note("OOF 한 번(seed 0) 임계값 없는 지표", "\n".join(lines) +
        f"\n로지스틱 계수 (폴드 평균): AE {coef[0]:+.3f} · ET {coef[1]:+.3f} · 절편 {icpt:+.3f}")


# %% 그림 1: 2차원 경계 — ET 축만, AE 축만으로는 겹치는 파일이 2차원에서는 갈리는가
fig, ax = plt.subplots(figsize=(9.5, 8))
LO, HI = -3, 4
zc = np.clip(Zall[:, :2], LO, HI)
ax.scatter(*zc[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="정상")
ax.scatter(*zc[y == 1].T, s=34, color=ABN, lw=0, label="고장")
ax.axvline(1, color=NOR, lw=.7, ls=":"); ax.axhline(1, color=NOR, lw=.7, ls=":")
xs = np.linspace(LO, HI, 50)
ax.plot(xs, -(coef[0] * xs + icpt) / coef[1], color="k", lw=1.2, label="로지스틱 경계 (확률 0.5, 폴드 평균)")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), zc[i], xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=ABN if y[i] else NOR)
ax.plot(xs, 2 * np.median(Zall[y == 0, :2].mean(1)) + 1.2 - xs, color="k", lw=.8, ls="--",
        label="평균 결합 방향 (기울기 −1, 위치는 예시)")
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
ax.set_xlabel("AE z (스펙트럼 모양이 정상과 다른 정도, 1 = 정상 90%선)")
ax.set_ylabel("ET z (충격 피처가 고장 같은 정도, 1 = 정상 90%선)")
ax.set_title(f"파일별 OOF 좌표 (5-fold, seed 0). 점선 = 각 전문가의 정상 90%선. 범위 밖은 끝({LO}, {HI})에 붙임", fontsize=10)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout()
ex.save_fig(fig, "boundary_2d", "ET만 놓친 고장(A002·A029)은 오른쪽, AE만 놓친 고장(A005·A014·A017)은 위쪽에 있는가. "
            "한쪽만 오경보인 정상은 경계 아래로 내려오는가. 라벨 = 이전 실험에서 틀린 파일")


# %% 그림 2: 어려운 파일의 '나쁜 정도' — 고장은 1 − 순위(놓칠 위험), 정상은 순위(오경보 위험)
cfg = [c[0] for c in CONFIGS]
M = np.array([[1 - R[c]["pct"][names.index(nm)] if nm.startswith("abn") else R[c]["pct"][names.index(nm)]
               for c in cfg] for nm in HARD])
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD))); ax.set_yticklabels([("놓침 " if n.startswith("abn") else "오경보 ") + n for n in HARD], fontsize=8)
ax.set_xticks(range(len(cfg))); ax.set_xticklabels([c.replace(" (", "\n(") for c in cfg], fontsize=8)
ax.set_title("이전 실험에서 틀린 파일의 위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율",
             fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "ET 문제(002·029 놓침, 027·050·054·073·095 오경보)와 AE 문제(005·014·017 놓침, "
            "039·059·062·106 오경보)가 결합에서 옅어지는가. 022·025·055·116(둘 다 틀림)은 남는가")


# %% 그림 3: t-SNE (매 실험 기본) — 모델 출력 끝단 (OOF, quick_cv seed 0과 같은 분할)
#   AE 출력 끝 = bin별 복원 오차 (EXP-012의 '출력 끝'), ET 출력 끝 = 나무별 고장 확률, 결합 = 두 블록을 같은 비중으로
@mem.cache
def ae_binerr(tr, te, seed=0):
    """ae_scores와 같은 AE (같은 seed·데이터 → 같은 모델). bin별 log10 복원 제곱오차 (구간 평균) → (te, 1025)"""
    tr, te = np.array(tr), np.array(te)
    S = LP[tr[y[tr] == 0]].reshape(-1, LP.shape[-1])
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Zs = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed).fit(Zs, Zs)
    Z = (LP[te] - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return np.log10(((R - Z) ** 2).mean(1) + 1e-12)


@mem.cache
def et_trees(tr, te, seed=0):
    """et_scores와 같은 ET. 나무 300그루 각각의 고장 확률 → (te, 300)"""
    tr, te = np.array(tr), np.array(te)
    m = make_et(seed=seed)().fit(E[tr], y[tr])
    return np.column_stack([t.predict_proba(E[te])[:, 1] for t in m.estimators_])


AEo, ETo = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    AEo[te], ETo[te] = ae_binerr(a, b), et_trees(a, b)
std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
AEs, ETs = std(AEo), std(ETo)
# 결합: 블록마다 열 수의 제곱근으로 나눠 두 블록의 전체 분산을 같게 (1025열인 AE가 300열인 ET를 덮지 않게)
COMB = np.hstack([AEs / np.sqrt(AEs.shape[1]), ETs / np.sqrt(ETs.shape[1])])
fig, _ = viz.tsne({"AE 출력 끝: bin별 복원 오차 (1025, OOF)": AEs,
                   "ET 출력 끝: 나무별 고장 확률 (300, OOF)": ETs,
                   "결합: AE 출력 끝 + ET 출력 끝 (블록 같은 비중)": COMB},
                  y, names, mark=HARD, score=Zall[:, :2].mean(1), scale=False)
ex.save_fig(fig, "tsne", "모델이 실제로 판단하는 출력 끝단에서 고장이 모이는가. AE·ET 각각보다 결합 공간에서 더 모이는가. "
            "2행 색 = 평균 결합 점수(OOF). 번호 = 이전 실험에서 틀린 파일")

# 비교용: 입력 공간 (모델을 거치기 전)
fig, _ = viz.tsne({"ET 입력: ext150 피처": E,
                   "AE 입력: 200 Hz 고역통과 log PSD (구간 평균, 1025)": LP.mean(1)},
                  y, names, mark=HARD)
ex.save_fig(fig, "tsne_input", "비교용: 모델을 거치기 전 입력 공간. 출력 끝(tsne)보다 고장이 덜 모이는가")


# %% 그림 4: 놓친 고장 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"주 구성이 놓친 고장 {' '.join(missed)}의 PSD vs 정상 5–95%"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)


# %% 최종 후보 확인: 중첩 CV F2 (바깥 5 × 안쪽 4, seed 5). quick_cv로 남긴 후보 2개만 (결정 "평가")
# 주 구성(노션 카드 F2)은 실행 전에 정한 로지스틱 결합. 평균 결합은 학습 0개 대안
for name, kw, main in (("최종 후보: 로지스틱 결합 ET+AE (주)", dict(combine="lr"), True),
                       ("최종 후보: 평균 결합 (학습 0개)", dict(combine="mean"), False)):
    r = nested_cv(g[:, None], y, g, lambda kw=kw: TwoExpert(**kw), n_jobs=-1)
    ex.record(name, r, names, main=main)
print("nested done", flush=True)
