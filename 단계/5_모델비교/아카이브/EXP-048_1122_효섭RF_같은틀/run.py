# %% [markdown]
# EXP-048 (11-22) 신효섭 RF 배깅 + 8.3k(11-19 최종)를 우리 중첩 CV 틀에 넣어 J(EXP-044) · S(EXP-047)와 같은 조건으로 비교
# - 바깥 분할 = 우리 틀 그대로 (StratifiedGroupKFold 5겹, seed 0–4). 세 모델이 같은 파일로 학습 · 같은 파일로 채점 → seed별 짝 비교
# - 효섭 모델 계산은 src/rf_hyoseop.py(원본 최종모델.py 복사, 고치지 않음)
#   H  = 효섭 임계값 방식(바깥 학습 안 5겹 OOF, F2 최대 −0.02 구간 가운데) — 효섭 모델 그대로
#   H′ = 우리 임계값 방식(nested_cv: 안쪽 4겹 OOF, F2 최대 · 가장 낮은 값) — 임계값 방식까지 같은 틀
#   절제: H 8.3k 없음 · H + 공진 규칙(J와 같은 8.3k = 스펙트럼 OR 공진). 같은 확률에서 임계값만 다시 정함
# - J · S는 EXP-044 · EXP-047 코드 그대로 다시 실행(seed별 F2가 기록에 없어서). 이상치 3개는 세 모델 모두 학습 포함(비교 대상 J · S · H가 모두 포함 학습)
# - 8.3k 스펙트럼 문턱(0.75 dec · 150 Hz)은 H · J 모두 같은 사후 값. J는 공진 규칙이 더 붙는다
# 실행: conda run -n qe-phm python "단계/5_모델비교/아카이브/EXP-048_1122_효섭RF_같은틀/run.py"

# %% 설정
import os, sys, json, re, pickle, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp import CACHE
from src.exp2util import *
from src.models2 import TwoAE
from src.features2 import cached_hi_split, HI_SPLIT
from src import rf_hyoseop as HY
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
warnings.filterwarnings("ignore")
B5 = os.path.join(_d, "단계", "5_모델비교")
ex = Experiment(__file__, "EXP-048", "신효섭 RF 배깅 + 8.3k를 같은 틀에서 J · S와 비교", ["H1", "H2", "H3"],
                member="이동훈 (모델: 신효섭)", model="RF 배깅(피처 32개 중 10개 선택 × 10번) OR 8.3k 스펙트럼 규칙 — vs J · S",
                changes="효섭 11-19 최종 모델을 우리 중첩 CV 분할(바깥 5 × seed 5)에 넣어 J · S와 seed별 짝 비교",
                key_changes=["모델 = 신효섭 최종모델.py 그대로 (src/rf_hyoseop.py)", "평가 = 우리 바깥 분할 · seed 5개 · 지표 (J · S와 같은 파일로 학습 · 채점)",
                             "임계값 = 효섭 방식(H) · 우리 방식(H′) 둘 다", "J · S는 같은 seed로 다시 실행"])
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0; n = ns.n


def as_nested(P, yy):
    P = np.asarray(P, bool)
    return dict(f2=np.array([fbeta_score(yy, p, beta=2) for p in P]), recall=np.array([recall_score(yy, p) for p in P]), precision=np.array([precision_score(yy, p, zero_division=0) for p in P]),
                fpr=np.array([p[yy == 0].mean() for p in P]), miss_counts=((yy == 1) & ~P).sum(0), fp_counts=((yy == 0) & P).sum(0))


# %% 효섭 피처 표 (파일마다 독립, 라벨 안 씀). 캐시 outputs/cache/feat_hyoseop_{train,val}.pkl
def hy_tables(tag, X, nm):
    p = os.path.join(CACHE, f"feat_hyoseop_{tag}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    out = HY.tables(X, nm); pickle.dump(out, open(p, "wb")); return out


XH, AC, r83s, P83, W83 = hy_tables("train", ns.X, names)
XHv, ACv, r83sv, P83v, W83v = hy_tables("val", ns.Xv, ns.nv)
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv)
ex.note("점검_8.3k스펙트럼_일치", f"효섭 계산 vs 우리 rule83_flags: train {int((r83s == spec).sum())}/{n} · val {int((r83sv == specv).sum())}/{len(specv)} 일치")
R83 = {"spec": spec, "spec|res": spec | res, "none": np.zeros(n, bool)}

# %% H: 효섭 모델 그대로 (임계값 = 효섭 방식). 같은 확률로 8.3k 규칙만 바꾼 절제도 함께
jobs = [(s, k, tr, te) for s in range(5) for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g))]


def htask(s, k, tr, te):
    """HY.fit_predict와 같은 계산(같은 안쪽 분할 · 배깅 seed)인데 안쪽 OOF 확률과 바깥 확률을 돌려줘 규칙별 임계값을 다시 정한다"""
    cand = HY.candidates(XH); ytr = y[tr]; seed = 1000 * s + 10 * k
    pairs = list(StratifiedKFold(5, shuffle=True, random_state=s).split(tr, ytr))
    inner = np.zeros(len(tr))
    for j, (a, b) in enumerate(pairs):
        Xi = HY.prep(XH.iloc[tr].reset_index(drop=True), AC[tr], ytr, a)
        inner[b] = HY.bag(Xi.iloc[a], ytr[a], Xi.iloc[b], seed + j, cand)
    Xf = HY.prep(XH, AC, y, tr); log = []
    p = HY.bag(Xf.iloc[tr], ytr, Xf.iloc[te], seed + 9, cand, log)
    return s, tr, te, inner, p, log


with parallel_config(backend="loky", inner_max_num_threads=1):
    HR = Parallel(n_jobs=-1, verbose=2)(delayed(htask)(*j) for j in jobs)
HP = {}; PROB0 = np.zeros(n); THR = {}
for rule in ("spec", "none", "spec|res"):
    P = np.zeros((5, n), bool); th = []
    for s, tr, te, inner, p, _ in HR:
        r = R83[rule]; t = HY.threshold(y[tr], inner + 100 * r[tr]); th.append(t)
        P[s, te] = (p >= t) | r[te]
    HP[rule] = P; THR[rule] = th
for s, tr, te, inner, p, _ in HR:
    if s == 0: PROB0[te] = p
from collections import Counter
SEL = Counter(f for *_, log in HR for fe in log for f in fe)
ex.note("H_선택피처_횟수", ", ".join(f"{k} {v}" for k, v in SEL.most_common()))
ex.note("H_임계값", f"평균 {np.mean(THR['spec']):.3f} · 범위 {min(THR['spec']):.3f}–{max(THR['spec']):.3f} (바깥 25겹)")
Hres = ex.record("중첩: H | 효섭 RF 배깅 + 8.3k (효섭 임계값 방식)", as_nested(HP["spec"], y), names, main=True)
ex.record("중첩: H-8.3k없음 | H에서 8.3k 규칙 뺌 (RF 배깅 단독)", as_nested(HP["none"], y), names, mid=True)
ex.record("중첩: H+공진 | H의 8.3k를 J와 같은 스펙트럼 OR 공진으로", as_nested(HP["spec|res"], y), names, mid=True)


# %% H′: 임계값까지 우리 방식 (nested_cv: 안쪽 4겹 OOF에서 F2 최대). 입력 행 = [파일 번호, 8.3k 켜짐]
class HRF:
    def __init__(self, XH, AC, y): self.XH, self.AC, self.yall = XH, AC, y
    def get_params(self, deep=True): return dict(XH=self.XH, AC=self.AC, y=self.yall)
    def fit(self, F, yy):
        self.tr_ = np.asarray(F)[:, 0].astype(int); self.Xf_ = HY.prep(self.XH, self.AC, self.yall, self.tr_); return self
    def decision_function(self, F):
        F = np.asarray(F); te = F[:, 0].astype(int); cand = HY.candidates(self.XH)
        p = HY.bag(self.Xf_.iloc[self.tr_], self.yall[self.tr_], self.Xf_.iloc[te], int(self.tr_.sum()) % 9973, cand)
        return np.maximum(p, F[:, 1])


FH = np.column_stack([np.arange(n, dtype=float), spec.astype(float)])
Hq = nested_cv(FH, y, g, lambda: HRF(XH, AC, y), n_jobs=-1)
ex.record("중첩: H′ | H, 임계값 = 우리 방식 (안쪽 4겹 OOF F2 최대)", Hq, names, mid=True)

# %% J (EXP-044 코드 그대로) · S (EXP-047 코드 그대로) 다시 실행
idx = np.arange(n, dtype=float); r83J = (spec | res).astype(float)
M_rule = np.column_stack([ns.M, idx, r83J])
mkAE = lambda experts=(): (lambda: TwoAE("lin", experts, fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
Jr = nested_cv(M_rule, y, g, lambda: Add83(mkAE(("E2",))), n_jobs=-1)
ex.record("중첩: J | EXP-044 다시 실행 (같은 seed)", Jr, names, mid=True)

SP = cached_hi_split("train", ns.X); DROPC = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]
FX = [c for c in ns.FIXED if c not in DROPC] + HI_SPLIT; Hs = pd.concat([ns.T, SP], axis=1)
Fn = np.column_stack([Hs[FX].values, ns.T[["L1_S"]].values, ns.L, idx]); r83b = spec | res
mkS = lambda: TwoAE("lin", ("E2",), fixed=tuple(FX), nf=len(FX), df=ns.df, peak=tuple(ns.PEAK))


def stask(seed, tr, te):
    f = mkS().fit(Fn[tr], y[tr]).fires(Fn[te]); return seed, te, f["main"] | f["E2"] | r83b[te]


with parallel_config(backend="loky", inner_max_num_threads=1):
    SR = Parallel(n_jobs=-1)(delayed(stask)(s, tr, te) for s, _, tr, te in jobs)
PS = np.zeros((5, n), bool)
for s, te, p in SR: PS[s, te] = p
Sr = as_nested(PS, y)
ex.record("중첩: S | EXP-047 다시 실행 (같은 seed)", Sr, names, mid=True)

# %% seed별 짝 비교 (같은 바깥 분할)
Hn = as_nested(HP["spec"], y)
pair = lambda a, b: f"좋아짐 {int((a > b + 1e-9).sum())} · 같음 {int((abs(a - b) <= 1e-9).sum())} · 나빠짐 {int((a < b - 1e-9).sum())} (평균 차 {np.mean(a - b):+.3f})"
PAIR = {"H vs J": pair(Hn["f2"], Jr["f2"]), "H vs S": pair(Hn["f2"], Sr["f2"]), "H′ vs J": pair(Hq["f2"], Jr["f2"]), "S vs J": pair(Sr["f2"], Jr["f2"])}
for k, v in PAIR.items(): ex.note("짝비교_" + k, v); print(k, v)
SEEDF2 = {k: [round(float(v), 3) for v in r["f2"]] for k, r in (("H", Hn), ("H′", Hq), ("J", Jr), ("S", Sr))}
ex.note("seed별_F2", json.dumps(SEEDF2, ensure_ascii=False))

# %% 핵심 그림: seed별 F2 짝 비교 + 파일별 틀린 횟수 (H · H′ · J · S)
RUNS = [("H (효섭)", Hn, "#2ca02c"), ("H′ (효섭, 우리 임계값)", Hq, "#98df8a"), ("J (EXP-044)", Jr, "#d62728"), ("S (EXP-047)", Sr, "#ff9896")]
wrong = {nm: np.where(y == 1, r["miss_counts"], r["fp_counts"]) for nm, r, _ in RUNS}
files = [names[i] for i in range(n) if any(w[i] >= 2 for w in wrong.values())]
files = sorted(files, key=lambda f: (not f.startswith("abn"), f))
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw=dict(width_ratios=[1, 2.2]))
ax = axs[0]
for i, (nm, r, c) in enumerate(RUNS):
    ax.plot(range(5), r["f2"], "-o", color=c, lw=2, label=f"{nm}  {r['f2'].mean():.3f} ± {r['f2'].std():.3f}")
ax.set_xticks(range(5)); ax.set_xlabel("seed (바깥 분할, 네 모델 같은 파일)"); ax.set_ylabel("F2"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower left")
ax.set_title("seed별 F2 — 같은 분할 짝 비교", fontsize=10.5)
ax = axs[1]; W = np.array([[wrong[nm][names.index(f)] for f in files] for nm, _, _ in RUNS])
im = ax.imshow(W, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        if W[i, j]: ax.text(j, i, int(W[i, j]), ha="center", va="center", fontsize=9, color="white" if W[i, j] >= 3 else "black")
ax.set_yticks(range(len(RUNS))); ax.set_yticklabels([nm for nm, _, _ in RUNS]); ax.set_xticks(range(len(files)))
ax.set_xticklabels([short(f) for f in files], rotation=60, fontsize=8.5)
for t, f in zip(ax.get_xticklabels(), files): t.set_color(ABN if f.startswith("abn") else NOR)
ax.set_title("파일별 틀린 횟수 (seed 5번 중, 어느 모델이든 2회 이상인 파일) — 주황 글자 = 고장(놓침), 파랑 = 정상(오경보)", fontsize=10.5)
fig.tight_layout()
ex.save_fig(fig, "짝비교", "왼쪽 = seed별 F2(같은 바깥 분할), 오른쪽 = 파일별 틀린 횟수. 고장은 놓친 횟수, 정상은 오경보 횟수", key=True)

# %% 오류 분석 표 (H 기준, 가로) + J · S 판정 횟수
cnt = lambda r: np.array([(5 - r["miss_counts"][i]) if y[i] else r["fp_counts"][i] for i in range(n)])
cH, cJ, cS = cnt(Hn), cnt(Jr), cnt(Sr)
efiles = [names[i] for i in range(n) if any(w[i] >= 1 for w in wrong.values()) and (wrong["H (효섭)"][i] >= 1 or wrong["J (EXP-044)"][i] >= 2 or wrong["S (EXP-047)"][i] >= 2)]
efiles = sorted(efiles, key=lambda f: (not f.startswith("abn"), f))
fx = lambda arr: {f: f"고장 {int(arr[names.index(f)])}" for f in efiles}
tab = error_md_table(names, y, cH, efiles, extra={"H 확률 (seed 0)": {f: f"{PROB0[names.index(f)]:.2f}" for f in efiles},
                                                  "J 판단 (seed 5번 중)": fx(cJ), "S 판단 (seed 5번 중)": fx(cS)})
ex.save_md("오류분석_표", f"# EXP-048 오류 분석 (중첩 CV seed 5개, 같은 바깥 분할)\n\n우리 판단 = 정답 라벨, 모델 판단 = H(효섭 RF 배깅 + 8.3k). 아래 두 줄은 같은 분할에서 J · S가 고장으로 판정한 횟수.\n파일 = H가 1번 이상 틀렸거나 J · S가 2번 이상 틀린 파일.\n\n{tab}\n")

# %% val 판정 (train 150 전체 학습, 제출 파일 아님): 효섭 run_predict와 같은 절차. J · S는 각 실험 val_판정.md에서 읽음
X2 = pd.concat([XH, XHv], ignore_index=True); AC2 = np.vstack([AC, ACv]); r2 = np.r_[spec, specv]
yall = np.r_[y, np.zeros(len(ns.nv), int)]; tr, te = np.arange(n), np.arange(n, n + len(ns.nv))
pv, thrv, predv = HY.fit_predict(X2, AC2, yall, r2, tr, list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y)), te, 777)
def read_val(sub):
    t = open(os.path.join(find_exp(sub), "val_판정.md"), encoding="utf-8").read()
    return {m.group(1): m.group(2) for m in re.finditer(r"^\| (\d{3}) \| (고장|정상) \|", t, re.M)}
vJ, vS = read_val("EXP-044_1122_AE추가"), read_val("EXP-047_1122_8-12k쪼개기")
etJ = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\| (\d{3}) \| (?:고장|정상) \| ([-\d.]+) \|", open(os.path.join(find_exp("EXP-044"), "val_판정.md"), encoding="utf-8").read(), re.M)}
sub = pd.read_csv(os.path.join(_d, "참고자료", "팀원 공유 정보", "신효섭", "1119_최종", "6_최종·제출", "제출", "submission_val_9조_상세.csv"))
subd = {r.file[-3:]: int(r.label) for r in sub.itertuples()}
k3 = [s[-3:] for s in ns.nv]
V = pd.DataFrame(dict(파일=k3, H=np.where(predv, "고장", "정상"), H_확률=np.round(pv, 3), H_8_3k=np.where(specv, "켜짐", "—"),
                      효섭_제출=["고장" if subd.get(k) else "정상" for k in k3], J=[vJ[k] for k in k3], S=[vS[k] for k in k3], J_ET=[etJ[k] for k in k3]))
V["일치"] = np.where((V.H == V.J) & (V.J == V.S), "셋 다 같음", np.where(V.H == V.J, "S만 다름", np.where(V.J == V.S, "H만 다름", "J만 다름")))
ex.note("val_재현", f"효섭 제출 csv와 판정 일치 {int((V.H == V.효섭_제출).sum())}/40 · 임계값 {thrv:.3f}")
fig, ax = plt.subplots(figsize=(11, 8))
cmap = {"셋 다 같음": "#999999", "H만 다름": "#2ca02c", "J만 다름": "#d62728", "S만 다름": "#ff9896"}
for k_, c in cmap.items():
    m = (V.일치 == k_).values
    for hit, mk_ in ((V.H == "고장").values, "^"), ((V.H != "고장").values, "o"):
        mm_ = m & hit
        if mm_.any(): ax.scatter(V.J_ET[mm_], V.H_확률[mm_], s=110, color=c, marker=mk_, edgecolor="black", lw=.6, label=f"{k_} · H {'고장' if mk_ == '^' else '정상'} ({int(mm_.sum())})")
for i, r in V.iterrows():
    if r.일치 != "셋 다 같음" or r.H == "고장": ax.annotate(r.파일, (r.J_ET, r.H_확률), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
ax.axhline(thrv, color="#2ca02c", ls="--", lw=1.5, label=f"H 임계값 {thrv:.3f}"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_xlabel("J의 ET 점수 (EXP-044 val_판정.md, J 판정은 ET · 박자 · AE 결합 OR 규칙)"); ax.set_ylabel("H의 RF 배깅 고장 확률")
ax.set_title(f"val 40개 — H 고장 {int((V.H == '고장').sum())} · J {int((V.J == '고장').sum())} · S {int((V.S == '고장').sum())} · 셋 다 같은 파일 {int((V.일치 == '셋 다 같음').sum())}", fontsize=10.5)
fig.tight_layout()
ex.save_fig(fig, "val_판정", "val 40개: 가로 = J의 ET 점수, 세로 = H 확률. 색 = 세 모델 판정의 일치, ▲ = H 고장 판정")
ex.save_md("val_판정", f"# EXP-048 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\nH 고장 {int((V.H == '고장').sum())}개: {' '.join(V.파일[V.H == '고장'])}\n"
           f"J {int((V.J == '고장').sum())}개 · S {int((V.S == '고장').sum())}개 · 셋 다 고장: {' '.join(V.파일[(V.H == '고장') & (V.J == '고장') & (V.S == '고장')])}\n\n![val](figs/val_판정.png)\n\n{md_table(V)}\n")

# %% t-SNE 기본 (4-10 방식, 핵심 그림). 빨간 원 = H 오경보 (seed 0)
fpH = np.flatnonzero(nor & HP["spec"][0])
fig, ti = tsne_default(ns, mark=fpH, mark_label="H 오경보 (seed 0)", title="EXP-048 H(효섭 RF)")
ex.save_fig(fig, "tsne_라벨", f"기본 t-SNE(4-10): 왼쪽 라벨 없음, 오른쪽 라벨 색. 빨간 원 = H 오경보(seed 0). 폴드 밖 5-NN AUC {ti['auc']:.3f}", key=True)
print("done")
