# %% [markdown]
# EXP-042 (2차 설계 3단계, 11-21) 2단계 승자 위에 정상 기준 전문가를 OR로 붙인다 (구성은 실행 전 고정, 결과 보고 안 바꿈)
# - 승자 W = EXP-041의 선택 규칙 결과(result.json notes.선택.winner)
# - D1 = W OR E1 (H16 가장 솟은 좁은 봉우리 높이 > 학습 정상 H16의 99 %, 겨냥 002형)
# - D2 = D1 OR E2 (H18 HFL > 학습 정상 99 % AND H1 충격 수 ≥ 1, 겨냥 017형)
# - 문턱은 모두 fit 행(학습 폴드) 정상의 99 % 분위, 라벨로 고르지 않음. W의 결합 문턱은 fit 행의 안쪽 OOF 점수 F2 최대 (src/models2.py)
# - 선택: D는 W보다 sd를 넘게 좋을 때만, D1·D2가 sd 안이면 D1
# - 1차 최종 모델 = 선택 규칙으로 남은 구성. train 150 전체로 학습해 val 40 채점 (제출 파일 없음)
# 실행: conda run -n qe-phm python "단계/5_모델실험/3_2차설계/EXP-042_2차설계_3단계전문가/자료/run.py"

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.expA import A as ACls
from src.evaluate import best_f2_threshold
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
EXP041 = find_exp("EXP-041")
ex = Experiment(__file__, "EXP-042", "2차 설계 3단계 — 2단계 승자 + 정상 기준 전문가 E1 · E2 (OR)", ["H2", "H3", "H1"],
                member="이동훈", model="2단계 승자(C-lin) OR E1(솟은 봉우리 H16) OR E2(하중 대비 덜컥 H18 & 충격 ≥ 1)",
                changes="전문가 문턱 = 학습 폴드 정상 99 % 분위(고정, 라벨로 안 고름). D1 = W + E1, D2 = D1 + E2")
ns = NS(); names, y, g = ns.names, ns.y, ns.g
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


r41 = json.load(open(os.path.join(EXP041, "result.json"), encoding="utf-8"))
W = r41["notes"]["선택"]["winner"]; log("2단계 승자 W =", W)
Wkey = {"max": "C-max | (ET ≥ t_H) OR (박자 z ≥ q99)", "lin": "C-lin | EXP-040 C 그대로", "svm": "C-svm | RBF SVM [ET OOF logit, 박자 z]"}[W]
W_row = r41["configs"]["중첩: " + Wkey]
A_row = r41["configs"]["중첩: A | EXP-034 b (EXP-040 값)"]; B_row = r41["configs"]["중첩: B | ET 하나 (EXP-040 값)"]
ex.res["configs"]["중첩: A | EXP-034 b (EXP-040 값)"] = A_row; ex.res["configs"]["중첩: B | ET 하나 (EXP-040 값)"] = B_row
ex.res["configs"]["중첩: W | 2단계 승자 C-" + W + " (EXP-041 값)"] = W_row; ex._write()

# %% 중첩 CV
RAW = {}
CFG = {"D1 | W + E1 (솟은 봉우리)": ns.mk(W, ("E1",)), "D2 | D1 + E2 (하중 대비 덜컥)": ns.mk(W, ("E1", "E2"))}
for nm, mk_ in CFG.items():
    RAW[nm] = nested_cv(ns.M, y, g, mk_, n_jobs=-1); c = ex.record("중첩: " + nm, RAW[nm], names)
    log(f"{nm:34s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · F2(10/40) {c['f2_p10']:.3f}/{c['f2_p40']:.3f} · 미탐 {c['miss_counts']} · 오경보 {c['fp_counts']}")
D1 = ex.res["configs"]["중첩: D1 | W + E1 (솟은 봉우리)"]; D2 = ex.res["configs"]["중첩: D2 | D1 + E2 (하중 대비 덜컥)"]

# %% 선택 규칙
RAW["W"] = nested_cv(ns.M, y, g, ns.mk(W), n_jobs=-1)      # 선택 규칙은 반올림 전 값으로 (경계 차이가 반올림에 흔들리지 않게). EXP-041 값과 같아야 함
log("W 재계산(반올림 전)", RAW["W"]["f2"].mean(), RAW["W"]["f2"].std(), "· EXP-041 기록", W_row["f2"], W_row["f2_sd"])
sdW = float(RAW["W"]["f2"].std())
fm = lambda k: float(RAW[k]["f2"].mean())
d1_ok = fm("D1 | W + E1 (솟은 봉우리)") - fm("W") > sdW                           # D는 W보다 sd를 넘게 좋을 때만
d2_ok = fm("D2 | D1 + E2 (하중 대비 덜컥)") - fm("W") > sdW
d2_over_d1 = fm("D2 | D1 + E2 (하중 대비 덜컥)") - fm("D1 | W + E1 (솟은 봉우리)") > max(float(RAW["D1 | W + E1 (솟은 봉우리)"]["f2"].std()), float(RAW["D2 | D1 + E2 (하중 대비 덜컥)"]["f2"].std()))
final = "W" if not (d1_ok or d2_ok) else ("D2" if (d2_ok and d2_over_d1) or (d2_ok and not d1_ok) else "D1")
log(f"D1 − W = {fm('D1 | W + E1 (솟은 봉우리)') - fm('W'):+.4f} · D2 − W = {fm('D2 | D1 + E2 (하중 대비 덜컥)') - fm('W'):+.4f} (W sd {sdW:.4f}, 반올림 전) → D1 채택 {d1_ok} · D2 우위 {d2_ok} · D2 − D1 우위 {d2_over_d1} · 최종 = {final}")
fkey = {"W": "W | 2단계 승자 C-" + W + " (EXP-041 값)", "D1": "D1 | W + E1 (솟은 봉우리)", "D2": "D2 | D1 + E2 (하중 대비 덜컥)"}[final]
fexp = {"W": (), "D1": ("E1",), "D2": ("E1", "E2")}[final]
ex.res["main"] = "중첩: " + fkey; ex._write()
ex.note("선택", dict(W=W, final=final, D1_minus_W=round(D1["f2"] - W_row["f2"], 4), D2_minus_D1=round(D2["f2"] - D1["f2"], 4), W_sd=sdW,
                    final_minus_A=round(ex.res["configs"]["중첩: " + fkey]["f2"] - A_row["f2"], 4), _x=round(ex.res["configs"]["중첩: " + fkey]["f2"] - A_row["f2"], 4),
                    final_minus_B=round(ex.res["configs"]["중첩: " + fkey]["f2"] - B_row["f2"], 4), A_sd=A_row["f2_sd"], B_sd=B_row["f2_sd"]))
labs = ["A | EXP-034 b (EXP-040 값)", "B | ET 하나 (EXP-040 값)", "W | 2단계 승자 C-" + W + " (EXP-041 값)", "D1 | W + E1 (솟은 봉우리)", "D2 | D1 + E2 (하중 대비 덜컥)"]
bad = sorted({k for c in labs for k in list(ex.res["configs"]["중첩: " + c]["miss_counts"]) + list(ex.res["configs"]["중첩: " + c]["fp_counts"])
              if max(ex.res["configs"]["중첩: " + c]["miss_counts"].get(k, 0), ex.res["configs"]["중첩: " + c]["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12]

# %% 전문가별 보고 (C1–C4): 중첩 CV와 같은 분할에서 E 단독 켜짐 횟수 · 문턱 ±20 % (문턱 거리 × 0.8 / 1.2)
EXPR = {}
for e, aim in (("E1", "002형 (고주파 좁은 봉우리)"), ("E2", "017형 (하중 대비 고주파 높음)")):
    cn = expert_counts(ns, e, scales=(0.8, 1.0, 1.2))
    c1 = cn[1.0]; fault_on = [(names[i], int(c1[i])) for i in np.flatnonzero((y == 1) & (c1 >= 1))]; nor_on = [(names[i], int(c1[i])) for i in np.flatnonzero((y == 0) & (c1 >= 1))]
    EXPR[e] = dict(aim=aim, faults_on=fault_on, faults_on_ge3=[f for f, v in fault_on if v >= 3], normals_on=nor_on,
                   C2_faults_ge3=len([f for f, v in fault_on if v >= 3]),
                   C3_normals_union={str(s): int(((y == 0) & (cn[s] >= 1)).sum()) for s in cn}, C3_normals_mean_per_seed={str(s): round(float(cn[s][y == 0].sum() / 5), 1) for s in cn},
                   C3_normals_ge3={str(s): int(((y == 0) & (cn[s] >= 3)).sum()) for s in cn})
    log(f"[{e}] 겨냥 {aim} · 켜진 고장(5회 중) {fault_on} · 켜진 정상 {nor_on} · C2(≥3회 켜진 고장) {EXPR[e]['C2_faults_ge3']} · 문턱 ×0.8/1.0/1.2 정상 합집합 {EXPR[e]['C3_normals_union']} 시드당 평균 {EXPR[e]['C3_normals_mean_per_seed']}")
ex.note("전문가", EXPR)

# %% 그림: 핵심 = 최종 평면 + A 평면
Acls = ACls(ns.X, y, names)
fo = oof_fires(ns, ns.mk(W, fexp))
fig, axs = plt.subplots(1, 2, figsize=(18, 8.2))
plane_A(axs[0], ns, Acls, ex.res["configs"]["중첩: " + labs[0]], "A | EXP-034 b 재측정 (규칙 2개 사후)")
plane_W(axs[1], ns, ns.mk(W, fexp), fo, ex.res["configs"]["중첩: " + fkey], f"1차 최종 모델 {final} (결합 C-{W} + 전문가 {'+'.join(fexp) or '없음'})")
fig.suptitle("logistic_plane — 왼쪽 기준 A (AE z × ET z) | 오른쪽 1차 최종 모델 (ET 점수 × 박자 z, 경계, 전문가가 켠 파일 표시). OOF 5-fold seed 0", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "A와 1차 최종 모델의 평면. 전문가 E1(초록 마름모)·E2(보라 +)가 켠 파일과 자주 틀린 파일(×)", key=True)
ex.save_fig(hard_fig(ns, ex, labs, bad), "hard_files", "어느 파일이 A · B · W · D1 · D2에서 맞고 틀리는가")

# %% 오류 분석 표 (최종 기준) + 전문가 켜짐
m_all = ns.mk(W, fexp)().fit(ns.M, y); imp = m_all.m_.feature_importances_
files = ["nor_065", "nor_037", "nor_110", "nor_095", "nor_116", "nor_059", "abn_002", "abn_017"]
ET_, top = error_table(ns, files, imp)
ET_.to_csv(os.path.join(ex.dir, "오류분석_표.csv"), index=False, encoding="utf-8-sig")
c_e1, c_e2 = expert_counts(ns, "E1")[1.0], expert_counts(ns, "E2")[1.0]
fz = oof_fires(ns, ns.mk(W, fexp))
md = ["| 파일 | 구분 | ET logit | 박자 z | E1 켜짐 | E2 켜짐 | 정상 95% 위 피처 | 5% 아래 피처 | ET 중요도 상위와 겹침 |", "|---|---|---|---|---|---|---|---|---|"]
for r in ET_.itertuples():
    i = names.index(r.file)
    md.append(f"| {r.file} | {r.label} | {fz['a'][i]:.2f} | {fz['z'][i]:.2f} | {c_e1[i]}/5 | {c_e2[i]}/5 | {r.over95 or '—'} | {r.under5 or '—'} | {r.top_overlap or '—'} |")
open(os.path.join(ex.dir, "오류분석_표.md"), "w", encoding="utf-8").write("\n".join(md) + "\n\nET 중요도 상위 6: " + ", ".join(t.split('_')[0] for t in top) + "\n")
log("\n".join(md)); log("ET 중요도 상위:", top)

# %% val 채점 (1차 최종 모델, train 150 전체 학습) — 제출 파일 없음
def thr_of(mk_, seeds=(0, 1, 2)):
    t = []
    for s in seeds:
        o = np.empty(ns.n)
        for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(g, y, g): o[te] = mk_().fit(ns.M[tr], y[tr]).decision_function(ns.M[te])
        t.append(best_f2_threshold(y, o))
    return float(np.median(t)), t


mk_f = ns.mk(W, fexp); m_f = mk_f().fit(ns.M, y)
thr, ts = (0.5, []) if (W == "max" or fexp) else thr_of(mk_f)
m_f.thr_ = thr
V = val_table(ns, m_f)
A_val = pd.read_csv(os.path.join(EXP040, "val_판정.csv"), encoding="utf-8-sig")
assert list(A_val.file) == ns.nv
V.insert(1, "A_b", A_val["A"].values); V["B"] = A_val["B"].values; V["C"] = A_val["C"].values
V["S"] = ns.Tv.L1_S.values.round(3); V["H16"] = PeakRef(ns.df).fit(ns.L[ns.nor]).transform(ns.Lv)[:, 0].round(2)
V.to_csv(os.path.join(ex.dir, "val_판정.csv"), index=False, encoding="utf-8-sig")
lst = lambda m: " ".join(ns.nv[i][-3:] for i in np.flatnonzero(m))
pf, pa = V["pred"].values.astype(bool), V["A_b"].values.astype(bool)
blocks = {"결합(main)": lst(V["main"].values.astype(bool)), "E1": lst(V["E1"].values.astype(bool)) if "E1" in V else "", "E2": lst(V["E2"].values.astype(bool)) if "E2" in V else ""}
VJ = dict(final=final, thr=thr, thr_seeds=ts, n=int(pf.sum()), files=lst(pf), blocks=blocks, A_b=dict(n=int(pa.sum()), files=lst(pa)),
          overlap=int((pf & pa).sum()), only_final=lst(pf & ~pa), only_A=lst(pa & ~pf), 박자z_top=[(r.file[-3:], float(r.박자z)) for r in V.sort_values("박자z", ascending=False).head(5).itertuples()])
ex.note("val", VJ); log("val", json.dumps(VJ, ensure_ascii=False))

# %% t-SNE · psd (맨 끝)
sc_all = np.concatenate([fo["score"], m_f.fires(ns.Mv)["score"]])
ex.save_fig(tsne_fig(ns, sc_all, bad + ["abn_022", "abn_014"], f"{final} 결합 점수"), "tsne", "새 피처 공간과 ext150 공간에서 고장이 따로 모이는가 · 색 = 1차 최종 모델의 결합 점수")
f, Lp = viz.log_psd(ns.X)
ex.save_fig(viz.psd_band(f, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
