# %% [markdown]
# EXP-041 (2차 설계 2단계, 11-21) C 위에서 결합만 바꾼다: C-lin(EXP-040 C 그대로) · C-max · C-svm
# - 구성은 실행 전 고정(계획 5절 + 사용자 요청 11-21). 피처 = 4-7 최종 목록, ET 기본값, 중첩 CV 바깥 5 × 안쪽 4 seed 5 (EXP-040과 같은 split)
# - C-lin: EXP-040 C 결과를 가져온다(같은 폴드·같은 시드 → 같은 값). 다시 돌리지 않음
# - C-max: 고장 = (ET logit ≥ t_H) OR (박자 z ≥ q99). t_H = fit 행 ET 안쪽 OOF logit의 F2 최대 문턱, q99 = fit 행 정상 박자 z 99 % 분위(고정).
#   nested_cv에서 문턱이 다시 골라지지 않게 점수를 이진(켜짐 1 / 0)으로 돌려준다 (src/models2.py Two 독스트링)
# - C-svm: RBF SVM(C = 1, gamma = 'scale', class_weight = balanced), 입력 = [ET OOF logit, 박자 z]를 표준화
# - 선택 규칙: 차 < sd면 max → lin → svm 순서(단순한 쪽)
# 실행: conda run -n qe-phm python "단계/5_모델비교/base/EXP-041_1121_2차설계_2단계결합/run.py"

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.expA import A as ACls
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
ex = Experiment(__file__, "EXP-041", "2차 설계 2단계 — C 위에서 결합 방식 비교 (max · lin · svm)", ["H1", "H2", "H4"],
                member="이동훈", model="ET(H 15열) OOF logit + 박자 z → C-max(OR) / C-lin(로지스틱) / C-svm(RBF)",
                changes="결합만 바꿈. 박자 문턱 q99 · t_H는 fit 안에서 고정 규칙(튜닝 없음)")
ns = NS(); names, y, g = ns.names, ns.y, ns.g
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


# %% 중첩 CV
r40 = json.load(open(os.path.join(EXP040, "result.json"), encoding="utf-8"))["configs"]
ex.res["configs"]["중첩: A | EXP-034 b (EXP-040 값)"] = r40["중첩: A | EXP-034 b 재측정"]
ex.res["configs"]["중첩: B | ET 하나 (EXP-040 값)"] = r40["중첩: B | ET 하나 (H + 박자 S)"]
ex.res["configs"]["중첩: C-lin | EXP-040 C 그대로"] = r40["중첩: C | ET(H) + 박자 z → 로지스틱"]; ex._write()
CFG = {"C-max | (ET ≥ t_H) OR (박자 z ≥ q99)": ns.mk("max"), "C-svm | RBF SVM [ET OOF logit, 박자 z]": ns.mk("svm")}
for nm, mk_ in CFG.items():
    r = nested_cv(ns.M, y, g, mk_, n_jobs=-1)
    c = ex.record("중첩: " + nm, r, names)
    log(f"{nm:44s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · F2(10/40) {c['f2_p10']:.3f}/{c['f2_p40']:.3f} · 미탐 {c['miss_counts']} · 오경보 {c['fp_counts']}")
rows = {"max": ex.res["configs"]["중첩: C-max | (ET ≥ t_H) OR (박자 z ≥ q99)"], "lin": ex.res["configs"]["중첩: C-lin | EXP-040 C 그대로"],
        "svm": ex.res["configs"]["중첩: C-svm | RBF SVM [ET OOF logit, 박자 z]"]}
log("C-lin (EXP-040 C 가져옴):", rows["lin"]["f2"], "±", rows["lin"]["f2_sd"])

# %% 선택 규칙 (max → lin → svm, 차 < 최고의 sd면 앞쪽)
best = max(rows, key=lambda k: rows[k]["f2"]); sd = rows[best]["f2_sd"]
win = next(k for k in ("max", "lin", "svm") if rows[best]["f2"] - rows[k]["f2"] < sd)
A_row, B_row = ex.res["configs"]["중첩: A | EXP-034 b (EXP-040 값)"], ex.res["configs"]["중첩: B | ET 하나 (EXP-040 값)"]
log(f"최고 F2 = {best} ({rows[best]['f2']:.3f} ± {sd:.3f}) → 선택 규칙 승자 = {win}")
log(f"승자 − A(b) {rows[win]['f2'] - A_row['f2']:+.3f} (A sd {A_row['f2_sd']}) · 승자 − B {rows[win]['f2'] - B_row['f2']:+.3f} (B sd {B_row['f2_sd']})")
ex.note("선택", dict(best=best, winner=win, rule="차 < 최고 sd면 max → lin → svm 순으로 단순한 쪽",
                    f2={k: rows[k]["f2"] for k in rows}, sd={k: rows[k]["f2_sd"] for k in rows},
                    winner_minus_A=round(rows[win]["f2"] - A_row["f2"], 4), winner_minus_B=round(rows[win]["f2"] - B_row["f2"], 4)))
labs = ["A | EXP-034 b (EXP-040 값)", "B | ET 하나 (EXP-040 값)", "C-lin | EXP-040 C 그대로", "C-max | (ET ≥ t_H) OR (박자 z ≥ q99)", "C-svm | RBF SVM [ET OOF logit, 박자 z]"]
bad = sorted({k for c in labs for k in list(ex.res["configs"]["중첩: " + c]["miss_counts"]) + list(ex.res["configs"]["중첩: " + c]["fp_counts"])
              if max(ex.res["configs"]["중첩: " + c]["miss_counts"].get(k, 0), ex.res["configs"]["중첩: " + c]["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12]

# %% 그림: 핵심 = 승자 평면 + A 평면
key_of = {"max": "C-max | (ET ≥ t_H) OR (박자 z ≥ q99)", "lin": "C-lin | EXP-040 C 그대로", "svm": "C-svm | RBF SVM [ET OOF logit, 박자 z]"}
ex.res["main"] = "중첩: " + key_of[win]; ex._write()
Acls = ACls(ns.X, y, names)
fo = oof_fires(ns, ns.mk(win))
fig, axs = plt.subplots(1, 2, figsize=(18, 8.2))
plane_A(axs[0], ns, Acls, ex.res["configs"]["중첩: " + labs[0]], "A | EXP-034 b 재측정 (규칙 2개 사후)")
plane_W(axs[1], ns, ns.mk(win), fo, ex.res["configs"]["중첩: " + key_of[win]], f"2단계 승자 C-{win}: H 점수 × 박자 z (규칙 없음)")
fig.suptitle("logistic_plane — 왼쪽 기준 A (AE z × ET z) | 오른쪽 2단계 승자 (ET 점수 × 박자 z, 경계 · 문턱선). OOF 5-fold seed 0", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "A는 규칙이 고장 일부를 빼고, 승자는 ET 점수 × 박자 z 평면에서 022 · 014가 박자 축으로 따로 서는가", key=True)
ex.save_fig(hard_fig(ns, ex, labs, bad), "hard_files", "어느 파일이 A · B · C-lin · C-max · C-svm에서 맞고 틀리는가")

# %% 오류 분석 표 (승자 기준): 새 오경보 · 놓친 002 · 017
m_all = ns.mk(win)().fit(ns.M, y)
imp = m_all.m_.feature_importances_
files = ["nor_065", "nor_037", "nor_110", "nor_095", "nor_116", "nor_059", "abn_002", "abn_017"]
ET_, top = error_table(ns, files, imp)
ET_.to_csv(os.path.join(ex.dir, "오류분석_표.csv"), index=False, encoding="utf-8-sig")
fz = oof_fires(ns, ns.mk(win))
ET_["ET_logit(OOF)"] = [round(float(fz["a"][names.index(f)]), 2) for f in files]; ET_["박자z"] = [round(float(fz["z"][names.index(f)]), 2) for f in files]
ET_ = ET_.rename(columns={"ET_logit(OOF)": "etl", "박자z": "bz"})
md = ["| 파일 | 구분 | ET logit | 박자 z | 정상 95% 위 피처 | 5% 아래 피처 | ET 중요도 상위와 겹침 |", "|---|---|---|---|---|---|---|"]
for r in ET_.itertuples():
    md.append(f"| {r.file} | {r.label} | {r.etl} | {r.bz} | {r.over95 or '—'} | {r.under5 or '—'} | {r.top_overlap or '—'} |")
open(os.path.join(ex.dir, "오류분석_표.md"), "w", encoding="utf-8").write("\n".join(md) + f"\n\nET 중요도 상위 {len(top)}: " + ", ".join(t.split('_')[0] for t in top) + "\n")
ex.note("오류분석", dict(top_importance=top, over95={r.file: r.over95 for r in ET_.itertuples()}))
log("ET 중요도 상위:", top); log("\n".join(md))

# %% t-SNE · psd (맨 끝)
sc_all = np.concatenate([fo["score"], m_all.fires(ns.Mv)["score"]])
ex.save_fig(tsne_fig(ns, sc_all, bad + ["abn_022", "abn_014"], f"C-{win} 결합 점수"), "tsne", "새 피처 공간과 ext150 공간에서 고장이 따로 모이는가 · 색 = 승자 결합 점수 (train OOF, val은 train 전체 학습)")
f, Lp = viz.log_psd(ns.X)
ex.save_fig(viz.psd_band(f, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
