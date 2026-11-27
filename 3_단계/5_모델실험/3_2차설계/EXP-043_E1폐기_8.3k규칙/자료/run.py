# %% [markdown]
# EXP-043 (2차 설계, 11-21) E1 폐기 + 우리 8.3k 규칙 — 사용자 결정 (EXP-042 해석 뒤)
# - 이유: E1(솟은 봉우리 H16)은 겨냥한 002를 못 잡고 정상 097·101·116·120만 켰다(C2·C3 실패, EXP-042). 8.3k 규칙(src/final.py 정의, 스펙트럼 OR 공진)은 002를 잡고 정상 0.
# - 최종 구성 G = W(2단계 승자 C-lin: ET + 박자 z → 로지스틱) OR E2(H18 > 정상 99 % AND 충격 ≥ 1) OR 8.3k 규칙. 구성은 실행 전 고정
# - 비교용(같은 실행에서 재측정): F = W + E2 (규칙 없음) · H = W + 8.3k (E2 없음, E2의 몫 확인) · I = D2 + 8.3k (E1 포함, 폐기 효과 확인)
# - 8.3k 문턱(돌출 ≥ 0.75 dec · 폭 ≥ 150 Hz · 공진 fit ≥ 0.7)은 라벨을 본 뒤 정한 사후 값 (결정.md '사후 문턱 예외'). 학습 없음 → 폴드와 무관
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/3_2차설계/EXP-043_E1폐기_8.3k규칙/자료/run.py"

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
EXP041 = find_exp("EXP-041")
EXP042 = find_exp("EXP-042")
ex = Experiment(__file__, "EXP-043", "E1 폐기 + 8.3k 규칙 — W OR E2 OR 8.3k", ["H2", "H3", "H1"],
                member="이동훈", model="W(C-lin: ET + 박자 z → 로지스틱) OR E2(하중 대비 덜컥 H18 & 충격 ≥ 1) OR 8.3k 규칙(스펙트럼 OR 공진)",
                changes="E1 폐기(사용자 결정), 8.3k 규칙(src/final.py 정의) OR로 추가. 문턱은 EXP-042와 같음 (E2 = fit 정상 99 %)")
ns = NS(); names, y, g = ns.names, ns.y, ns.g
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


W = json.load(open(os.path.join(EXP041, "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]; log("2단계 승자 W =", W)
r42 = json.load(open(os.path.join(EXP042, "result.json"), encoding="utf-8"))["configs"]
for k, v in (("A | EXP-034 b (EXP-040 값)", r42["중첩: A | EXP-034 b (EXP-040 값)"]), ("B | ET 하나 (EXP-040 값)", r42["중첩: B | ET 하나 (EXP-040 값)"]),
             ("W | 2단계 승자 C-" + W + " (EXP-041 값)", r42["중첩: W | 2단계 승자 C-" + W + " (EXP-041 값)"]), ("D2 | W + E1 + E2 (EXP-042 값)", r42["중첩: D2 | D1 + E2 (하중 대비 덜컥)"])):
    ex.res["configs"]["중첩: " + k] = v
ex._write()

# 8.3k 규칙 켜짐 (학습 없음 → 행마다 고정). M 마지막 열로 붙여 Add83가 분리
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = spec | res; r83v = specv | resv
log("8.3k 규칙 켜진 train — 고장", [names[i] for i in np.flatnonzero(r83 & (y == 1))], "· 정상", [names[i] for i in np.flatnonzero(r83 & (y == 0))])
log("8.3k 규칙 켜진 val", [ns.nv[i][-3:] for i in np.flatnonzero(r83v)])
M_rule = np.column_stack([ns.M, r83.astype(float)]); Mv_rule = np.column_stack([ns.Mv, r83v.astype(float)])

# %% 중첩 CV
CFG = {"F | W + E2 (규칙 없음)": (ns.mk(W, ("E2",)), False),
       "G | W + E2 + 8.3k (최종)": (ns.mk(W, ("E2",)), True),
       "H | W + 8.3k (E2 없음)": (ns.mk(W), True),
       "I | W + E1 + E2 + 8.3k (E1 폐기 전)": (ns.mk(W, ("E1", "E2")), True)}
for nm, (mk_, with_rule) in CFG.items():
    F_, m_ = (M_rule, (lambda mk_=mk_: Add83(mk_))) if with_rule else (ns.M, mk_)
    c = ex.record("중첩: " + nm, nested_cv(F_, y, g, m_, n_jobs=-1), names, main=nm.startswith("G"))
    log(f"{nm:38s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · F2(10/40) {c['f2_p10']:.3f}/{c['f2_p40']:.3f} · 미탐 {c['miss_counts']} · 오경보 {c['fp_counts']}")
labs = ["A | EXP-034 b (EXP-040 값)", "B | ET 하나 (EXP-040 값)", "D2 | W + E1 + E2 (EXP-042 값)", "F | W + E2 (규칙 없음)", "G | W + E2 + 8.3k (최종)", "I | W + E1 + E2 + 8.3k (E1 폐기 전)"]
cf = lambda i: ex.res["configs"]["중첩: " + labs[i]]
G = cf(4)
log(f"G − A = {G['f2'] - cf(0)['f2']:+.3f} (A sd {cf(0)['f2_sd']}) · G − I = {G['f2'] - cf(5)['f2']:+.3f} · G − F = {G['f2'] - cf(3)['f2']:+.3f} (G sd {G['f2_sd']})")
bad = sorted({k for c in labs for k in list(cf(labs.index(c))["miss_counts"]) + list(cf(labs.index(c))["fp_counts"])
              if max(cf(labs.index(c))["miss_counts"].get(k, 0), cf(labs.index(c))["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12]

# %% 그림: 핵심 = 최종 평면 (왼쪽 기준 A, 오른쪽 G + 8.3k가 켠 고장 표시)
mkG = lambda: Add83(ns.mk(W, ("E2",)))
Acls = ACls(ns.X, y, names)
M_keep = ns.M; ns.M = M_rule                     # oof_fires · plane_W는 ns.M으로 fit → 규칙 열 포함 행렬로
fo = oof_fires(ns, mkG, seed=0)
fig, axs = plt.subplots(1, 2, figsize=(18, 8.2))
ns.M = M_keep; plane_A(axs[0], ns, Acls, cf(0), "A | EXP-034 b 재측정 (규칙 2개 사후)"); ns.M = M_rule
plane_W(axs[1], ns, mkG, fo, G, "최종 G (결합 C-" + W + " + 전문가 E2 + 8.3k 규칙)")
ns.M = M_keep
xc, zc = np.clip(fo["a"], -7, 7), np.clip(fo["z"], -3, 20)
rw = r83 & (y == 1)
axs[1].scatter(xc[rw], zc[rw], s=170, marker="D", facecolor="none", edgecolor="k", lw=1.5, label="8.3k 규칙이 켠 고장")
axs[1].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 왼쪽 기준 A (AE z × ET z) | 오른쪽 최종 G (ET 점수 × 박자 z, 경계, E2 · 8.3k가 켠 파일). OOF 5-fold seed 0", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "A와 최종 G의 평면. E2(보라 +)와 8.3k 규칙(검은 ◆)이 켠 파일, 자주 틀린 파일(×)", key=True)
ex.save_fig(hard_fig(ns, ex, labs, bad), "hard_files", "어느 파일이 A · B · D2 · F · G · I에서 맞고 틀리는가")

# %% 오류 분석 표 (최종 G 기준)
m_all = ns.mk(W, ("E2",))().fit(ns.M, y); imp = m_all.m_.feature_importances_
files = ["nor_065", "nor_095", "nor_110", "nor_116", "nor_059", "abn_002", "abn_017", "abn_022"]
ET_, top = error_table(ns, files, imp); ET_.to_csv(os.path.join(ex.dir, "오류분석_표.csv"), index=False, encoding="utf-8-sig")
c_e2 = expert_counts(ns, "E2")[1.0]
md = ["| 파일 | 구분 | ET logit | 박자 z | E2 켜짐 | 8.3k 규칙 | 정상 95% 위 피처 | 5% 아래 피처 |", "|---|---|---|---|---|---|---|---|"]
for r in ET_.itertuples():
    i = names.index(r.file)
    md.append(f"| {r.file} | {r.label} | {fo['a'][i]:.2f} | {fo['z'][i]:.2f} | {c_e2[i]}/5 | {'켜짐' if r83[i] else '—'} | {r.over95 or '—'} | {r.under5 or '—'} |")
open(os.path.join(ex.dir, "오류분석_표.md"), "w", encoding="utf-8").write("\n".join(md) + "\n"); log("\n".join(md))

# %% val 채점 (train 150 전체 학습) — 제출 파일 없음
m_f = mkG().fit(M_rule, y)
f = m_f.fires(Mv_rule); pred = m_f.decision_function(Mv_rule) >= 0.5
V = pd.DataFrame(dict(file=ns.nv, ET_logit=f["a"].round(2), 박자z=f["z"].round(2), main=f["main"].astype(int), E2=f["E2"].astype(int), R83=f["R83"].astype(int), pred=pred.astype(int)))
v42 = pd.read_csv(os.path.join(EXP042, "val_판정.csv"), encoding="utf-8-sig")
V["D2_042"] = v42["pred"].values; V["A_b"] = v42["A_b"].values
V.to_csv(os.path.join(ex.dir, "val_판정.csv"), index=False, encoding="utf-8-sig")
lst = lambda m: " ".join(ns.nv[i][-3:] for i in np.flatnonzero(m))
pf, pd2 = V["pred"].values.astype(bool), V["D2_042"].values.astype(bool)
VJ = dict(n=int(pf.sum()), files=lst(pf), 결합=lst(V["main"].values.astype(bool)), E2=lst(V["E2"].values.astype(bool)), R83=lst(V["R83"].values.astype(bool)),
          D2_042=dict(n=int(pd2.sum()), files=lst(pd2)), only_G=lst(pf & ~pd2), only_D2=lst(pd2 & ~pf))
ex.note("val", VJ); log("val", json.dumps(VJ, ensure_ascii=False))

# %% t-SNE · psd (맨 끝)
sc_all = np.concatenate([fo["score"], m_f.fires(Mv_rule)["score"]])
ex.save_fig(tsne_fig(ns, sc_all, bad + ["abn_022", "abn_014"], "G 결합 점수 (규칙 제외)"), "tsne", "새 피처 공간과 ext150 공간에서 고장이 따로 모이는가 · 색 = 결합 점수")
fq, Lp = viz.log_psd(ns.X)
ex.save_fig(viz.psd_band(fq, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
