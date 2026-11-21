# %% [markdown]
# EXP-046 (11-22) 최종 후보 J 위에 세 가지: 고주파 사건 반복 수(hf_rep) · N1 반동(n1_reb) · 8.3k 규칙 → u_ring 로지스틱 4번째 입력
# - 사전 등록: 단계/5_모델비교/검토문서/협의_1121/사전등록_046.md (구성 G0–G4 · 정의 · 선택 규칙은 실행 전 고정, 결과를 보고 고치지 않는다)
#   G0 = J 재측정 · G1 = J + hf_rep(ET 열 1개, lo는 fit 안에서 단변량 AUC로) · G2 = J + n1_reb(ET 열 1개)
#   G3 = J − 8.3k OR + u_ring(로지스틱 4번째 입력, 키·t는 fit 안 pick_ring_m, m = 0.2) · G4 = G3 + G1 열 + G2 열
# - 가설: H1 (유격 반복 충격) · H2 (고주파 울림) · H3 (정상 증거)
# - 코드: src/features3.py (피처) · src/models3.py (모델). 기존 파일은 고치지 않았다
# 실행: conda run -n qe-phm python "단계/5_모델비교/EXP-046_1122_J위세가지/run.py"

# %% 설정
import os, sys, json, warnings
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.features3 import cached_feat3, COLS, HF_GRID, RING_KEYS, N_EX, I_HF, I_N1
from src.models3 import Models3
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
ex = Experiment(__file__, "EXP-046", "J 위 세 가지 (hf_rep · n1_reb · u_ring)", ["H1", "H2", "H3"],
                member="이동훈", model="J = ET OOF logit + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k (G0), 세 가지 변형 G1–G4",
                changes="J에 hf_rep 열(G1) · n1_reb 열(G2) · 8.3k OR 대신 u_ring 4번째 입력(G3) · 셋 모두(G4). 사전등록_046")
ns = NS(); names, y, g = ns.names, ns.y, ns.g
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()
SEL = os.path.join(ex.figs, "selection_log.jsonl"); open(SEL, "w", encoding="utf-8").close()
os.environ["EXP046_LOG"] = SEL


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


Dt, Dv = cached_feat3("train", ns.X), cached_feat3("val", ns.Xv)
assert (Dt.n_imp[y == 0].sum(), Dt.n_imp[y == 1].sum(), Dv.n_imp.sum()) == (854, 402, 247), "충격 수 불일치"
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
EX, EXv = Dt[COLS].values, Dv[COLS].values
nM = ns.M.shape[1]
F = np.column_stack([ns.M, EX, np.arange(ns.n, dtype=float), r83])
Fv = np.column_stack([ns.Mv, EXv, np.zeros(len(ns.Mv)), r83v])


def mk(tag, **kw):
    return lambda: Models3(nM=nM, fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), tag=tag, **kw)


CFG = {"G0 | J 재측정": ("G0", dict(or83=True)),
       "G1 | J + hf_rep": ("G1", dict(hf=True, or83=True)),
       "G2 | J + n1_reb": ("G2", dict(n1=True, or83=True)),
       "G3 | J − 8.3k + u_ring": ("G3", dict(ring=True, or83=False)),
       "G4 | G3 + hf_rep + n1_reb": ("G4", dict(hf=True, n1=True, ring=True, or83=False))}
labs = list(CFG)

# %% 중첩 CV (G0 먼저, 재현 확인 후 나머지)
RAW = {}
for nm, (tag, kw) in CFG.items():
    RAW[nm] = nested_cv(F, y, g, mk(tag, **kw), n_jobs=-1)
    c = ex.record("중첩: " + nm, RAW[nm], names, main=nm.startswith("G0"))
    log(f"{nm:28s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · 미탐 {c['miss_counts']} · 오경보 {c['fp_counts']}")
    if tag == "G0" and abs(c["f2"] - 0.977) > 0.01:
        log("G0 재현 실패 (0.977 ± 0.01 밖) — 중단"); ex.note("abort", "G0 재현 실패"); sys.exit(1)
bad = sorted({k for nm in labs for k in list(ex.res["configs"]["중첩: " + nm]["miss_counts"]) + list(ex.res["configs"]["중첩: " + nm]["fp_counts"])
              if max(ex.res["configs"]["중첩: " + nm]["miss_counts"].get(k, 0), ex.res["configs"]["중첩: " + nm]["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12]

# %% 선택 기록 (lo · u_ring 키 · t · 계수 부호)
import glob
recs = [json.loads(l) for fn in glob.glob(SEL + "*") for l in open(fn, encoding="utf-8")]
STAB = {}
for nm, (tag, kw) in CFG.items():
    R = [r for r in recs if r["tag"] == tag and r["n"] < 140]
    s = dict(n_fits=len(R))
    if kw.get("hf"):
        s["lo_counts"] = dict(Counter(r["lo"] for r in R).most_common())
    if kw.get("ring"):
        kc = Counter(tuple(r["key"]) for r in R); top = kc.most_common(1)[0]
        s["key_counts"] = {str(k): v for k, v in kc.most_common()}; s["key_agree_pct"] = round(100 * top[1] / len(R), 1)
        s["t_counts"] = dict(Counter(round(r["t"], 3) for r in R).most_common()); s["t_range"] = [min(r["t"] for r in R), max(r["t"] for r in R)]
        s["coef_u_pos_pct"] = round(100 * float(np.mean([r["coef_u"] > 0 for r in R])), 1); s["coef_u_median"] = float(np.median([r["coef_u"] for r in R]))
    STAB[nm] = s
    log(nm, json.dumps(s, ensure_ascii=False))
ex.note("stability", STAB)

# %% 선택 규칙 (사전등록 그대로)
rw = {nm: RAW[nm] for nm in labs}
f2 = {nm.split(" | ")[0]: float(rw[nm]["f2"].mean()) for nm in labs}; sd = {k.split(" | ")[0]: float(rw[k]["f2"].std()) for k in labs}
fp_ = {nm.split(" | ")[0]: float(rw[nm]["fpr"].mean()) for nm in labs}
sd0 = sd["G0"]
adopt = {}
for k in ("G1", "G2"):
    adopt[k] = (f2[k] - f2["G0"]) > sd0
agree = STAB[labs[3]]["key_agree_pct"]
adopt["G3"] = (f2["G3"] >= f2["G0"] - sd0) and (fp_["G3"] <= fp_["G0"] + 0.010) and agree >= 60
combo = ("G3" if adopt["G3"] else "G0") + (" + hf_rep" if adopt["G1"] else "") + (" + n1_reb" if adopt["G2"] else "")
RUN_AS = {"G0": "G0", "G0 + hf_rep": "G1", "G0 + n1_reb": "G2", "G3": "G3", "G3 + hf_rep + n1_reb": "G4"}
final = RUN_AS.get(combo)
g4_final = adopt["G1"] and adopt["G2"] and adopt["G3"]
RULE = dict(adopt=adopt, f2=f2, sd=sd, fpr=fp_, sd0=sd0, key_agree=agree, combo=combo, final_run_config=final, G4_is_final_candidate=bool(g4_final),
            note=("조합이 실행한 구성과 다름 → 새 실험 없이 다음 후보" if final is None else f"최종 = {final}"))
ex.note("rule", RULE); log("rule", json.dumps(RULE, ensure_ascii=False, default=bool))
chosen = labs[[l.split(" | ")[0] for l in labs].index(final if final else "G4")]
if final is None: log("조합이 실행 구성과 다름 → plane은 G4(참고)로 그림")
ex.res["main"] = "중첩: " + (chosen if final else labs[0]); ex._write()

# %% OOF (seed 0) 와 val — train 150 전체 학습
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); ae_val = _ae_score(ae_tr, LPv)


def oof_run(tag, kw, seed=0):
    o = dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), u=np.zeros(ns.n), score=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
        f = mk("P" + tag, **kw)().fit(F[tr], y[tr]).fires(F[te])
        for k in o:
            if k in f: o[k][te] = f[k]
    return o


FO, MD, VF, VP = {}, {}, {}, {}
for nm, (tag, kw) in CFG.items():
    FO[nm] = oof_run(tag, kw); m = mk("Pfull" + tag, **kw)().fit(F, y); m.ae_override_ = ae_val; MD[nm] = m
    VF[nm] = m.fires(Fv); VP[nm] = m.decision_function(Fv) >= 0.5
lst = lambda mask: " ".join(ns.nv[i][-3:] for i in np.flatnonzero(mask))
VAL = {}
for nm in labs:
    kw = CFG[nm][1]; m, f, p = MD[nm], VF[nm], VP[nm]; p0 = VP[labs[0]]; sdv = float(np.std(FO[nm]["score"]))
    VAL[nm] = dict(n=int(p.sum()), files=lst(p), 겹침_G0=int((p & p0).sum()), only_this=lst(p & ~p0), only_G0=lst(p0 & ~p),
                   main=lst(f["main"]), E2=lst(f["E2"]) if "E2" in f else "", R83=lst(f["R83"]), u_on=lst(f["u"] >= 1) if "u" in f else "",
                   경계밀집=int((np.abs(f["score"] - m.tc_) < 0.5 * sdv).sum()), 경계_파일=lst(np.abs(f["score"] - m.tc_) < 0.5 * sdv), oof_sd=round(sdv, 3), tau=round(float(m.tc_), 3),
                   lo_full=(HF_GRID[m.hf_j_] if kw.get("hf") else None), key_full=(list(RING_KEYS[m.rk_]) + [round(m.rt_, 3)] if kw.get("ring") else None))
    log(nm, json.dumps(VAL[nm], ensure_ascii=False))
ex.note("val", VAL)
pd.DataFrame(dict(file=ns.nv, **{nm.split(" | ")[0]: VP[nm].astype(int) for nm in labs}, **{nm.split(" | ")[0] + "_score": np.round(VF[nm]["score"], 3) for nm in labs})).to_csv(os.path.join(ex.dir, "val_판정.csv"), index=False, encoding="utf-8-sig")
# train 오경보 · 미탐이 G0과 달라진 파일 (횟수 차이 >= 1)
DIFF = {}
for nm in labs[1:]:
    a, b = ex.res["configs"]["중첩: " + labs[0]], ex.res["configs"]["중첩: " + nm]
    d = {}
    for kind in ("miss_counts", "fp_counts"):
        for fl in set(a[kind]) | set(b[kind]):
            if a[kind].get(fl, 0) != b[kind].get(fl, 0): d[f"{fl} {'미탐' if kind == 'miss_counts' else '오경보'}"] = f"{a[kind].get(fl, 0)}→{b[kind].get(fl, 0)}"
    DIFF[nm] = d; log(nm, "G0 대비 변화", d)
ex.note("diff_vs_G0", DIFF)

# %% 그림: logistic_plane 핵심 — G0 vs 규칙이 고른 구성 (ET 점수 × AE z, 박자 z = 0 · u = 0 단면)
def panel(ax, nm):
    fo, m = FO[nm], MD[nm]; c = ex.res["configs"]["중첩: " + nm]
    a, aez, nor = np.clip(fo["a"], -7, 7), np.clip(fo["ae"], -3, 12), y == 0
    xx, yy = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-3, 12, 300))
    D = m._score(xx.ravel(), np.zeros(xx.size), yy.ravel(), np.zeros(xx.size)).reshape(xx.shape)
    ax.contourf(xx, yy, D, levels=[-1e9, m.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy, D, levels=[m.tc_], colors="k", linewidths=1.5)
    caught = (y == 1) & ((fo["R83"]) | (fo["u"] >= 1))                              # 규칙 · u_ring 으로 잡힌 고장은 평면에서 뺀다 (t-SNE에는 남김)
    show = ~caught
    ax.scatter(a[nor], aez[nor], color=NOR, s=24, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(a[~nor & show], aez[~nor & show], color=ABN, s=30, lw=0, label="고장 (규칙·u 밖)")
    e2 = fo["E2"] & show; ax.scatter(a[e2], aez[e2], s=130, marker="P", facecolor="none", edgecolor="#7a3db8", lw=1.5, label="E2가 켠 파일")
    w = wrong_of(ns, c); wi = list(w); ax.scatter(a[wi], aez[wi], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for k, i in enumerate(sorted(wi, key=lambda i: a[i])):
        ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {w[i]}/5", (a[i], aez[i]), xytext=(6, 8 + 10 * (k % 3)), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR, weight="bold")
    gone = [short(names[i]) for i in np.flatnonzero(caught)]
    ax.text(.98, .02, f"규칙 · u_ring이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlabel("ET 점수 (logit, OOF)"); ax.set_ylabel("AE z (정상 기준, OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{nm}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} (경계는 박자 z = 0{' · u = 0' if CFG[nm][1].get('ring') else ''} 단면)", fontsize=9.5)


fig, axs = plt.subplots(1, 2, figsize=(18, 8.2)); panel(axs[0], labs[0]); panel(axs[1], chosen)
fig.suptitle(f"logistic_plane — 왼쪽 G0 (J) | 오른쪽 {'규칙이 고른 ' + final if final else '규칙 결론은 다음 후보라 G4(참고)'}. OOF 5-fold seed 0", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "G0(J)와 선택 규칙이 고른 구성의 평면. 8.3k·u_ring이 잡은 고장은 평면에서 빼고 상자에 목록, E2(보라 +), 자주 틀린 파일(×)", key=True)
ex.save_fig(hard_fig(ns, ex, labs, bad), "hard_files", "어느 파일이 G0–G4에서 맞고 틀리는가")

# %% hf_rep · n1_reb 분포 (정상 / 고장 / val)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8)); nor = y == 0
j1 = MD[labs[1]].hf_j_; lo1 = HF_GRID[j1]
for ax, (col_t, col_v, ttl) in zip(axs, [(EX[:, I_HF + j1], EXv[:, I_HF + j1], f"hf_rep (lo = {lo1} Hz, G1 전체 학습이 고른 값)"), (EX[:, I_N1], EXv[:, I_N1], "n1_reb (N1 반동)"),
                                         (EX[:, I_HF + HF_GRID.index(8750)], EXv[:, I_HF + HF_GRID.index(8750)], "hf_rep (lo = 8750 Hz, 참고 · 사령탑이 본 값)")]):
    for k, (v, lab, cc) in enumerate([(col_t[nor], f"정상 {nor.sum()}", NOR), (col_t[~nor], f"고장 {(~nor).sum()}", ABN), (col_v, f"val {len(col_v)}", PUR)]):
        ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).uniform(-.18, .18, len(v)), s=18, color=cc, alpha=.7, lw=0, label=lab)
        ax.plot([np.median(v)] * 2, [k - .3, k + .3], color="k", lw=1.5)
    ax.set_yticks([0, 1, 2], ["정상", "고장", "val"]); ax.set_title(ttl, fontsize=9.5); ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8)
fig.tight_layout(); ex.save_fig(fig, "dist_hf_n1", "새 열의 분포. 굵은 검은 막대 = 중앙값. 고장이 정상과 갈라지는가, val은 어디에 놓이는가")

# %% t-SNE (G0 결합 점수 색, 규칙으로 잡힌 고장도 모두 포함) · psd (맨 끝)
ex.save_fig(tsne_fig(ns, np.concatenate([FO[labs[0]]["score"], VF[labs[0]]["score"]]), bad + ["abn_022", "abn_014"], "G0 결합 점수 (규칙 제외)"), "tsne", "새 피처 공간과 ext150 공간에서 고장이 따로 모이는가 · 색 = G0 결합 점수")
fq, Lp = viz.log_psd(ns.X)
ex.save_fig(viz.psd_band(fq, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
