# %% [markdown]
# EXP-022 EXP-021 기준으로 후보 모델 재검토 — 남은 오류(014·017·022, N-A 정상)를 가르는 모델이 있는가
# - 기준 = EXP-021 주 구성: AE + ET → 로지스틱 + 8.3 kHz 모드 규칙(돌출 ≥ 0.75 & 폭 ≥ 150 Hz → 무조건 고장). 중첩 F2 0.892 ± 0.015
#   남은 오류: 놓침 017 5/5 · 022 5/5 · 014 4/5 / 오경보 025 5/5 · 055 4/5 · 116 4/5 (N-A 큰 사건 한 번) · 059 3/5 (N-B)
# - 왜 다시 보나: EXP-019는 규칙 없는 EXP-014 기준이었다. 규칙이 002·029를 확실히 올리면 결합 임계값이 002·029에 끌리지 않아
#   후보가 남은 파일(014·017·022·N-A)에 주는 정보가 다르게 쓰일 수 있다
# - 후보 = EXP-019의 8종 (점수는 캐시 재사용, 하이퍼파라미터 그대로). 각 후보를
#   추가(AE + ET + 후보) / ET 자리 교체(AE + 후보) / AE 자리 교체(ET + 후보) 로 넣고 모두 규칙을 덮어쓴다
# - 판단 (실행 전 고정, 사용자 11-14 '수치보다 고장을 확실히 가르는가'): quick_cv OOF 분리 판정 (src.evaluate.separation)
#   통과 = 확실한 고장 수 ≥ 기준 + 1 이고, 회색 고장 수 ≤ 기준, 위험 정상(고장 하위 10%선 위 정상) ≤ 기준
#   통과 구성(확실한 고장 수 상위 2개까지)만 중첩 CV F2로 확인 (기준 0.892 ± 0.015)
# 실행: conda run -n qe-phm python 단계/5_모델비교/스크리닝/EXP-022_1114_EXP021기준후보재검토/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import RuleOverride, Rule83, ae_mean, et_logit, ae_segerr, et_trees, warm_cache
import src.candidates as C
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ex = Experiment(__file__, "EXP-022", "EXP-021 기준으로 후보 모델 재검토 (추가·교체, 분리 판정)", ["H1", "H2", "H3"],
                member="이동훈", model="EXP-021(AE + ET → 로지스틱 + 8.3k 규칙) + 후보 8종 추가·교체",
                changes="EXP-021 주 구성을 기준으로 EXP-019 후보를 추가·AE/ET 교체하고 OOF 분리 판정으로 비교", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
RULE = Rule83(0.75, 150)
KEY_A = ["abn_014", "abn_017", "abn_022", "abn_005", "abn_025", "abn_002", "abn_029"]
KEY_N = ["nor_025", "nor_055", "nor_116", "nor_059", "nor_060", "nor_062", "nor_086", "nor_106", "nor_027", "nor_073", "nor_039"]
KEY = KEY_A + KEY_N
CANDS = [("MiniRocket", C.minirocket_ridge), ("1D-CNN", C.cnn1d_mil), ("합성곱AE", C.convae_stft), ("DeepSAD", C.deepsad_stft),
         ("2D-CNN", C.cnn2d_mil), ("AR잔차", C.ar_resid_lr), ("MSPC", C.mspc_ext), ("CatBoost", C.catboost_ext)]

warm_cache([ae_segerr, et_trees], seeds=SEEDS)
for _, fn in CANDS:      # EXP-019에서 채운 캐시 확인 (없으면 채움)
    warm_cache([fn], seeds=SEEDS, n_jobs=1)


def make(experts):
    return lambda: RuleOverride(base_kw=dict(experts=experts, combine="lr"), rule=RULE)


def qcv(experts):
    q = quick_cv(g[:, None], y, g, make(experts), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    q["sep"] = separation(q["oof"], y, names)
    return q


# %% 기준 + 후보별 추가·교체
CONF = [("기준 EXP-021", (ae_mean, et_logit))]
for nm, fn in CANDS:
    CONF += [(f"{nm} 추가", (ae_mean, et_logit, fn)), (f"{nm} ET교체", (ae_mean, fn)), (f"{nm} AE교체", (et_logit, fn))]
R = {}
for name, exps in CONF:
    R[name] = qcv(exps)
    s = R[name]["sep"]
    print(f"{name:16s} PR-AUC {R[name]['pr_auc'].mean():.3f} · 확실한 고장 {s['sure_a']:2d} · 회색 고장 {len(s['gray_a']):2d} · "
          f"위험 정상 {s['risk_n']:5.1f} · 여유 {s['margin']:+.2f}", flush=True)

B = R["기준 EXP-021"]["sep"]
rows = [f"{'구성':16s} PR-AUC  확실한고장/30  회색고장  위험정상  여유    회색 고장 (기준 대비)                     판정"]
PASS = []
for name, q in R.items():
    s = q["sep"]
    ga = set(s["gray_a"])
    diff = " ".join([f"+{x[4:]}" for x in sorted(ga - set(B["gray_a"]))] + [f"−{x[4:]}" for x in sorted(set(B["gray_a"]) - ga)])
    ok = name != "기준 EXP-021" and s["sure_a"] >= B["sure_a"] + 1 and len(ga) <= len(B["gray_a"]) and s["risk_n"] <= B["risk_n"]
    if ok:
        PASS.append(name)
    rows.append(f"{name:16s} {q['pr_auc'].mean():.3f}   {s['sure_a']:4d}          {len(ga):3d}      {s['risk_n']:5.1f}    {s['margin']:+.2f}   "
                f"{' '.join(x[4:] for x in s['gray_a']):30s} [{diff or '같음'}]  {'통과' if ok else ('' if name.startswith('기준') else '탈락')}")
ex.note("분리 판정 (quick_cv OOF, seed 3 모두). 위험 정상 = 고장 하위 10%선 위 정상 수", "\n".join(rows))
print("\n".join(rows), flush=True)

# 남은 오류 파일별 위험도 (고장 = 1 − pct, 정상 = pct; 낮을수록 좋음)
lines = ["구성             " + " ".join(f"{n.replace('abn_', 'A').replace('nor_', 'N'):>5s}" for n in KEY)]
for name, q in R.items():
    lines.append(f"{name:16s} " + " ".join(f"{(1 - q['pct'][names.index(n)]) if n.startswith('abn') else q['pct'][names.index(n)]:5.2f}" for n in KEY))
ex.note("남은 오류 파일 위험도 (낮을수록 좋음. 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율)", "\n".join(lines))


# %% 그림: 후보별 분리 지표 (확실한 고장 · 회색 고장 · 위험 정상), 기준 선
names_c = list(R)
fig, axs = plt.subplots(1, 3, figsize=(17, 6.5), sharey=True)
for ax, key, lab, better in ((axs[0], "sure_a", "확실한 고장 수 (/30, 많을수록 좋음)", "high"),
                             (axs[1], "gray_a", "회색 고장 수 (적을수록 좋음)", "low"),
                             (axs[2], "risk_n", "위험 정상 수 (고장 하위 10%선 위, 적을수록 좋음)", "low")):
    v = np.array([len(R[k]["sep"][key]) if key == "gray_a" else R[k]["sep"][key] for k in names_c], float)
    b = v[0]
    col = ["#555555"] + [("#eb6834" if (x > b if better == "high" else x < b) else "#9aa5b1") for x in v[1:]]
    ax.barh(range(len(v)), v, color=col)
    ax.axvline(b, color="k", lw=1, ls="--")
    ax.set_title(lab, fontsize=10); ax.grid(alpha=.3, axis="x")
axs[0].set_yticks(range(len(names_c))); axs[0].set_yticklabels(names_c, fontsize=8); axs[0].invert_yaxis()
fig.suptitle("EXP-021 기준(점선) 대비 후보 추가·교체의 분리 지표. 주황 = 기준보다 좋음", fontsize=11)
fig.tight_layout()
ex.save_fig(fig, "separation_bars", "어떤 후보·넣는 방식이 확실한 고장을 늘리면서 회색 고장·위험 정상을 늘리지 않는가")

M = np.array([[(1 - R[k]["pct"][names.index(n)]) if n.startswith("abn") else R[k]["pct"][names.index(n)] for k in names_c] for n in KEY])
fig, ax = plt.subplots(figsize=(2 + .55 * len(names_c), 8))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="k" if M[i, j] < .7 else "w")
ax.axhline(len(KEY_A) - .5, color="k", lw=.8)
ax.set_yticks(range(len(KEY))); ax.set_yticklabels(KEY, fontsize=8)
ax.set_xticks(range(len(names_c))); ax.set_xticklabels(names_c, rotation=90, fontsize=7)
ax.set_title("남은 오류 파일 위험도 (위 = 고장, 아래 = 정상). 낮을수록 좋음", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.5); fig.tight_layout()
ex.save_fig(fig, "hard_files", "EXP-021이 놓친 014·017·022와 오경보 N-A(025·055·116)·059를 후보가 다르게 보는가")


# %% 통과 구성만 중첩 CV F2 (확실한 고장 수 상위 2개까지). 기준 0.892 ± 0.015 (EXP-021)
PASS = sorted(PASS, key=lambda k: -R[k]["sep"]["sure_a"])[:2]
ex.note("중첩 CV 대상", ", ".join(PASS) or "없음 (통과 구성 없음)")
if PASS:
    warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
    for name in PASS:
        exps = dict(CONF)[name]
        for fn in exps:
            if fn not in (ae_mean, et_logit):
                warm_cache([fn], seeds=range(5), nested=True, n_jobs=1)
        r = nested_cv(g[:, None], y, g, make(exps), n_jobs=-1)
        ex.record("중첩: " + name, r, names, main=(name == PASS[0]))
print("done", flush=True)
