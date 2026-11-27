# %% [markdown]
# 4-18 (11-25) J-축소 + K83 입력 피처 검증 (EDA, 모델 실험 없음)
# 대상 = ET 입력 15열(H7–H10은 JShrink와 같은 축소, H16 · H17은 학습 정상 중앙 log PSD 기준) + 박자 L1_S + 8.3k 규칙 재료 P83 · W83 · K83 · RES
# 열마다: 분리도(AUC + 부트스트랩 95 % CI) · 꼬리 · val 이동(학습 고장 섞임 기준과 비교) · 하중 의존 · 중복 · 어려운 파일 백분위
# 계산은 src/final.py j_tables 기본값 그대로 (11 kHz 변형 쓰지 않음). p0 · 봉우리 기준은 FinalJ와 같이 학습 정상 전체(이상치 3개 포함)
# val은 라벨 없이 쓴다 (분포 비교만). 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.data import load_folder
from src.final import j_tables, J_FIXED, J_RING, J_ALPHA, _PeakRef

plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
NOR, ABN, VAL, INK, MUTED = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575"
RNG = np.random.default_rng(0); FS = 25000

# %% 신호 → J 입력 (캐시: 이 폴더 figs/_jtab_*.npz, AE 입력 LP는 저장 안 함)
X, y, names = load_train(); names = list(names)
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val"))
KEEP = ["H", "S", "L", "df", "P83", "W83", "K83", "RES"]
_sos20 = butter(4, 20, btype="lowpass", fs=FS, output="sos")


def tables(tag, A):
    p = os.path.join(FIG, f"_jtab_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in KEEP + ["load"]}
    Tb = j_tables(A); Tb = {k: np.asarray(Tb[k]) for k in KEEP}
    Ac = A - A.mean(1, keepdims=True)
    Tb["load"] = np.log10(np.sqrt((sosfiltfilt(_sos20, Ac, axis=-1) ** 2).mean(1)))   # 하중 = 0–20 Hz 저역 RMS (H18 분모와 같은 필터)
    np.savez(p, **Tb); return Tb


Tt, Tv = tables("train", X), tables("val", Xv)
nor = y == 0

# %% J와 같은 변환: 울림 비율 축소(p0 = 학습 정상 전체) + 봉우리(학습 정상 중앙 log PSD)
i1 = J_FIXED.index("H1_nimp"); ri = [J_FIXED.index(c) for c in J_RING]
nn = np.rint(np.expm1(Tt["H"][:, i1])); m = nor & (nn > 0)
p0 = (Tt["H"][np.ix_(m, ri)] * nn[m, None]).sum(0) / nn[m].sum()


def shrink(H):
    H = np.array(H, float); n = np.rint(np.expm1(H[:, i1])); has = n > 0
    k = H[np.ix_(has, ri)] * n[has, None]
    H[np.ix_(has, ri)] = (k + J_ALPHA * p0) / (n[has, None] + J_ALPHA); return H


pr = _PeakRef(float(Tt["df"])).fit(Tt["L"][nor])
COLS = J_FIXED + ["H16_peak_h", "H17_peak_w", "L1_S", "P83", "W83", "K83", "RES"]


def frame(T):
    return pd.DataFrame(np.column_stack([shrink(T["H"]), pr.transform(T["L"]), T["S"], T["P83"], T["W83"], T["K83"], T["RES"]]), columns=COLS)


F, Fv = frame(Tt), frame(Tv)
F.insert(0, "file", names); F.insert(1, "y", y); F["load"] = Tt["load"]
Fv.insert(0, "file", nv); Fv["load"] = Tv["load"]
F.to_csv(os.path.join(FIG, "피처값_train.csv"), index=False, encoding="utf-8-sig")
Fv.to_csv(os.path.join(FIG, "피처값_val.csv"), index=False, encoding="utf-8-sig")
say(f"p0(H7–H10 축소 기준) = {np.round(p0, 3).tolist()}, α = {J_ALPHA}")
say(f"하중(log10 g) 학습 정상 중앙 {np.median(Tt['load'][nor]):.3f} · val 중앙 {np.median(Tv['load']):.3f}")

# %% 열 메타: 물리가설 · 근거 등급(new 4_근거등급)
META = {
    "H1_nimp": ("log1p(충격 수)", "H1", "L5 · L6"),
    "H2_ratio21": ("2위/1위 충격 높이(바탕 대비)", "H1′", "L9"),
    "H3_kurt_1-2k": ("1–2 kHz 대역 log10 첨도", "H1", "L10"),
    "H5_kurt_4-8k": ("4–8 kHz 대역 log10 첨도", "H1", "—"),
    "H7_ring_1-2k": ("충격 울림 1–2 kHz 비율(축소)", "H1", "—"),
    "H8_ring_2-4k": ("충격 울림 2–4 kHz 비율(축소)", "H1(연결부)", "L3 계열"),
    "H9_ring_4-8k": ("충격 울림 4–8 kHz 비율(축소)", "H1(본체=정상)", "L4 계열"),
    "H10_ring_8-12.5k": ("충격 울림 8–12.5 kHz 비율(축소)", "H2′", "L1 계열"),
    "H12_share_1-2k": ("1–2 kHz 파워 몫(log)", "H2 · H3", "—"),
    "H13_share_2-4k": ("2–4 kHz 파워 몫(log)", "H2 · H3", "—"),
    "H14_share_4-8k": ("4–8 kHz 파워 몫(log)", "H2 · H3", "L4"),
    "H15_share_8-12.5k": ("8–12.5 kHz 파워 몫(log)", "H2 · H3", "L10"),
    "H18_hfl": ("1 kHz 고역 포락 중앙 ÷ 20 Hz 저역 RMS(log)", "H1′(작은 덜컥)", "L8"),
    "H16_peak_h": ("정상 중앙 대비 최대 돌출(dec)", "H2", "X9"),
    "H17_peak_w": ("그 봉우리 반높이 폭(Hz)", "H2", "X9"),
    "L1_S": ("주기 잠금 S = r_LP × hit", "H4", "P4"),
    "P83": ("8.20–8.45 kHz 돌출(dec)", "H2′", "P1(문턱)"),
    "W83": ("8.3k 봉우리 반높이 폭(Hz)", "H2′", "P1(문턱)"),
    "K83": ("8.20–8.45 kHz 포락 첨도", "H2′(충격성)", "문턱 3 = 잡음 기준(라벨 무관)"),
    "RES": ("8.0–8.7 kHz · ζ≤0.02 충격 최대 fit", "H2′", "L2"),
}
HARD_N = ["nor_059", "nor_065", "nor_116", "nor_025"]
HARD_A = ["abn_002", "abn_017", "abn_022", "abn_014", "abn_028"]


# %% 지표 함수
def auc(v, yy):
    return roc_auc_score(yy, v)


def boot_auc(v, B=2000):
    i0, i1_ = np.flatnonzero(y == 0), np.flatnonzero(y == 1); out = np.empty(B)
    for b in range(B):
        s = np.r_[RNG.choice(i0, len(i0)), RNG.choice(i1_, len(i1_))]
        out[b] = auc(v[s], y[s])
    return np.percentile(out, [2.5, 97.5])


def p_gt(a, b):
    """P(a > b) + 0.5 P(=) (Mann–Whitney AUC)"""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def pctl(ref, v):
    ref = np.asarray(ref); return 100 * float(((ref < v) + 0.5 * (ref == v)).mean())


rho_all = F[COLS].corr(method="spearman").abs().values.copy()
np.fill_diagonal(rho_all, 0)
rows, hard_rows = [], []
for j, c in enumerate(COLS):
    v, vv = F[c].values.astype(float), Fv[c].values.astype(float)
    a = auc(v, y); d = 1.0 if a >= 0.5 else -1.0           # 방향: 고장 쪽이 커지게
    lo, hi = boot_auc(v); o, ov = d * v, d * vv
    on, oa = o[nor], o[~nor]
    # 꼬리: 고장 중 정상 95 %선 위 / 정상 중 고장 5 %선 아래 (방향 맞춘 값)
    tail_a = float((oa > np.percentile(on, 95)).mean()); tail_n = float((on < np.percentile(oa, 5)).mean())
    # 양쪽 꼬리 (원래 방향): 고장 중 정상 99 % 위 / 1 % 아래 — 평균 방향과 반대쪽에 고장 소집단이 있는지 (예: H18 · 017형)
    up99 = float((v[~nor] > np.percentile(v[nor], 99)).mean()); lo01 = float((v[~nor] < np.percentile(v[nor], 1)).mean())
    # val 이동: P(val > 학습 정상) (방향 맞춤) vs 학습 정상 40개 중 p만큼 고장으로 바꿔 뽑은 기준 (1000번)
    av = p_gt(ov, on); q1, q99 = np.percentile(on, [1, 99]); outv = float(((ov < q1) | (ov > q99)).mean())
    ref = {}
    for p in (0.0, 0.2, 0.4):
        k = int(round(40 * p)); sa, so = [], []
        for _ in range(1000):
            s = np.r_[RNG.choice(on, 40 - k), RNG.choice(oa, k)] if k else RNG.choice(on, 40)
            sa.append(p_gt(s, on)); so.append(((s < q1) | (s > q99)).mean())
        ref[p] = (np.percentile(sa, [2.5, 97.5]), np.percentile(so, 97.5))
    if av > ref[0.4][0][1]:
        sh = "넘침(고장 40 %로도)"
    elif av < ref[0.0][0][0]:
        sh = "반대(조건 이동)"
    elif av > ref[0.0][0][1]:
        sh = "고장 쪽(섞임으로 설명 가능)"
    else:
        sh = "안정"
    r_load = spearmanr(v[nor], Tt["load"][nor])[0]
    r_load_v = spearmanr(vv, Tv["load"])[0]
    k = int(rho_all[j].argmax())
    rows.append(dict(열=c, 정의=META[c][0], 가설=META[c][1], 등급=META[c][2], AUC=a, CI_lo=lo, CI_hi=hi, 분리도=abs(a - 0.5) * 2,
                     방향="+" if d > 0 else "-", 고장꼬리=tail_a, 정상꼬리=tail_n, 고장_정상99위=up99, 고장_정상1아래=lo01,
                     val_AUC=av, ref0_lo=ref[0.0][0][0], ref0_hi=ref[0.0][0][1], ref20_hi=ref[0.2][0][1], ref40_hi=ref[0.4][0][1],
                     val_범위밖=outv, 범위밖_ref0=ref[0.0][1], 범위밖_ref40=ref[0.4][1], val_판정=sh,
                     하중rho=r_load, 하중rho_val=r_load_v, 중복rho=rho_all[j, k], 중복상대=COLS[k]))
    hr = {"열": c}
    for f in HARD_N + HARD_A:
        hr[f] = pctl(on, o[names.index(f)])
    hard_rows.append(hr)

R = pd.DataFrame(rows); Hd = pd.DataFrame(hard_rows)
R.to_csv(os.path.join(FIG, "열별_검증.csv"), index=False, encoding="utf-8-sig", float_format="%.3f")
Hd.to_csv(os.path.join(FIG, "어려운파일_백분위.csv"), index=False, encoding="utf-8-sig", float_format="%.0f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
say("\n[열별 검증] AUC(고장 vs 정상) · CI · 분리도 · 꼬리 · val AUC(방향 맞춤 P(val>정상)) · 기준 범위 · 하중 ρ · 중복")
say(R.drop(columns=["정의", "등급"]).round(3).to_string(index=False))
say("\n[어려운 파일] 학습 정상 안 백분위(방향 맞춤: 높을수록 고장 쪽)")
say(Hd.round(0).to_string(index=False))

# 이상치 3개를 정상에서 뺐을 때 AUC (참고)
from src.exp import OUTLIER_NORMALS
keep = ~np.isin(names, list(OUTLIER_NORMALS))
say("\n[참고] 이상치 059 · 065 · 116 뺀 AUC: " + ", ".join(f"{c} {roc_auc_score(y[keep], F[c].values[keep]):.3f}" for c in COLS))
# 규칙 재료: 규칙 문턱 기준 학습 · val 켜짐 수
from src.final import P83_MIN, W83_MIN, K83_MIN, RES_FIT_MIN
for nm, D in (("train", F), ("val", Fv)):
    sp = (D.P83 >= P83_MIN) & (D.W83 >= W83_MIN); spk = sp & (D.K83 > K83_MIN); rs = D.RES >= RES_FIT_MIN
    say(f"[규칙] {nm}: 스펙트럼(P·W) {int(sp.sum())} · +K83 {int(spk.sum())} · 공진 {int(rs.sum())} · OR {int((spk | rs).sum())}"
        + (f" | 고장 {int(((spk | rs) & (D.y == 1)).sum())} 정상 {int(((spk | rs) & (D.y == 0)).sum())}" if "y" in D else "")
        + f" | 켜진 파일 {list(D.file[spk | rs])}")
# H16 · H17이 val에서 넘친 이유: 가장 솟은 봉우리가 어느 주파수에 있나 (val 배경 3–5 kHz +6 dB와 겹치는지)
fr = 1000 + np.arange(Tt["L"].shape[1]) * float(Tt["df"])
pf, pfv = fr[(Tt["L"] - pr.ref_).argmax(1)], fr[(Tv["L"] - pr.ref_).argmax(1)]
bins = [1000, 3000, 5000, 8000, 9000, 11000, 12600]
cut = lambda a: np.histogram(a, bins)[0]
say("\n[H16 봉우리 주파수] 칸 " + str(bins))
say(f"  정상 {cut(pf[nor])} · 고장 {cut(pf[~nor])} · val {cut(pfv)}")
say(f"  val 3–5 kHz 봉우리 파일 H16 중앙 {np.median(Fv.H16_peak_h[(pfv >= 3000) & (pfv < 5000)]) if ((pfv >= 3000) & (pfv < 5000)).any() else np.nan:.2f} · 학습 정상 H16 중앙 {np.median(F.H16_peak_h[nor]):.2f}")
F["H16_freq"], Fv["H16_freq"] = pf, pfv
F.to_csv(os.path.join(FIG, "피처값_train.csv"), index=False, encoding="utf-8-sig")
Fv.to_csv(os.path.join(FIG, "피처값_val.csv"), index=False, encoding="utf-8-sig")
say("\n[val 파일별] 학습 정상 1–99 % 밖 열 수 상위")
Q = np.array([[np.percentile(F[c][nor], 1), np.percentile(F[c][nor], 99)] for c in COLS])
cnt = ((Fv[COLS].values < Q[:, 0]) | (Fv[COLS].values > Q[:, 1])).sum(1)
say(pd.Series(cnt, index=nv).sort_values(ascending=False).head(10).to_string())

# %% 그림 1: 20열 분포 (정상 · 고장 · val), 정상 1–99 % 띠, 어려운 파일 표시
fig, axs = plt.subplots(4, 5, figsize=(20, 15))
for ax, c in zip(axs.ravel(), COLS):
    v, vv = F[c].values, Fv[c].values; r = R.set_index("열").loc[c]
    for k, (vals, col, lab) in enumerate(((v[nor], NOR, "정상 120"), (v[~nor], ABN, "고장 30"), (vv, VAL, "val 40"))):
        xj = k + RNG.uniform(-0.18, 0.18, len(vals))
        ax.scatter(xj, vals, s=12, color=col, alpha=0.55, lw=0)
        ax.hlines(np.median(vals), k - 0.28, k + 0.28, color=INK, lw=1.5)
    q1, q99 = np.percentile(v[nor], [1, 99]); ax.axhspan(q1, q99, color=NOR, alpha=0.07, lw=0)
    for f, mk in [(f, "D") for f in HARD_N] + [(f, "X") for f in HARD_A]:
        i = names.index(f); kx = 0 if f.startswith("nor") else 1
        ax.scatter(kx + 0.33, v[i], marker=mk, s=28, color="none" if mk == "D" else INK, edgecolor=INK, lw=0.9, zorder=5)
        ax.annotate(f[-3:], (kx + 0.38, v[i]), fontsize=6.5, color=MUTED, va="center")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["정상", "고장", "val"], fontsize=9)
    ax.set_title(f"{c}\nAUC {r.AUC:.2f} [{r.CI_lo:.2f}, {r.CI_hi:.2f}] · val {r.val_AUC:.2f} · {r.val_판정}", fontsize=9, color=INK)
    ax.tick_params(labelsize=8); ax.spines[["top", "right"]].set_visible(False)
    if c in ("H17_peak_w", "K83"):
        ax.set_yscale("log")
    elif c == "W83":
        ax.set_yscale("symlog", linthresh=50); ax.set_ylim(bottom=0)
fig.suptitle("J-축소 + K83 입력 20열: 정상 · 고장 · val 분포 (띠 = 학습 정상 1–99 %, ◇ 어려운 정상 · × 어려운 고장)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, "1_열별분포.png"), dpi=110); plt.close(fig)

# %% 그림 2: 분리도 vs val 이동(val AUC − 고장 20 % 섞임 기준 상한), 크기 = 하중 |ρ|
fig, ax = plt.subplots(figsize=(10, 6.5))
ex = R.val_AUC - R.ref20_hi
col = np.where(R.val_판정.str.startswith("넘침") | R.val_판정.str.startswith("반대"), ABN, np.where(R.val_판정.str.startswith("고장"), VAL, NOR))
ax.scatter(R.분리도, ex, s=40 + 300 * R.하중rho.abs(), c=col, alpha=0.75, edgecolor="white", lw=1)
for _, r in R.iterrows():
    ax.annotate(r.열, (r.분리도, r.val_AUC - r.ref20_hi), fontsize=8, color=INK, xytext=(5, -11 if r.열 == "H16_peak_h" else 3), textcoords="offset points")
ax.axhline(0, color=MUTED, lw=0.8, ls="--"); ax.text(0.99, 0.004, "고장 20 % 섞임 기준 상한", fontsize=8, color=MUTED, ha="right", transform=ax.get_yaxis_transform())
ax.set_xlabel("분리도 |AUC - 0.5| × 2 (학습 150)"); ax.set_ylabel("val 이동 초과 = val AUC - (고장 20 % 섞임 97.5 %)")
ax.set_title("분리도 vs val 이동 (점 크기 = 학습 정상 안 하중 |ρ|, 주황 = 고장 40 %로도 넘침 또는 반대 방향)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_분리도_vs_val이동.png"), dpi=120); plt.close(fig)
say("\n그림 저장 완료"); OUT.close()
