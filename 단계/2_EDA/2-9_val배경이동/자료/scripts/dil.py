import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train
from src.impacts import highpass
FS = 25000; W = 500                                    # 20 ms 창
X, y, names = load_train(); names = [str(n) for n in names]
z = np.load("/home/user/PHM_Tension_Arm/outputs/cache/pho001_train.npz", allow_pickle=True); T = z["T"].item(); imps = list(z["imps"])
rows = []
for x, im in zip(X, imps):
    h = highpass(x); w = np.sqrt((h[:len(h) // W * W].reshape(-1, W) ** 2).mean(1))
    floor = np.percentile(w, 10)
    t = np.sort(im["t"]); n = len(t)
    gaps = np.diff(np.r_[0.0, t, 2.0])
    iso = np.mean([np.all(np.abs(np.delete(t, k) - t[k]) > 0.1) for k in range(n)]) if n else 1.0
    if n:
        top = im["t"][np.argmax(im["height_x"])]; m = np.ones(len(h), bool); a = int((top - .03) * FS); m[max(a, 0):int((top + .03) * FS)] = False
    else:
        m = np.ones(len(h), bool)
    r_all = np.sqrt((h ** 2).mean()); r_ex = np.sqrt((h[m] ** 2).mean())
    fitmax = im["rings"][:, 2].max() if n else 0.0
    q = (w <= 3 * floor).mean()
    rows.append(dict(n_imp=n, peak_med=w.max() / np.median(w), quiet=q, max_gap=gaps.max(), iso=iso,
                     rms_ex_top1=r_ex, rms_ex_frac=r_ex / r_all, eff_res=fitmax * (1 - q), fitmax=fitmax))
D = pd.DataFrame(rows, index=names); D["y"] = y
nor = y == 0
print("=== 단독 AUC (고장이 큰 쪽 기준, 0.5 미만 = 정상이 더 큼) · 기존 피처와 순위상관 ===")
for c in ["peak_med", "quiet", "max_gap", "iso", "rms_ex_top1", "rms_ex_frac", "eff_res"]:
    v = D[c].values
    print(f"{c:12s} AUC {roc_auc_score(y, v):.2f} | ρ H1충격수 {spearmanr(v, T['H1_nimp'])[0]:+.2f} · H2(2위/1위) {spearmanr(v, T['H2_ratio21'])[0]:+.2f} · H5첨도 {spearmanr(v, T['H5_kurt_4-8k'])[0]:+.2f}")
print("\n=== 제안 문서가 '단발 정상'이라 한 파일과 어려운 고장 (정상 중 백분위 %) ===")
cols = ["n_imp", "quiet", "max_gap", "iso", "rms_ex_frac", "peak_med", "eff_res"]
pct = lambda c, v: (D.loc[nor, c] < v).mean() * 100
for fn in ["nor_027", "nor_059", "nor_065", "nor_116", "nor_025", "abn_002", "abn_014", "abn_017", "abn_022", "abn_020", "abn_021"]:
    r = D.loc[fn]
    print(f"{fn} 충격 {int(r.n_imp):2d} | quiet {r.quiet:.2f} ({pct('quiet', r.quiet):3.0f}%) · gap {r.max_gap:.2f}s ({pct('max_gap', r.max_gap):3.0f}%) · iso {r.iso:.2f} ({pct('iso', r.iso):3.0f}%) · ex_frac {r.rms_ex_frac:.2f} ({pct('rms_ex_frac', r.rms_ex_frac):3.0f}%) · peak/med {r.peak_med:5.1f} · eff {r.eff_res:.2f} ({pct('eff_res', r.eff_res):3.0f}%)")
print("\n=== 집단 요약 (중앙값) ===")
print(D.groupby("y")[cols].median().round(2).to_string())
few = D[(D.y == 1) & (D.n_imp <= 3)]
print(f"\n충격 ≤ 3개인 고장 {len(few)}개: " + ", ".join(f"{i}(충격 {int(r.n_imp)}, iso {r.iso:.2f}, gap {r.max_gap:.2f}s)" for i, r in few.iterrows()))
print("충격 ≤ 3개인 정상 비율: %.0f%%" % ((D.loc[nor, 'n_imp'] <= 3).mean() * 100))
