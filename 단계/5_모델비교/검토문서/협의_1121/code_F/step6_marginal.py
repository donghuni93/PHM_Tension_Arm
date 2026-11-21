# F step6: step5_final.py의 정의(경계 선택·ring12·new14·standardize·purity)를 그대로 쓰고
# (1) 피처 묶음별 한계 기여(new14 기준에 하나씩 더함): quick_cv PR-AUC, 오경보 정상 11개 위험, 고장 5-NN/정상 5-NN 순도(폴드 밖)
# (2) val 분포 점검 (고장 쪽 · 정상 증거 피처가 val에서 어떻게 보이나)
# (3) 오경보 정상 11개가 t-SNE에서 어디에 놓이나 (그림 F6_FA_tsne.png)
import os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "step5_final.py"), encoding="utf-8").read()
cut = src.index("RANDOM = (fault.sum()")
pre = src[:cut].replace("result_F5", "result_F6_pre").replace("figs_F5", "figs_F6pre").replace("F5_diag", "F6pre_diag")
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step5_final.py")
exec(compile(pre, "step5_pre", "exec"))
RANDOM = (fault.sum() - 1) / (ntr - 1)
OUT6 = open(os.path.join(HERE, "result_F6.txt"), "w", encoding="utf-8")
def Q(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT6.write(s + "\n"); OUT6.flush()

# 피처 묶음 (Rg 열 번호: 0 lo_share 1 mid_share 2 body_share 3 hi_share 4 mid_n 5 hi_n 6 mid_fit 7 hi_fit 8 dev_tsa 9 top1 10 tone_prom 11 tone_w)
GRP = {"hi(고주파 충격)": [3, 5, 7], "mid(2-3.4k 유격 칸)": [1, 4, 6], "body(본체 몫)": [2], "lo": [0], "dev_tsa(대표 울림 이탈)": [8],
       "top1(한 사건 집중)": [9], "tone(8.3k 순음)": [10, 11]}
def build2(space, tr):
    base, add = space
    Rg = None
    if add:
        wa, wb, _ = sel_win(tr); bh = sel_hi(tr)[0]; Rg = ring_feats(wa, wb, bh)
    parts = [{"new14": NEW16, "ext_rep": Mrep[:, sel_cols(Mrep, tr, 0.08)], "A_in": np.hstack([E_all, AOLD]), "ext150": E_all}[base]]
    if add:
        cols = sorted(set(sum([GRP[g] for g in add], [])) if add != ("all",) else range(12))
        parts.append(Rg[:, cols])
    return np.hstack(parts)
SP = [("A_in", ()), ("ext150", ()), ("new14", ())] + [("new14", (g,)) for g in GRP] + [("new14", ("hi(고주파 충격)", "mid(2-3.4k 유격 칸)")), ("new14", ("hi(고주파 충격)", "tone(8.3k 순음)")),
      ("new14", ("hi(고주파 충격)", "mid(2-3.4k 유격 칸)", "tone(8.3k 순음)", "top1(한 사건 집중)")), ("new14", ("all",)), ("ext_rep", ("all",))]
def name(sp): return sp[0] + ("" if not sp[1] else "+" + "+".join(g.split("(")[0] for g in sp[1]))
class SpET(BaseEstimator):
    def __init__(self, sp=None): self.sp = sp
    def fit(self, Fi, yy):
        tr = Fi[:, 0].astype(int); self.M_ = build2(self.sp, tr)
        self.m_ = make_et(seed=0)().fit(self.M_[tr], yy); return self
    def predict_proba(self, Fi): return self.m_.predict_proba(self.M_[Fi[:, 0].astype(int)])
Fidx = np.arange(ntr)[:, None].astype(float)
grp83 = [files.index(f) for f in ("abn_002", "abn_006", "abn_010", "abn_029")]
grp24 = [files.index(f) for f in ("abn_014", "abn_017", "abn_005", "abn_009", "abn_016", "abn_021", "abn_022")]
Q("== 1) 한계 기여: new14에 묶음을 하나씩 더함. quick_cv(ET300, 5겹 x seed 5, 열 선택·경계는 폴드 안), 오경보 정상 11개(FA)의 OOF 점수 백분위(정상 중, 낮을수록 좋음), 위험 정상 = 고장 하위 10% 선보다 높은 정상 수 ==")
Q("공간 | 열 수 | PR-AUC ± sd | ROC-AUC | 8.3k군(002 006 010 029) 순위 | 유격·조용군(014 017 005 009 016 021 022) 순위 | FA 정상 점수 백분위 | 위험 정상 수 | 고장 5-NN 순도(폴드밖) | FA 정상 5-NN 순도(폴드밖)")
res6 = {}
for sp in SP:
    q = quick_cv(Fidx, y, np.arange(ntr), lambda sp=sp: SpET(sp), seeds=range(5))
    oof = q["oof"]; on = oof[:, nor]
    fa_pct = []
    for s_ in range(oof.shape[0]):
        pn = oof[s_, nor]
        for i in fa_idx: fa_pct.append(np.mean(pn <= oof[s_, i]))
    risk = np.mean([(oof[s_, nor] > np.percentile(oof[s_, fault], 10)).sum() for s_ in range(oof.shape[0])])
    mr = q["miss_rank"]; r83 = np.mean([mr[i] for i in grp83]); r24 = np.mean([mr[i] for i in grp24])
    pv, pf = [], []
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            Mf = build2(sp, tr); Zf = np.clip(standardize(Mf, tr), -10, 10)
            who = te[fault[te]]; wn = np.array([i for i in te if isFA[i]])
            if len(who): pv.append(purity(Zf, who))
            if len(wn): pf.append(npurity(Zf, wn))
    dim = build2(sp, idx_all).shape[1]
    res6[name(sp)] = dict(dim=dim, pr=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std(ddof=1)), roc=float(q["roc_auc"].mean()), r83=float(r83), r24=float(r24), fa=float(np.mean(fa_pct)), risk=float(risk), pur=float(np.mean(pv)), fapur=float(np.mean(pf)))
    r = res6[name(sp)]
    Q(f"{name(sp):44s} | {dim:3d} | {r['pr']:.3f} ± {r['pr_sd']:.3f} | {r['roc']:.3f} | {r['r83']:.3f} | {r['r24']:.3f} | {r['fa']:.3f} | {r['risk']:.1f} | {r['pur']:.3f} | {r['fapur']:.3f}")

# ---------------- 2) val 분포 점검 ----------------
Q("\n== 2) val 분포 점검 (경계는 train 150 전체 선택값: 중간창 [%.0f,%.0f), b_hi=%.0f) ==" % (WA_FULL, WB_FULL, BHI_FULL))
RF = RING_FULL
def cnt(mask_fn, label):
    a = mask_fn(RF[:ntr][fault]).sum(); b = mask_fn(RF[:ntr][nor]).sum(); c = mask_fn(RF[ntr:]).sum()
    Q(f"  {label}: 고장 {a}/30 · 정상 {b}/120 ({b/120:.3f}) · val {c}/40 ({c/40:.3f})")
cnt(lambda M: M[:, 2] >= 0.9, "본체 몫 >= 0.9 (정상 증거)")
cnt(lambda M: M[:, 2] <= 0.1, "본체 몫 <= 0.1 (정상 증거 없음)")
cnt(lambda M: M[:, 5] > 0, "고주파(>= b_hi) 충격 있음")
cnt(lambda M: M[:, 4] > 0, "중간창 충격 있음")
cnt(lambda M: M[:, 1] > 0.5, "중간창 충격 몫 > 0.5")
cnt(lambda M: (M[:, 10] > 1.0), "8.3k 돌출 > 1.0 dex")
cnt(lambda M: (M[:, 10] < 0.2), "8.3k 돌출 < 0.2 dex (봉우리 없음)")
nimp_f = nimp
cnt(lambda M: np.zeros(len(M), bool), "(자리표시)")
Q(f"  충격 0개 파일: 정상 {int((nimp[:ntr][nor]==0).sum())}/120 · val {int((nimp[ntr:]==0).sum())}/40 ; 충격 수 중앙: 정상 {np.median(nimp[:ntr][nor]):.0f} 고장 {np.median(nimp[:ntr][fault]):.0f} val {np.median(nimp[ntr:]):.0f}")
# val 충격의 울림 주파수 분포 (파일 단위가 아니라 충격 단위)
allf = {"정상": np.concatenate([FILE_F[i] for i in np.flatnonzero(nor)]), "고장": np.concatenate([FILE_F[i] for i in np.flatnonzero(fault)]), "val": np.concatenate([FILE_F[i] for i in range(ntr, N)])}
for k, v in allf.items():
    Q(f"  {k} 충격 {len(v)}개: < 3.4k {np.mean(v<3375):.2f} | 3.4-7.9k {np.mean((v>=3375)&(v<8000)):.2f} | >= 8k {np.mean(v>=8000):.2f}")
# 정상 증거 피처가 val에서 고장 쪽으로 미는가: val 파일별 본체 몫과 중간창 몫
vb = RF[ntr:, 2]; vm = RF[ntr:, 1]
Q("  val 파일별 (본체 몫 / 중간창 몫 / 고주파 충격 수 / 충격 수): " + " ".join(f"{files[ntr+i][-3:]}({vb[i]:.2f}/{vm[i]:.2f}/{int(np.expm1(RF[ntr+i,5])+.5)}/{int(nimp[ntr+i])})" for i in range(40)))

# ---------------- 3) 오경보 정상 11개 t-SNE ----------------
NOR_, ABN_, PUR_ = "#2a78d6", "#eb6834", "#7B1FA2"
panels = [("ext150", ("ext150", ())), ("A_in (ext150+예전 울림 2열)", ("A_in", ())), ("new14", ("new14", ())), ("new14 + ring12 (제안 후보)", ("new14", ("all",)))]
fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
tab = {}
for a, (ttl, sp) in zip(axs, panels):
    M = build2(sp, idx_all); Z = np.clip(standardize(M, idx_all), -10, 10)
    Em = tsne_best(Z, 15)
    a.scatter(*Em[:ntr][nor].T, s=14, color=NOR_, alpha=.45, lw=0, label="정상")
    a.scatter(*Em[:ntr][fault].T, s=28, color=ABN_, lw=0, label="고장")
    a.scatter(*Em[ntr:].T, s=26, color=PUR_, marker="^", alpha=.8, lw=0, label="val")
    a.scatter(*Em[fa_idx].T, s=70, facecolors="none", edgecolors="k", lw=1.2, label="오경보 정상 11")
    for i in fa_idx: a.annotate(files[i][-3:], Em[i], fontsize=7, xytext=(3, 3), textcoords="offset points")
    nb = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fa_idx], return_distance=False)[:, 1:]
    nbf = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fault_idx], return_distance=False)[:, 1:]
    a.set_title(f"{ttl}\n고장 5-NN {y[nbf].mean():.2f} · 오경보정상 정상5-NN {1-y[nb].mean():.2f} (perp 15)", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    tab[ttl] = {files[i][-3:]: float(1 - y[nb[k]].mean()) for k, i in enumerate(fa_idx)}
axs[0].legend(fontsize=7, loc="lower left")
fig.suptitle("t-SNE (라벨은 색칠에만, train 150 + val 40): 오경보를 자주 내는 정상 11개(검은 원)가 정상 무리 안에 있는가", fontsize=10, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(HERE, "figs_F5", "F6_FA_tsne.png"), dpi=80); plt.close(fig)
Q("\n== 3) 오경보 정상 11개의 정상 5-NN 순도(t-SNE 2차원, perp 15, 파일별) ==")
for k, v in tab.items(): Q(f"  {k}: " + " ".join(f"{f}:{p:.1f}" for f, p in v.items()) + f" | 평균 {np.mean(list(v.values())):.2f}")
json.dump(res6, open(os.path.join(HERE, "result_F6.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
Q("완료")
