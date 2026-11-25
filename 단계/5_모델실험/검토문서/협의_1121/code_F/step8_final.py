# F step8: (A) new14에 열 1개씩 / L1_S 제거 한계 기여 (B) 오경보 정상 11개 t-SNE 그림. 출력은 파일로만(콘솔 인코딩 문제 회피).
import os, sys
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "step5_final.py"), encoding="utf-8").read()
cut = src.index("RANDOM = (fault.sum()")
pre = src[:cut].replace("result_F5", "result_F8_pre").replace("figs_F5", "figs_F8pre").replace("F5_diag", "F8pre_diag")
pre = pre.replace("def P(*a):\n    s = \" \".join(str(x) for x in a); print(s, flush=True); OUT.write(s + \"\\n\"); OUT.flush()", "def P(*a):\n    s = \" \".join(str(x) for x in a); OUT.write(s + \"\\n\"); OUT.flush()")
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step5_final.py")
exec(compile(pre, "step5_pre", "exec"))
RANDOM = (fault.sum() - 1) / (ntr - 1)
OUT6 = open(os.path.join(HERE, "result_F8b.txt" if os.environ.get("SKIPA") else "result_F8.txt"), "w", encoding="utf-8")
from sklearn.manifold import TSNE
fault_idx = np.flatnonzero(fault)
def tsne_best(A, perp, nseed=3):
    best = None
    for s_ in range(nseed):
        t = TSNE(2, perplexity=perp, init="pca", random_state=s_, learning_rate="auto"); Em = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)
    return best[1]
def Q(*a):
    s = " ".join(str(x) for x in a); OUT6.write(s + "\n"); OUT6.flush()
iL1 = FINAL16.index("L1_S")
NEW_noS = np.delete(NEW16, iL1, axis=1)
def build3(sp, tr):
    if sp == "new14": return NEW16
    if sp == "new14-L1_S": return NEW_noS
    wa, wb, _ = sel_win(tr); bh = sel_hi(tr)[0]; Rg = ring_feats(wa, wb, bh)
    add = {"+hi_n": [5], "+hi_fit": [7], "+mid_n": [4], "+hi_n+mid_n": [5, 4], "+body": [2], "+tone2": [10, 11], "+hi_n+mid_n+body+tone2": [5, 4, 2, 10, 11]}
    if sp in add: return np.hstack([NEW16, Rg[:, add[sp]]])
    if sp.startswith("-L1_S"):
        k = sp[len("-L1_S"):]; return np.hstack([NEW_noS, Rg[:, add[k]]])
    raise KeyError(sp)
SPS = ["new14", "new14-L1_S", "+hi_n", "+hi_fit", "+mid_n", "+hi_n+mid_n", "+body", "+tone2", "+hi_n+mid_n+body+tone2"]
class SpET(BaseEstimator):
    def __init__(self, sp=None): self.sp = sp
    def fit(self, Fi, yy):
        tr = Fi[:, 0].astype(int); self.M_ = build3(self.sp, tr)
        self.m_ = make_et(seed=0)().fit(self.M_[tr], yy); return self
    def predict_proba(self, Fi): return self.m_.predict_proba(self.M_[Fi[:, 0].astype(int)])
Fidx = np.arange(ntr)[:, None].astype(float)
grp83 = [files.index(f) for f in ("abn_002", "abn_006", "abn_010", "abn_029")]
grp24 = [files.index(f) for f in ("abn_014", "abn_017", "abn_005", "abn_009", "abn_016", "abn_021", "abn_022")]
Q("== A) new14 기준 한계 기여 (열 1개씩 / L1_S 제거). quick_cv ET300 5겹 x seed 5")
Q("공간 | 열 수 | PR-AUC ± sd | ROC | 8.3k군 순위 | 유격·조용군 순위 | FA 정상 점수 백분위 | 위험 정상 수 | 고장 5-NN 순도(폴드밖) | FA 정상 5-NN 순도(폴드밖)")
for sp in ([] if os.environ.get("SKIPA") else SPS):
    q = quick_cv(Fidx, y, np.arange(ntr), lambda sp=sp: SpET(sp), seeds=range(5))
    oof = q["oof"]; fa_pct = []
    for s_ in range(oof.shape[0]):
        pn = oof[s_, nor]
        for i in fa_idx: fa_pct.append(np.mean(pn <= oof[s_, i]))
    risk = np.mean([(oof[s_, nor] > np.percentile(oof[s_, fault], 10)).sum() for s_ in range(oof.shape[0])])
    mr = q["miss_rank"]; r83 = np.mean([mr[i] for i in grp83]); r24 = np.mean([mr[i] for i in grp24])
    pv, pf = [], []
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            Zf = np.clip(standardize(build3(sp, tr), tr), -10, 10)
            who = te[fault[te]]; wn = np.array([i for i in te if isFA[i]])
            if len(who): pv.append(purity(Zf, who))
            if len(wn): pf.append(npurity(Zf, wn))
    Q(f"{sp:26s} | {build3(sp, idx_all).shape[1]:3d} | {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std(ddof=1):.3f} | {q['roc_auc'].mean():.3f} | {r83:.3f} | {r24:.3f} | {np.mean(fa_pct):.3f} | {risk:.1f} | {np.mean(pv):.3f} | {np.mean(pf):.3f}")

# ---------------- B) 오경보 정상 11개 t-SNE ----------------
NOR_, ABN_, PUR_ = "#2a78d6", "#eb6834", "#7B1FA2"
def spaceM(name):
    if name == "ext150": return E_all
    if name == "A_in": return np.hstack([E_all, AOLD])
    if name == "new14": return NEW16
    return build3(name, idx_all)
panels = [("ext150", "ext150"), ("A 입력 (ext150 + 예전 울림 2열)", "A_in"), ("new14 (2차 설계 14열)", "new14"), ("new14 + 고주파·유격창·본체·순음 5열", "+hi_n+mid_n+body+tone2")]
fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
tab = {}
for a, (ttl, sp) in zip(axs, panels):
    Z = np.clip(standardize(spaceM(sp), idx_all), -10, 10)
    purs = []; fas = []
    for perp in (5, 15, 30):
        Em = tsne_best(Z, perp)
        nbf = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fault_idx], return_distance=False)[:, 1:]
        nb = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fa_idx], return_distance=False)[:, 1:]
        purs.append(float(y[nbf].mean())); fas.append(float(1 - y[nb].mean()))
        if perp == 15: Em15, nb15 = Em, nb
    Em = Em15
    a.scatter(*Em[:ntr][nor].T, s=14, color=NOR_, alpha=.45, lw=0, label="정상")
    a.scatter(*Em[:ntr][fault].T, s=28, color=ABN_, lw=0, label="고장")
    a.scatter(*Em[ntr:].T, s=26, color=PUR_, marker="^", alpha=.8, lw=0, label="val")
    a.scatter(*Em[fa_idx].T, s=70, facecolors="none", edgecolors="k", lw=1.2, label="오경보 정상 11")
    for i in fa_idx: a.annotate(files[i][-3:], Em[i], fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_title(f"{ttl}\n고장 5-NN {np.mean(purs):.2f} · 오경보정상의 정상 5-NN {np.mean(fas):.2f} (perp 5·15·30 평균)", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    tab[ttl] = dict(pur=purs, fa=fas, per_file={files[i][-3:]: float(1 - y[nb15[k]].mean()) for k, i in enumerate(fa_idx)})
axs[0].legend(fontsize=7, loc="lower left")
fig.suptitle("t-SNE (라벨은 색칠에만, train 150 + val 40, perplexity 15 그림): 오경보를 자주 내는 정상 11개(검은 원)가 정상 무리 안에 있는가", fontsize=10, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(HERE, "figs_F5", "F8_FA_tsne.png"), dpi=80); plt.close(fig)
Q("\n== B) t-SNE (perp 5 · 15 · 30) 고장 5-NN 순도 / 오경보 정상 11개의 정상 5-NN 순도")
for k, v in tab.items():
    Q(f"  {k}: 고장 {[round(x,3) for x in v['pur']]} 평균 {np.mean(v['pur']):.3f} | FA정상 {[round(x,3) for x in v['fa']]} 평균 {np.mean(v['fa']):.3f}")
    Q("     파일별(perp15): " + " ".join(f"{f}:{p:.1f}" for f, p in v["per_file"].items()))
Q("완료")
