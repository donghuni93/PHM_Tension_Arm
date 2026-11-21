from common import *
from joblib import Parallel, delayed
from src.impacts import detect_impacts, impact_rings_v2, highpass
P = log_to("d1a_impacts_log.txt")
X, y, names, Xv, nv = load_all()
XA = np.vstack([X, Xv]); NA = names + nv
def work(x):
    xh = highpass(x); imp = detect_impacts(x, _hp=xh); r = impact_rings_v2(x, imp=imp, _hp=xh)
    return dict(t=imp["t"], hx=imp["height_x"], hg=imp["height_g"], f=r[:,0] if len(r) else np.zeros(0), z=r[:,1] if len(r) else np.zeros(0), fit=r[:,2] if len(r) else np.zeros(0), idx=imp["idx"])
res = Parallel(n_jobs=4)(delayed(work)(x) for x in XA)
pickle.dump((NA, res), open(os.path.join(CACHE, "impacts.pkl"), "wb"))
tot = dict(normal=sum(len(res[i]["t"]) for i in range(150) if y[i]==0), val=sum(len(r["t"]) for r in res[150:]), fault=sum(len(res[i]["t"]) for i in range(150) if y[i]==1))
P("충격 합계", tot, "(기대 854/247/402)")
