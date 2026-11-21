"""R 4f: body_frac 분포(0 비율), hf 울림 고장 수, val의 본체 증거 이동. 출력 out/R4f.txt"""
import os, sys, pickle
here = os.path.dirname(os.path.abspath(__file__)); root = here
while not os.path.isdir(os.path.join(root, "src")): root = os.path.dirname(root)
sys.path.insert(0, root); os.chdir(root)
import numpy as np
R = pickle.load(open(os.path.join(here, "out", "raw.pkl"), "rb"))
names, split, y, imps = R["names"], R["split"], R["y"], R["imps"]
sn = [s.replace("abn_", "A").replace("nor_", "N").replace("val_", "V") for s in names]
def bf(im):
    r = im["rings"]; k = len(im["height_x"])
    if not k: return -1.0, 0
    inb = (r[:, 0] >= 5500) & (r[:, 0] < 7750) & (r[:, 2] >= 0.5); return inb.mean(), int(inb.sum())
B = np.array([bf(im) for im in imps]); k = np.array([len(im["height_x"]) for im in imps])
nor = (split == "train") & (y == 0); fau = (split == "train") & (y == 1); val = split == "val"
o = open(os.path.join(here, "out", "R4f.txt"), "w", encoding="utf-8")
for nm_, m in [("정상", nor), ("고장", fau), ("val", val)]:
    o.write(f"{nm_}: n={m.sum()} 충격0개 {int((k[m]==0).sum())} body_frac==0(충격>=1) {int(((B[m,0]==0)&(k[m]>=1)).sum())} body_frac>=0.5 {int((B[m,0]>=.5).sum())} body_frac>=0.9 {int((B[m,0]>=.9).sum())} body_cnt>=1 {int((B[m,1]>=1).sum())} 충격 중앙 {np.median(k[m])}\n")
o.write("충격 개수별 body_frac>=0.5 비율: " + "; ".join(f"{nm_} 충격>=2: {np.mean(B[m&(k>=2),0]>=.5):.2f} (n={int((m&(k>=2)).sum())})" for nm_, m in [("정상", nor), ("고장", fau), ("val", val)]) + "\n")
hf = np.array([(im["rings"][:, 0] >= 9250).sum() if len(im["rings"]) else 0 for im in imps])
o.write("hf(>=9.25k) 울림 충격이 있는 파일: 고장 " + " ".join(sn[i] for i in np.flatnonzero(fau & (hf > 0))) + " | 정상 " + " ".join(sn[i] for i in np.flatnonzero(nor & (hf > 0))) + " | val " + " ".join(sn[i] for i in np.flatnonzero(val & (hf > 0))) + "\n")
o.write("val 충격 수 분포: " + str(np.bincount(np.minimum(k[val], 10))) + "  정상: " + str(np.bincount(np.minimum(k[nor], 10))) + "\n")
o.write("val body_frac 0 이고 충격>=1 파일: " + " ".join(sn[i] for i in np.flatnonzero(val & (B[:, 0] == 0) & (k >= 1))) + "\n")
o.close()
