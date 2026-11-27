import pickle, sys
R=pickle.load(open("out/raw.pkl","rb"))
o=open("out/insp.txt","w",encoding="utf-8")
o.write(str(R.keys())+"\n"); im=R["imps"][0]
o.write(str({k:(getattr(v,'shape',None) or type(v)) for k,v in im.items()})+"\n"); o.write(str(list(R["T2"].columns))+"\n")
o.write(str(type(R["K"]))+str(R["K"].keys() if hasattr(R["K"],'keys') else getattr(R["K"],'shape',''))+"\n")
