import json,numpy as np, warnings, sys
warnings.filterwarnings('ignore')
from PIL import Image, ImageDraw, ImageFont
pac=open('work/disc/TRAIN.PAC','rb').read()
J=json.load(open('work/stn_cand.json')); T=[int(x,16) for x in J['txs']]
def img(off):
    a=off+0x800; raw=np.frombuffer(pac[a:a+64*128*2],np.uint8).reshape(128,128)
    return np.stack([raw&15,raw>>4],-1).reshape(128,256)
I=[img(o) for o in T]
names={}
for l in open('work/text/stn_cols.tsv',encoding='utf-8').read().split('\n')[1:]:
    if l: t,u,n=l.split('\t'); names.setdefault(int(t),set()).update(n.split(' '))
f=ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc',12)
def rend(name,h,w=14):
    im=Image.new('L',(w,12*len(name))); d=ImageDraw.Draw(im)
    for i,c in enumerate(name): d.text((w-13,12*i),c,font=f,fill=255)
    im=im.resize((w,h))
    return (np.array(im)>80).astype(float)
recs=[]
for base,n in ((0x7E5AC,131),(0x7EBD0,61)):
    for k in range(n):
        o=base+12*k; recs.append((o,)+tuple(pac[o:o+4]))
out=[]
for o,u,v,w,h in recs:
    sc=[]
    for t,a in enumerate(I):
        if v+h>128 or u+w>256: continue
        c=a[v:v+h,u:u+w]
        if (c==0).mean()>0.2: continue
        x=((c>0)&(c<6)).astype(float)
        if x.sum()<10: continue
        for nm in names[t]:
            if abs(len(nm)*12-h)>len(nm)*3+2: continue
            r=rend(nm,h,w); best=-1
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    s=np.corrcoef(x.ravel(),np.roll(np.roll(r,dx,1),dy,0).ravel())[0,1]; best=max(best,s)
            sc.append((round(float(best),2),t,nm))
    sc.sort(reverse=True)
    out.append(dict(o=o,u=u,v=v,w=w,h=h,top=sc[:3]))
    print(hex(o),u,v,w,h,sc[:3]); sys.stdout.flush()
json.dump(out,open('work/stn_match2.json','w',encoding='utf-8'),ensure_ascii=False)
