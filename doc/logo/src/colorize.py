import numpy as np, re, subprocess, sys, json
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
# Colours the cleaned Onyxia line drawing. Usage: python3 colorize.py [BG|none]
# Regions/colours are in regions.json (seed coords in 1000-px space of the drawing).
ink=np.array(Image.open('onyxia-lines.png').convert('L'))<128; H,W=ink.shape
cx0,cy0,r0=0.5*W,0.53*H,0.465*W
yy,xx=np.mgrid[:H,:W]; circ=(xx-cx0)**2+(yy-cy0)**2<=r0*r0
# manual seams (1000-scale coords) added to ink for region finding only
seams=[[(370,162),(400,150),(440,147),(480,157)],[(748,152),(772,140),(798,138)]]
seamimg=Image.new('L',(W,H),0); d=ImageDraw.Draw(seamimg)
for s in seams: d.line([(x*4,y*4) for x,y in s], fill=255, width=16)
inkS=ink|(np.array(seamimg)>0)
dist=ndi.distance_transform_edt(~inkS)
cfg=json.load(open('regions.json'))   # name -> {seed:[x,y], rad, color}
labs={}
for rad in sorted({v['rad'] for v in cfg.values() if 'rad' in v}):
    labs[rad]=ndi.label((dist>rad)&circ)[0]
cores={}
for name,v in cfg.items():
    if 'poly' in v:
        pm=Image.new('L',(W,H),0); ImageDraw.Draw(pm).polygon([(x*4,y*4) for x,y in v['poly']],fill=255)
        cores[name]=(np.array(pm)>0)&~ink&circ; print(f'{name:8s} poly'); continue
    x,y=v['seed']; lab=labs[v['rad']]; l=lab[y*4,x*4]
    if l==0:
        ys,xs=np.where(lab[y*4-60:y*4+60, x*4-60:x*4+60]>0)
        if len(ys)==0: print('SEED ON INK:',name); continue
        k=np.argmin((ys-60)**2+(xs-60)**2); l=lab[y*4-60+ys[k], x*4-60+xs[k]]; print('moved seed',name)
    cores[name]=lab==l
    print(f'{name:8s} area {cores[name].sum()//1000}k')
allcore=np.zeros_like(ink); 
for m in cores.values(): allcore|=m
paint={}
for name,m in cores.items():
    others=allcore&~m
    grown=ndi.binary_dilation(m, iterations=cfg[name].get('rad',14)+8, mask=~others)
    paint[name]=grown
# fill small enclosed unpainted islands with neighbouring colour
painted=np.zeros_like(ink)
for m in paint.values(): painted|=m
free=(~painted)&(~ink)&circ
lab,n=ndi.label(free); sizes=ndi.sum(free,lab,range(1,n+1))
edge=np.zeros_like(ink); edge[0,:]=edge[-1,:]=edge[:,0]=edge[:,-1]=True
edge|=~ndi.binary_erosion(circ,iterations=3)&circ
names=list(paint)
objs=ndi.find_objects(lab)
for i in range(1,n+1):
    if sizes[i-1]>80000: continue
    sl=objs[i-1]; sl=tuple(slice(max(0,s.start-50),s.stop+50) for s in sl)
    m=lab[sl]==i
    if (edge[sl]&m).any(): continue
    ring=ndi.binary_dilation(m,iterations=40)&painted[sl]
    if not ring.any(): continue
    best=max(names,key=lambda k:(paint[k][sl]&ring).sum())
    paint[best][sl]|=ndi.binary_dilation(m,iterations=10,mask=~painted[sl]|m)
# raster preview
BG=sys.argv[1] if len(sys.argv)>1 else '#fffdeb'
def hx(c): return tuple(int(c[i:i+2],16) for i in (1,3,5))
img=np.zeros((H,W,4),np.uint8)
if BG!='none': img[circ]=(*hx(BG),255)
for name,m in paint.items(): img[m]=(*hx(cfg[name]['color']),255)
INK='#0e0a2e'; img[ink&circ]=(*hx(INK),255)
Image.fromarray(img).resize((1000,904),Image.LANCZOS).save('color_prev.png')
# vector: potrace each colour mask
cx,cy,r=0.5*W,0.53*H,0.465*W
paths=[]
def trace(mask):
    Image.fromarray((~mask*255).astype(np.uint8)).save('_m.pbm')
    s=subprocess.run(['potrace','_m.pbm','-s','-t','20','-a','1.3','-O','0.8','-o','-'],capture_output=True,text=True).stdout
    g=re.search(r'<g transform="([^"]+)"[^>]*>(.*)</g>',s,re.S)
    return g.group(1), "".join(re.findall(r'<path[^>]*/>',g.group(2),re.S))
bycolor={}
for name,m in paint.items(): bycolor.setdefault(cfg[name]['color'],np.zeros_like(ink)); bycolor[cfg[name]['color']]|=m
layers=[]
for col,m in bycolor.items():
    tr,p=trace(m); layers.append(f'<g fill="{col}" stroke="none" transform="{tr}">{p}</g>')
tr,p=trace(ink)
bgel='' if BG=='none' else f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="{BG}"/>'
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{cx-r:.0f} {cy-r:.0f} {2*r:.0f} {2*r:.0f}" width="512" height="512">
<clipPath id="c"><circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}"/></clipPath>
<g clip-path="url(#c)">{bgel}
{chr(10).join(layers)}
<g fill="{INK}" stroke="none" transform="{tr}">{p}</g>
</g></svg>
'''
open('onyxia-mark-color.svg','w').write(svg)
