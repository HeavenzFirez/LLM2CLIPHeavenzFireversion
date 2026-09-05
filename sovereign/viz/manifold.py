"""Generate the Sovereign Hyperdimensional Manifold (manifold.html).

This embeds the interactive 4D tesseract / lattice / cloud engine, but sources
its node payloads from real Sovereign pipeline output: base-9216 radix averages,
gyroid mesh statistics, and sampled lattice points. The result is a single,
self-contained, offline HTML file — open it in any browser, no server required.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List

from .. import gyroid as gy
from ..radix9216 import to_radix_9216


def build_node_payloads(averages: Dict[str, float], mesh_stats: Dict,
                        resolution: int = 8) -> List[Dict]:
    """Build inspectable node payloads from real pipeline data.

    Each node carries a verifiable payload (radix-encoded averages, mesh
    vertex/triangle counts, and a gyroid sample point) so the inspector panel
    shows genuine Sovereign state rather than placeholder text.
    """
    radix = {k: to_radix_9216(int(round(v * 1000))) for k, v in averages.items()}
    points = gy.near_surface_points(resolution, tolerance=0.20)
    payloads = []
    for i, (x, y, z, val) in enumerate(points[:16]):
        payloads.append({
            "id": "NODE_{0:02d}".format(i),
            "label": "GYROID_VERTEX_{0}".format(i),
            "payload": (
                "[VECTOR_BUS_{0}] State: Active\n"
                "Gyroid point: ({1:.4f}, {2:.4f}, {3:.4f})\n"
                "Implicit value: {4:.5f}\n"
                "Mesh vertices: {5}  triangles: {6}\n"
                "Radix averages: {7}"
            ).format(i, x, y, z, val,
                     mesh_stats.get("vertices", 0),
                     mesh_stats.get("triangles", 0),
                     radix),
        })
    # Pad to 16 nodes if the surface yielded fewer points.
    while len(payloads) < 16:
        i = len(payloads)
        payloads.append({
            "id": "NODE_{0:02d}".format(i),
            "label": "CONTINUUM_SLOT_{0}".format(i),
            "payload": "[VECTOR_BUS_{0}] State: Standby\nRadix averages: {1}".format(i, radix),
        })
    return payloads[:16]


def render_manifold_html(node_payloads: List[Dict],
                         title: str = "SOVEREIGN :: HYPERDIMENSIONAL MANIFOLD") -> str:
    """Return a complete self-contained manifold.html document string."""
    data_json = json.dumps(node_payloads)
    return MANIFOLD_TEMPLATE.replace("__NODES__", data_json).replace("__TITLE__", title)


def write_manifold_html(node_payloads: List[Dict], path: str,
                        title: str = "SOVEREIGN :: HYPERDIMENSIONAL MANIFOLD") -> str:
    """Write the interactive manifold viewer to path and return the path."""
    html = render_manifold_html(node_payloads, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def build_from_pipeline(averages: Dict[str, float], mesh_stats: Dict,
                        resolution: int = 8) -> List[Dict]:
    """Convenience: build node payloads directly from a pipeline summary."""
    return build_node_payloads(averages, mesh_stats, resolution)


MANIFOLD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{--bg:#030712;--panel:rgba(15,23,42,.75);--border:rgba(56,189,248,.2);
--cyan:#38bdf8;--magenta:#f43f5e;--amber:#fbbf24;--txt:#f8fafc;--mut:#94a3b8;
--mono:'JetBrains Mono','Fira Code','Courier New',monospace}
*{box-sizing:border-box;margin:0;padding:0;user-select:none}
body,html{width:100%;height:100%;overflow:hidden;background:var(--bg);
font-family:var(--mono);color:var(--txt)}
#c{position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab}
.glass{background:var(--panel);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
border:1px solid var(--border);border-radius:8px;box-shadow:0 8px 32px rgba(0,0,0,.5);
position:absolute;z-index:10;padding:16px}
header{top:20px;left:20px;display:flex;align-items:center;gap:16px}
h1{font-size:1.1rem;letter-spacing:2px;text-transform:uppercase;color:var(--cyan);
text-shadow:0 0 10px rgba(56,189,248,.5)}
.badge{font-size:.7rem;padding:3px 8px;border-radius:4px;background:rgba(56,189,248,.1);
border:1px solid var(--cyan);color:var(--cyan)}
#ctl{bottom:20px;left:20px;width:320px;display:flex;flex-direction:column;gap:12px}
.cg{display:flex;flex-direction:column;gap:6px}
.cg label{font-size:.75rem;color:var(--mut);display:flex;justify-content:space-between}
input[type=range]{appearance:none;width:100%;height:4px;background:rgba(255,255,255,.1);
border-radius:2px;outline:none}
input[type=range]::-webkit-slider-thumb{appearance:none;width:14px;height:14px;
border-radius:50%;background:var(--cyan);cursor:pointer;box-shadow:0 0 8px var(--cyan)}
.bg3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
button{background:rgba(255,255,255,.05);border:1px solid var(--border);color:var(--txt);
padding:8px;border-radius:4px;font-family:var(--mono);font-size:.75rem;cursor:pointer;
transition:all .2s}
button:hover{background:rgba(56,189,248,.2);border-color:var(--cyan);box-shadow:0 0 10px rgba(56,189,248,.3)}
button.active{background:var(--cyan);color:#000;font-weight:bold}
#insp{top:20px;right:20px;width:320px;max-height:calc(100vh - 40px);overflow-y:auto}
.nt{font-size:.9rem;color:var(--amber);margin-bottom:8px;border-bottom:1px solid var(--border);padding-bottom:4px}
.nd{font-size:.75rem;color:var(--mut);line-height:1.4;white-space:pre-wrap;word-break:break-all}
#atog{position:absolute;bottom:20px;right:20px;z-index:10}
</style>
</head>
<body>
<canvas id="c"></canvas>
<header class="glass"><h1>Sovereign Engine</h1><span class="badge" id="smode">4D HYPERCUBE</span></header>
<div id="ctl" class="glass">
<div class="cg"><label>4D ROTATION SPEED (XW/YW) <span id="sv">0.005</span></label>
<input type="range" id="rs" min="0" max="0.03" step="0.001" value="0.005"></div>
<div class="cg"><label>PROJECTION DISTANCE (W-DEPTH) <span id="wv">2.5</span></label>
<input type="range" id="wd" min="1.2" max="5.0" step="0.1" value="2.5"></div>
<div class="cg"><label>GEOMETRY MODE</label>
<div class="bg3"><button id="bt" class="active">4D CUBE</button>
<button id="bl">LATTICE</button><button id="bc">CLOUD</button></div></div>
</div>
<div id="insp" class="glass">
<div class="nt" id="nt">NO NODE SELECTED</div>
<div class="nd" id="nd">Click any hyper-dimensional node to extract payload memory from the local continuum vector space. Node data is sourced live from the Sovereign pipeline (base-9216 radix averages + gyroid lattice).</div>
</div>
<button id="atog" class="glass">AUDIO: OFF</button>
<script>
const NODES=__NODES__;
class AudioEngine{constructor(){this.ctx=null;this.muted=true;
this.notes=[130.81,146.83,164.81,196,220,261.63,293.66,329.63]}
init(){if(this.ctx)return;const A=window.AudioContext||window.webkitAudioContext;this.ctx=new A();
this.flt=this.ctx.createBiquadFilter();this.flt.type='lowpass';this.flt.frequency.value=400;
this.gain=this.ctx.createGain();this.gain.gain.value=0.15;
this.flt.connect(this.gain);this.gain.connect(this.ctx.destination);
this.drone=this.ctx.createOscillator();this.drone.type='sawtooth';this.drone.frequency.value=65.41;
const dg=this.ctx.createGain();dg.gain.value=0.08;
this.drone.connect(this.flt);this.drone.start()}
toggle(){if(!this.ctx)this.init();if(this.ctx.state==='suspended')this.ctx.resume();
this.muted=!this.muted;
this.gain.gain.setTargetAtTime(this.muted?0:0.15,this.ctx.currentTime,0.1);
return !this.muted}
pulse(i){if(this.muted||!this.ctx)return;const o=this.ctx.createOscillator(),g=this.ctx.createGain();
o.type='sine';o.frequency.value=this.notes[i%this.notes.length];
g.gain.setValueAtTime(0.2,this.ctx.currentTime);
g.gain.exponentialRampToValueAtTime(0.0001,this.ctx.currentTime+1.2);
o.connect(this.flt);o.connect(g);g.connect(this.ctx.destination);
o.start();o.stop(this.ctx.currentTime+1.2)}
updateFilter(v){if(this.flt)this.flt.frequency.setTargetAtTime(200+v*800,this.ctx.currentTime,0.1)}}
const audio=new AudioEngine();
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let W,H;function resize(){W=cv.width=innerWidth;H=cv.height=innerHeight}
addEventListener('resize',resize);resize();
let rotSpeed=0.005,wDist=2.5,mode='tesseract';
let aXY=0,aXZ=0,aXW=0,aYW=0,aZW=0;
// Tesseract: 16 verts of {-1,1}^4, each tagged with a real pipeline node.
const tv=[];for(let i=0;i<16;i++){const n=NODES[i]||NODES[0];
tv.push([(i&1)?1:-1,(i&2)?1:-1,(i&4)?1:-1,(i&8)?1:-1,n])}
const te=[];for(let i=0;i<16;i++)for(let j=i+1;j<16;j++){const d=i^j;
if((d&(d-1))===0)te.push([i,j])}
const cv2=[];for(let i=0;i<32;i++){const n=NODES[i%16];
cv2.push([(Math.random()-0.5)*3,(Math.random()-0.5)*3,(Math.random()-0.5)*3,(Math.random()-0.5)*3,n])}
let sel=null,proj=[];
function rot4(p,aXY,aXZ,aXW,aYW,aZW){let[x,y,z,w]=p;
let c=Math.cos(aXW),s=Math.sin(aXW),x1=x*c-w*s,w1=x*s+w*c;
c=Math.cos(aYW);s=Math.sin(aYW);let y1=y*c-w1*s,w2=y*s+w1*c;
c=Math.cos(aZW);s=Math.sin(aZW);let z1=z*c-w2*s,w3=z*s+w2*c;
c=Math.cos(aXY);s=Math.sin(aXY);let x2=x1*c-y1*s,y2=x1*s+y1*c;
return[x2,y2,z1,w3]}
function frame(){ctx.fillStyle='rgba(3,7,18,0.25)';ctx.fillRect(0,0,W,H);
aXW+=rotSpeed;aYW+=rotSpeed*0.7;aXY+=rotSpeed*0.3;
const verts=mode==='cloud'?cv2:tv;proj=[];
const p3=verts.map((v,idx)=>{const r=rot4(v,aXY,aXZ,aXW,aYW,aZW);
const wf=1/(wDist-r[3]);const x3=r[0]*wf,y3=r[1]*wf,z3=r[2]*wf;
const fov=350,scale=fov/(fov*0.002+z3+3);
const sx=W/2+x3*scale*100,sy=H/2+y3*scale*100;
const o={x:sx,y:sy,size:Math.max(3,(wf*8)*(scale*0.005)),wf:wf,data:v[4],index:idx};
proj.push(o);return o});
if(mode==='tesseract'){te.forEach(([i,j])=>{const a=p3[i],b=p3[j];
const al=Math.min(1,Math.max(0.1,(a.wf+b.wf)*0.3));
ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);
ctx.strokeStyle='rgba(56,189,248,'+al+')';ctx.lineWidth=1.5;ctx.stroke()})}
else if(mode==='lattice'){for(let i=0;i<p3.length;i++)for(let j=i+1;j<p3.length;j++){
const dx=p3[i].x-p3[j].x,dy=p3[i].y-p3[j].y,d=Math.sqrt(dx*dx+dy*dy);
if(d<120){ctx.beginPath();ctx.moveTo(p3[i].x,p3[i].y);ctx.lineTo(p3[j].x,p3[j].y);
ctx.strokeStyle='rgba(244,63,94,'+(1-d/120)+')';ctx.lineWidth=1;ctx.stroke()}}}
proj.forEach(n=>{const isSel=sel&&sel.data.id===n.data.id;
ctx.beginPath();ctx.arc(n.x,n.y,isSel?n.size*2:n.size,0,Math.PI*2);
if(isSel){ctx.fillStyle='#fbbf24';ctx.shadowColor='#fbbf24';ctx.shadowBlur=15}
else{ctx.fillStyle='#38bdf8';ctx.shadowColor='#38bdf8';ctx.shadowBlur=8}
ctx.fill();ctx.shadowBlur=0;
ctx.fillStyle='rgba(248,250,252,0.5)';ctx.font='9px monospace';
ctx.fillText(n.data.id,n.x+10,n.y+3)});
requestAnimationFrame(frame)}
cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect();
const cx=e.clientX-r.left,cy=e.clientY-r.top;let cl=null,md=20;
proj.forEach(n=>{const dx=n.x-cx,dy=n.y-cy,d=Math.sqrt(dx*dx+dy*dy);
if(d<md){md=d;cl=n}});
if(cl){sel=cl;document.getElementById('nt').innerText=cl.data.label;
document.getElementById('nd').innerText=cl.data.payload;audio.pulse(cl.index)}});
document.getElementById('rs').addEventListener('input',e=>{rotSpeed=parseFloat(e.target.value);
document.getElementById('sv').innerText=rotSpeed.toFixed(3)});
document.getElementById('wd').addEventListener('input',e=>{wDist=parseFloat(e.target.value);
document.getElementById('wv').innerText=wDist.toFixed(1);audio.updateFilter(wDist/5)});
const bt=document.getElementById('bt'),bl=document.getElementById('bl'),bc=document.getElementById('bc'),sm=document.getElementById('smode');
function setMode(m,b,l){mode=m;[bt,bl,bc].forEach(x=>x.classList.remove('active'));b.classList.add('active');sm.innerText=l}
bt.addEventListener('click',()=>setMode('tesseract',bt,'4D HYPERCUBE'));
bl.addEventListener('click',()=>setMode('lattice',bl,'NEURAL LATTICE'));
bc.addEventListener('click',()=>setMode('cloud',bc,'QUANTUM CLOUD'));
const ab=document.getElementById('atog');
ab.addEventListener('click',()=>{const a=audio.toggle();ab.innerText=a?'AUDIO: ACTIVE':'AUDIO: OFF';
ab.style.borderColor=a?'#38bdf8':'rgba(56,189,248,0.2)'});
frame();
</script>
</body>
</html>
"""


__all__ = ["build_node_payloads", "render_manifold_html", "write_manifold_html", "build_from_pipeline"]
