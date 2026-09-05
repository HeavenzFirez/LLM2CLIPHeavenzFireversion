"""Self-contained interactive HTML visualizer for the Sovereign pipeline.

Emits a single .html file that renders the gyroid mesh / tesseract / data nodes
in an interactive 3D wireframe using pure JavaScript + Canvas2D. No external
CDN, no network fetch — the whole viewer is embedded, so it runs by opening the
file locally. This is the "multidimensional browser" payload, fully offline.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple


def _vertices_to_rows(vertices) -> List[List[float]]:
    return [[float(v[0]), float(v[1]), float(v[2])] for v in vertices]


def build_gyroid_payload(mesh, averages: Dict[str, float],
                         radix: Dict[str, str]) -> Dict:
    """Pack a Marching Cubes mesh + data summary into a JSON-serialisable dict."""
    return {
        "kind": "gyroid",
        "vertices": _vertices_to_rows(mesh.vertices),
        "triangles": [[int(a), int(b), int(c)] for (a, b, c) in mesh.triangles],
        "averages": averages,
        "radix_9216": radix,
    }


def build_tesseract_payload() -> Dict:
    """Build a 4D hypercube (tesseract) as 16 vertices + 32 edges.

    The tesseract vertices are the 16 points of {0,1}^4, scaled and centered.
    Edges connect vertices differing in exactly one coordinate. The viewer
    projects 4D -> 3D -> 2D with a time-varying rotation in the XW and YW
    planes so the hypercube appears to "turn inside out" — the classic
    tesseract animation.
    """
    verts = []
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                for w in (0, 1):
                    verts.append([float(x), float(y), float(z), float(w)])
    edges = []
    for i, a in enumerate(verts):
        for j, b in enumerate(verts):
            if j <= i:
                continue
            diff = sum(1 for k in range(4) if a[k] != b[k])
            if diff == 1:
                edges.append([i, j])
    return {"kind": "tesseract", "vertices": verts, "edges": edges}


def render_html(payload: Dict, title: str = "Sovereign Multidimensional Browser") -> str:
    """Return a complete self-contained HTML document string from ``payload``."""
    data_json = json.dumps(payload)
    return HTML_TEMPLATE.replace("__PAYLOAD__", data_json).replace("__TITLE__", title)


def write_html(payload: Dict, path: str, title: str = "Sovereign Multidimensional Browser") -> str:
    """Write the interactive viewer to ``path`` and return the path."""
    html = render_html(payload, title)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html,body{margin:0;height:100%;background:#05060a;color:#9ad;font-family:monospace;overflow:hidden}
  #c{display:block;width:100vw;height:100vh;cursor:grab}
  #c:active{cursor:grabbing}
  #hud{position:fixed;top:10px;left:12px;font-size:13px;line-height:1.5;
       background:rgba(8,10,18,.6);padding:10px 14px;border:1px solid #234;border-radius:8px;pointer-events:none}
  #hud b{color:#7ef}
  #ctl{position:fixed;bottom:12px;left:12px;right:12px;display:flex;gap:8px;flex-wrap:wrap}
  #ctl button{background:#0c1320;color:#9ad;border:1px solid #2a4660;padding:8px 12px;
       border-radius:6px;font-family:monospace;cursor:pointer}
  #ctl button:hover{background:#13243a;color:#7ef}
  #ctl .sp{flex:1}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"></div>
<div id="ctl">
  <button data-m="drag">Drag to rotate</button>
  <button data-m="wheel">Scroll to zoom</button>
  <span class="sp"></span>
  <button id="bwire">Wireframe</button>
  <button id="bspin">Spin</button>
  <button id="breset">Reset view</button>
</div>
<script>
const DATA = __PAYLOAD__;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
const hud = document.getElementById('hud');
let W=0,H=0;
function resize(){W=cv.width=innerWidth;H=cv.height=innerHeight;}
addEventListener('resize',resize);resize();

// View state
let yaw=0.6, pitch=0.4, zoom=1, spinOn=true, wire=true;
let dragging=false, lx=0, ly=0;

cv.addEventListener('mousedown',e=>{dragging=true;lx=e.clientX;ly=e.clientY;});
addEventListener('mouseup',()=>dragging=false);
addEventListener('mousemove',e=>{if(!dragging)return;
  yaw+=(e.clientX-lx)*0.01; pitch+=(e.clientY-ly)*0.01;
  pitch=Math.max(-1.5,Math.min(1.5,pitch)); lx=e.clientX; ly=e.clientY;});
cv.addEventListener('wheel',e=>{e.preventDefault(); zoom*=e.deltaY>0?0.92:1.08;},{passive:false});

document.getElementById('bspin').onclick=function(){spinOn=!spinOn;this.style.color=spinOn?'#7ef':''};
document.getElementById('bwire').onclick=function(){wire=!wire;this.style.color=wire?'#7ef':''};
document.getElementById('breset').onclick=function(){yaw=0.6;pitch=0.4;zoom=1;};

function rot3(p,y,pch){
  // rotate around Y then X
  let cy=Math.cos(y),sy=Math.sin(y);
  let x=p[0]*cy-p[2]*sy, z=p[0]*sy+p[2]*cy, y2=p[1];
  let cp=Math.cos(pch),sp=Math.sin(pch);
  let yy=y2*cp-z*sp; z=y2*sp+z*cp;
  return [x,yy,z];
}

let t=0;
function frame(){
  if(spinOn){yaw+=0.004; t+=0.012;}
  ctx.fillStyle='rgba(5,6,10,0.35)';
  ctx.fillRect(0,0,W,H);
  const cx=W/2, cy=H/2, sc=Math.min(W,H)*0.32*zoom;

  const kind=DATA.kind;
  let pts=[];
  let segs=[]; // [a,b,color,width]

  if(kind==='gyroid'){
    const vs=DATA.vertices, tris=DATA.triangles;
    // center the mesh
    let min=[1e9,1e9,1e9],max=[-1e9,-1e9,-1e9];
    for(const v of vs){for(let i=0;i<3;i++){min[i]=Math.min(min[i],v[i]);max[i]=Math.max(max[i],v[i]);}}
    let cen=[(min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2];
    let ext=Math.max(max[0]-min[0],max[1]-min[1],max[2]-min[2])||1;
    let s=2/ext;
    for(const v of vs){
      let p=[(v[0]-cen[0])*s,(v[1]-cen[1])*s,(v[2]-cen[2])*s];
      pts.push(rot3(p,yaw,pitch));
    }
    // draw triangle edges (wireframe) or filled translucent faces
    const cols=['#1d3a5c','#244f7a','#2c6499'];
    if(wire){
      for(const f of tris){
        for(let i=0;i<3;i++){segs.push([f[i],f[(i+1)%3],'#2a5680',0.5]);}
      }
    }else{
      // sort faces by average z for painter's algo
      let faces=tris.map(f=>({f,z:(pts[f[0]][2]+pts[f[1]][2]+pts[f[2]][2])/3}));
      faces.sort((a,b)=>a.z-b.z);
      for(let k=0;k<faces.length;k++){
        const f=faces[k].f;
        ctx.beginPath();
        for(let i=0;i<3;i++){const p=pts[f[i]];const sx=cx+p[0]*sc,sy2=cy-p[1]*sc;
          i?ctx.lineTo(sx,sy2):ctx.moveTo(sx,sy2);}
        ctx.closePath();
        ctx.fillStyle='rgba(40,90,140,'+(0.10+0.18*(k/faces.length))+')';
        ctx.fill();ctx.strokeStyle='#2a5680';ctx.lineWidth=0.4;ctx.stroke();
      }
    }
  } else if(kind==='tesseract'){
    // 4D verts, project 4D->3D with rotating XW,YW planes
    const vs=DATA.vertices, es=DATA.edges;
    const a=t, ca=Math.cos(a), sa=Math.sin(a), cb=Math.cos(a*0.7), sb=Math.sin(a*0.7);
    for(const v of vs){
      // rotate in XW then YW
      let x=v[0]-0.5, y=v[1]-0.5, z=v[2]-0.5, w=v[3]-0.5;
      let x2=x*ca - w*sa, w2=x*sa + w*ca;
      let y2=y*cb - w2*sb, w3=y*sb + w2*cb;
      // project w into z (4D->3D perspective)
      let dist=3;
      let pw=1/(dist - w3);
      let p3=[x2*pw*2, y2*pw*2, z*pw*2];
      pts.push(rot3(p3,yaw,pitch));
    }
    for(const e of es){segs.push([e[0],e[1],'#5af',1]);}
  }

  // draw segments
  ctx.lineWidth=1;
  for(const s of segs){
    const a=pts[s[0]], b=pts[s[1]];
    if(!a||!b) continue;
    // depth fade
    const dz=(a[2]+b[2])*0.5;
    const alpha=0.25+0.55*(1-(dz+1.5)/3);
    ctx.strokeStyle=s[2]; ctx.globalAlpha=Math.max(0.1,Math.min(1,alpha));
    ctx.lineWidth=s[3];
    ctx.beginPath();
    ctx.moveTo(cx+a[0]*sc, cy-a[1]*sc);
    ctx.lineTo(cx+b[0]*sc, cy-b[1]*sc);
    ctx.stroke();
  }
  ctx.globalAlpha=1;

  // HUD
  let h='<b>'+kind.toUpperCase()+'</b><br>';
  if(kind==='gyroid'){
    h+='vertices: <b>'+DATA.vertices.length+'</b> &nbsp; triangles: <b>'+DATA.triangles.length+'</b><br>';
    h+='averages:<br>';
    for(const k in (DATA.averages||{})) h+='&nbsp;'+k+': <b>'+DATA.averages[k].toFixed(3)+'</b> (radix '+DATA.radix_9216[k]+')<br>';
  }else{
    h+='vertices: <b>16</b> &nbsp; edges: <b>'+DATA.edges.length+'</b><br>4D projection (XW,YW rotation)';
  }
  hud.innerHTML=h;

  requestAnimationFrame(frame);
}
frame();
</script>
</body>
</html>
"""


__all__ = ["build_gyroid_payload", "build_tesseract_payload", "render_html", "write_html"]
