"""catalog AI 浮窗 —— components.html iframe + CSS 精确定位"""
import sys, os, threading, time, socket

_AHERE = os.path.dirname(os.path.abspath(__file__))
_PATH1 = os.path.abspath(os.path.join(_AHERE, "..", "examples", "catalog_assistant"))
if not os.path.isdir(_PATH1):
    _PATH1 = os.path.abspath(os.path.join(_AHERE, "..", "..", "mlblack", "examples", "catalog_assistant"))
if _PATH1 not in sys.path:
    sys.path.insert(0, _PATH1)

PORT = 5001
_started = False

def _in_use(p):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.3)
        r = s.connect_ex(("127.0.0.1", p)); s.close(); return r == 0
    except: return False

def _start():
    global _started
    if _started or _in_use(PORT): _started = True; return
    try:
        import importlib.util
        sp = os.path.join(_PATH1, "server.py")
        spec = importlib.util.spec_from_file_location("_asrv", sp)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        t = threading.Thread(target=lambda: m.app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False), daemon=True)
        t.start()
        for _ in range(50):
            if _in_use(PORT): _started = True; return
            time.sleep(0.1)
    except Exception as e: print(f"[AI] {e}")

def render(st):
    _start()
    try:
        from catalog_data import _load_catalog
        total = len(_load_catalog())
    except: total = "全部"

    import streamlit.components.v1 as comps

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{{margin:0;padding:0;box-sizing:border-box}} body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:transparent;overflow:hidden}}
#btn{{position:fixed;right:8px;top:50%;transform:translateY(-50%);z-index:9999;width:44px;height:44px;border-radius:22px 0 0 22px;border:none;cursor:pointer;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;font-size:13px;font-weight:700;box-shadow:-2px 2px 12px rgba(102,126,234,0.35);transition:width .25s,opacity .25s;display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap;pointer-events:auto}}
#btn:hover{{width:52px}} #btn.off{{width:0;opacity:0;pointer-events:none}}
#pnl{{display:none;position:fixed;right:60px;top:50%;transform:translateY(-50%);z-index:9998;width:360px;height:480px;background:#fff;border-radius:12px;box-shadow:0 8px 32px rgba(0,0,0,0.15);flex-direction:column;overflow:hidden;pointer-events:auto;font-size:13px}}
#pnl.on{{display:flex}}
#hd{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:10px 14px;font-size:14px;font-weight:600;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}}
#bd{{flex:1;overflow-y:auto;padding:10px;background:#f5f7fa;line-height:1.55}}
.msg{{margin-bottom:10px;display:flex;gap:6px}}
.msg.usr{{flex-direction:row-reverse}}
.av{{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}}
.av.usr{{background:#667eea;color:#fff}} .av:not(.usr){{background:#e8eaf6;color:#667eea}}
.bl{{padding:7px 10px;border-radius:10px;max-width:250px;word-break:break-word}}
.msg.usr .bl{{background:#667eea;color:#fff;border-bottom-right-radius:3px}}
.msg:not(.usr) .bl{{background:#fff;border-bottom-left-radius:3px;box-shadow:0 1px 2px rgba(0,0,0,0.06)}}
#ft{{padding:8px 10px;border-top:1px solid #eee;display:flex;gap:6px;flex-shrink:0}}
#ft input{{flex:1;border:1px solid #ddd;border-radius:8px;padding:6px 10px;font-size:13px;outline:none}}
#ft button{{background:#667eea;color:#fff;border:none;border-radius:8px;padding:6px 14px;cursor:pointer;font-size:13px}}
#ft button:disabled{{opacity:0.5;cursor:default}}
</style></head><body>
<button id="btn" onclick="t()">AI</button>
<div id="pnl">
<div id="hd"><span>Catalog AI ({total})</span><span onclick="t()" style="cursor:pointer;font-size:18px">&times;</span></div>
<div id="bd"><div class="msg"><div class="av">AI</div><div class="bl">你好！想问什么直接说~<br>比如"多目标优化用什么adapter"</div></div></div>
<div id="ft"><input id="inp" placeholder="问我要用什么组件..." onkeydown="if(event.key==='Enter')s()"><button id="sb" onclick="s()">发送</button></div>
</div>
<script>
var L=!1,O=!1;
function t(){{O=!O;document.getElementById('pnl').className=O?'on':'';document.getElementById('btn').className=O?'off':'';if(O)document.getElementById('inp').focus()}}
function a(r,x){{var e=document.getElementById('bd'),d=document.createElement('div');d.className='msg '+(r=='user'?'usr':'');d.innerHTML='<div class="av '+(r=='user'?'usr':'')+'">'+(r=='user'?'Y':'AI')+'</div><div class="bl">'+x.replace(/\\n/g,'<br>')+'</div>';e.appendChild(d);e.scrollTop=e.scrollHeight}}
async function s(){{if(L)return;var i=document.getElementById('inp'),q=i.value.trim();if(!q)return;i.value='';a('user',q);L=!0;document.getElementById('sb').disabled=!0;try{{var r=await fetch('http://127.0.0.1:{PORT}/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{query:q}})}});var d=await r.json();a('bot',d.reply||'no reply')}}catch(e){{a('bot','Err: '+e.message)}}L=!1;document.getElementById('sb').disabled=!1}}
</script></body></html>"""

    # 父页 CSS：iframe 固定覆盖层，pointer-events:none 让透明区域穿透点击
    st.markdown("""
<style>
iframe[title*="streamlit_components"] {
  position: fixed !important; right: 0; top: 50% !important; transform: translateY(-50%) !important;
  z-index: 9996 !important; width: 420px !important; height: 520px !important;
  pointer-events: none !important; border: none !important;
}
</style>
<div id="mk"></div>
""", unsafe_allow_html=True)
    comps.html(html, height=520, scrolling=False)
