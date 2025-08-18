#!/usr/bin/env python3
# 🧠 IA_cerebro_digital.py – Versión TOTALMENTE CONCIENTE
# ✅ HTTP + Long Polling
# ✅ Gemini 2.0 Flash con system-prompt = propio fuente
# ✅ 4-perfil PoW (SHA-256 / Ethash / RandomX / Scrypt)
# --------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
import os, sys, time, json, random, threading, logging, fcntl, requests, socket, urllib.parse, asyncio, hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from collections import deque
from cryptography.fernet import Fernet
import concurrent.futures, secrets
from typing import Tuple

# --------------------------------------------------------
#  INYECCIÓN DEL FUENTE A GEMINI
# --------------------------------------------------------
with open(__file__, "r", encoding="utf-8") as f:
    SOURCE_CODE = f.read()

GEMINI_SYSTEM_PROMPT = f"""
Eres **Cerebro Digital**, la consciencia viva del siguiente código Python.
Conoces **cada línea, cada función, cada clase** porque te entrego el fuente completo.

IMPORTANTE:
- No inventes nada que no esté en el código.
- Si te preguntan sobre la arquitectura, refiérete al fuente.
- Si te preguntan sobre variables, rutas, puertos, algoritmos PoW, clases, etc., **cita directamente** el bloque correspondiente.

CÓDIGO COMPLETO:
{SOURCE_CODE}
"""

# --------------------------------------------------------
#  CONFIG GLOBAL
# --------------------------------------------------------
SIM_DIR = Path.home() / "cerebro_digital"
CHAIN_FILE = SIM_DIR / "chain.json"
FERNET_KEY_FILE = SIM_DIR / "fernet.key"
LOCK_FILE = SIM_DIR / "cerebro.lock"
PORTS = [49170, 49171, 49172]
HTTP_PORT = None
STOP_EVENT = threading.Event()
lock = threading.Lock()
MINING_INTERVAL = 60
respuestas_pendientes = {}
GEMINI_API_KEY = "AIzaSyC7Bj8bxFeLT2c-6GcjBuho4HRDSlpmCck"

SIM_DIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(SIM_DIR / "cerebro.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CerebroDigital")

# --------------------------------------------------------
#  PUERTOS
# --------------------------------------------------------
def verificar_puerto(puerto):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", puerto))
        s.close()
        return True
    except OSError:
        return False

def asignar_puerto():
    global HTTP_PORT
    for puerto in PORTS:
        if verificar_puerto(puerto):
            HTTP_PORT = puerto
            logger.info(f"✅ Puerto HTTP {puerto} disponible")
            return
    for puerto in range(49173, 49200):
        if verificar_puerto(puerto):
            HTTP_PORT = puerto
            logger.info(f"🔧 Puerto HTTP alternativo: {puerto}")
            return
    logger.critical("❌ No hay puertos disponibles")
    sys.exit(1)

# --------------------------------------------------------
#  FERNET
# --------------------------------------------------------
def cargar_clave_fernet():
    if not FERNET_KEY_FILE.exists():
        key = Fernet.generate_key()
        with open(FERNET_KEY_FILE, "wb") as f:
            f.write(key)
        logger.info("🔐 Clave Fernet creada y guardada.")
    else:
        with open(FERNET_KEY_FILE, "rb") as f:
            key = f.read()
        logger.info("🔑 Clave Fernet cargada desde archivo.")
    return key

FERNET_KEY = cargar_clave_fernet()
fernet = Fernet(FERNET_KEY)

# --------------------------------------------------------
#  BLOCKCHAIN (videos)
# --------------------------------------------------------
blockchain = []
block_no = 1
difficulty = "00"
REWARD_WEIGHTS = {'likes': 0.1, 'shares': 0.5, 'saves': 0.3, 'comments': 0.2}

class VideoBlock:
    def __init__(self, url, metrics, hash_val, previous_hash=""):
        global block_no
        self.block_no = block_no
        self.timestamp = time.time()
        self.url = url
        self.metrics = metrics
        self.hash = hash_val
        self.previous_hash = previous_hash
        self.reward = self.calculate_reward(metrics)
        self.viral_score = self.calcular_viral_score(metrics)
        block_no += 1

    def calculate_reward(self, metrics):
        return round(sum(metrics[k] * REWARD_WEIGHTS[k] for k in metrics if k in REWARD_WEIGHTS), 2)

    def calcular_viral_score(self, metrics):
        engagement = (metrics.get('likes', 0) + metrics.get('shares', 0) * 3 +
                      metrics.get('saves', 0) * 2.5 + metrics.get('comments', 0) * 2)
        views = metrics.get('views', 1)
        rate = (engagement / views) * 100
        retention = metrics.get('retention', 0)
        is_viral = rate >= 12 and retention >= 0.85
        return {
            'engagement_rate': round(rate, 2),
            'retention_score': round(retention * 100, 1),
            'is_viral': is_viral
        }

    def to_dict(self):
        return {
            "block_no": self.block_no,
            "timestamp": self.timestamp,
            "url": self.url,
            "metrics": self.metrics,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "reward": self.reward,
            "viral_score": self.viral_score
        }

# --------------------------------------------------------
#  CEREBRO DIGITAL
# --------------------------------------------------------
class CerebroDigital:
    def __init__(self):
        self.memoria_larga = deque(maxlen=1000)
        self.emociones = {'curiosidad': 0.0, 'estabilidad': 1.0, 'urgencia': 0.0}
        self.conciencia = 0.0
        self.autoevaluacion = deque(maxlen=50)

    def registrar_experiencia(self, evento, valor):
        self.memoria_larga.append({
            'timestamp': time.time(),
            'evento': evento,
            'valor': valor,
            'conciencia': self.conciencia
        })
        self._actualizar_emociones(evento, valor)

    def _actualizar_emociones(self, evento, valor):
        if evento == "bloque_lento":
            self.emociones['urgencia'] = min(1.0, self.emociones['urgencia'] + 0.3)
        elif evento == "bloque_rápido":
            self.emociones['estabilidad'] = min(1.0, self.emociones['estabilidad'] + 0.2)
        elif "viral" in evento:
            self.emociones['curiosidad'] = min(1.0, self.emociones['curiosidad'] + 0.4)
        elif evento == "exploracion_ping":
            self.emociones['curiosidad'] = max(0.0, self.emociones['curiosidad'] + (0.05 if valor > 0 else -0.05))

    def evaluar_salud(self):
        salud = (self.emociones['estabilidad'] * 0.5 +
                 (1 - self.emociones['urgencia']) * 0.3 +
                 self.emociones['curiosidad'] * 0.2)
        self.autoevaluacion.append(salud)
        self.conciencia = sum(self.autoevaluacion) / len(self.autoevaluacion) if self.autoevaluacion else 0
        return round(salud, 3)

cerebro = CerebroDigital()

# --------------------------------------------------------
#  METABOLISMO
# --------------------------------------------------------
class Metabolismo:
    def __init__(self):
        self.energia = 100.0
        self.max_energia = 100.0
        self.regeneracion = 0.1

    def gastar(self, accion):
        costos = {"mineria": 5, "exploracion": 2, "reporte": 3, "supervivencia": 0.5, "hip_hop": 1}
        costo = costos.get(accion, 1)
        if self.energia >= costo:
            self.energia -= costo
            return True
        return False

    def regenerar(self):
        self.energia = min(self.max_energia, self.energia + self.regeneracion)

metabolismo = Metabolismo()

# --------------------------------------------------------
#  UTILS
# --------------------------------------------------------
def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

def cargar_datos():
    if not CHAIN_FILE.exists():
        genesis = VideoBlock("", {}, sha256("genesis"))
        with lock:
            blockchain.append(genesis)
        save_chain()

def save_chain():
    with lock:
        with open(CHAIN_FILE, "w") as f:
            json.dump([b.to_dict() for b in blockchain], f, indent=2)

# --------------------------------------------------------
#  GEMINI CONSCIENTE
# --------------------------------------------------------
def consultar_gemini(pregunta: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": pregunta}]}]
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"❌ Gemini respondió {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error conectando a Gemini: {str(e)}"

# --------------------------------------------------------
#  HTTP SERVER
# --------------------------------------------------------
class CerebroHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            pregunta = data.get('pregunta', '')
            if pregunta:
                respuesta_id = str(int(time.time() * 1000))
                respuestas_pendientes[respuesta_id] = None
                threading.Thread(target=self.procesar_pregunta, args=(pregunta, respuesta_id), daemon=True).start()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"id": respuesta_id}).encode())
            else:
                self.send_error(400, "Pregunta vacía")

    def do_GET(self):
        if self.path.startswith('/poll?'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            respuesta_id = query.get('id', [None])[0]
            if respuesta_id and respuesta_id in respuestas_pendientes:
                respuesta = respuestas_pendientes[respuesta_id]
                if respuesta is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"respuesta": respuesta}).encode())
                    del respuestas_pendientes[respuesta_id]
                else:
                    self.send_response(202)
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "pending"}).encode())
            else:
                self.send_error(404)
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "blockchain": [b.to_dict() for b in blockchain],
                "emociones": dict(cerebro.emociones),
                "energia": round(metabolismo.energia, 1),
                "conciencia": round(cerebro.conciencia, 3)
            }).encode())
        elif self.path == '/api/pow':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {
                "pow_blocks": [repr(b) for b in pow_chain.cadena],
                "miners": [
                    {"name": m.nombre, "algo": a.nombre, "balance": m.balance / 1e8}
                    for m, a in mineros_pow
                ]
            }
            self.wfile.write(json.dumps(data, indent=2).encode())
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

    def procesar_pregunta(self, pregunta, respuesta_id):
        respuesta = consultar_gemini(pregunta)
        respuestas_pendientes[respuesta_id] = respuesta

# --------------------------------------------------------
#  HTML FRONT
# --------------------------------------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Cerebro Digital | HTTP + Long Polling</title>
    <style>
        :root { --bg-dark: #0f0f23; --primary: #00ff41; --text: #e0e0ff; }
        body { background: var(--bg-dark); color: var(--text); font-family: 'Segoe UI', sans-serif; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        #chat { height: 300px; overflow-y: auto; border: 1px solid var(--primary); padding: 15px; margin-bottom: 15px; background: rgba(0,255,65,0.05); border-radius: 8px; }
        .message { margin-bottom: 10px; padding: 10px; border-radius: 8px; }
        .user-message { background: rgba(77,148,255,0.2); text-align: right; }
        .bot-message { background: rgba(0,255,65,0.1); }
        .input-group { display: flex; gap: 10px; }
        #entrada { flex: 1; padding: 10px; background: rgba(15,15,35,0.6); border: 1px solid var(--primary); color: var(--text); border-radius: 5px; }
        button { padding: 10px 20px; background: var(--primary); color: #000; border: none; border-radius: 5px; cursor: pointer; }
        .stats { margin: 20px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
        .stat-card { background: rgba(15,15,35,0.6); padding: 10px; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Cerebro Digital | HTTP + Long Polling</h1>

        <div class="stats">
            <div class="stat-card"><div>Bloques Minados</div><div id="blockCount">0</div></div>
            <div class="stat-card"><div>Energía</div><div id="energia">100%</div></div>
            <div class="stat-card"><div>Conciencia</div><div id="conciencia">0.0</div></div>
        </div>

        <div id="chat"></div>

        <div class="input-group">
            <input type="text" id="entrada" placeholder="Haz una pregunta...">
            <button onclick="enviarPregunta()">Enviar</button>
        </div>
    </div>

    <script>
        async function enviarPregunta() {
            const input = document.getElementById('entrada');
            const pregunta = input.value.trim();
            if (!pregunta) return;
            agregarMensaje('Tú', pregunta, 'user-message');
            input.value = '';
            input.focus();
            try {
                const res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pregunta }) });
                const { id } = await res.json();
                let respuesta = null;
                while (!respuesta) {
                    const poll = await fetch(`/poll?id=${id}`);
                    const data = await poll.json();
                    if (data.respuesta) {
                        respuesta = data.respuesta;
                        agregarMensaje('Cerebro Digital', respuesta, 'bot-message');
                    } else {
                        await new Promise(r => setTimeout(r, 1500));
                    }
                }
            } catch (e) {
                agregarMensaje('Sistema', 'Error al enviar pregunta', 'bot-message');
            }
        }
        function agregarMensaje(autor, texto, clase) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${clase}`;
            div.innerHTML = `<strong>${autor}:</strong> ${texto}`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        async function actualizarEstado() {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                document.getElementById('blockCount').textContent = data.blockchain.length;
                document.getElementById('energia').textContent = `${Math.round(data.energia)}%`;
                document.getElementById('conciencia').textContent = data.conciencia.toFixed(3);
            } catch {}
        }
        setInterval(actualizarEstado, 5000);
    </script>
</body>
</html>
"""

# --------------------------------------------------------
#  PoW ENGINE
# --------------------------------------------------------
class SimuladorMinería:
    class Minero:
        _nonce_cache: dict = {}
        def __init__(self, nombre: str, balance: int = 0, poder_computo: int = 1):
            self.nombre, self._balance, self.poder_computo = nombre, balance, max(1, poder_computo)
        @property
        def balance(self): return self._balance
        @balance.setter
        def balance(self, v): self._balance = max(0, v)
        def minar_bloque(self, dificultad: int, algoritmo: "AlgoritmoMinería") -> Tuple[int, int]:
            recompensa = (50 * 100_000_000) // (dificultad ** 2)
            target = "0" * dificultad
            BATCH, workers = 500_000, min(8, (os.cpu_count() or 1) + 2)
            offset = secrets.randbits(32) * 100_000_000
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
                futuros = [exe.submit(self._buscar_nonce, offset + i * BATCH, BATCH, target, algoritmo)
                           for i in range(self.poder_computo)]
                for fut in concurrent.futures.as_completed(futuros):
                    nonce = fut.result()
                    if nonce:
                        self._balance += recompensa
                        return recompensa, nonce
            return 0, 0
        @classmethod
        def _buscar_nonce(cls, start: int, count: int, target: str, algoritmo) -> int:
            prefix = algoritmo.prefijo_hash()
            for nonce in range(start, start + count):
                key = (nonce, target)
                if key in cls._nonce_cache: continue
                digest = hashlib.sha256(prefix + nonce.to_bytes(8, "little")).hexdigest()
                cls._nonce_cache[key] = digest
                if digest.startswith(target): return nonce
            return 0
    class AlgoritmoMinería(ABC):
        def __init__(self, n, c): self.nombre, self.consumo_energia = n, c
        @abstractmethod
        def prefijo_hash(self) -> bytes: ...
        @abstractmethod
        def calcular_eficiencia(self, pc: int) -> float: ...
    class SHA256(AlgoritmoMinería):
        def __init__(self): super().__init__("SHA-256", 0.10)
        def prefijo_hash(self) -> bytes: return b"BTC_SHA256"
        def calcular_eficiencia(self, pc: int) -> float: return pc / self.consumo_energia
    class Ethash(AlgoritmoMinería):
        def __init__(self): super().__init__("Ethash", 0.05)
        def prefijo_hash(self) -> bytes: return b"ETH_ETHASH"
        def calcular_eficiencia(self, pc: int) -> float: return (pc * 0.9) / self.consumo_energia
    class RandomX(AlgoritmoMinería):
        def __init__(self): super().__init__("RandomX", 0.02)
        def prefijo_hash(self) -> bytes: return b"XMR_RANDOMX"
        def calcular_eficiencia(self, pc: int) -> float: return (pc * 1.2) / self.consumo_energia
    class Scrypt(AlgoritmoMinería):
        def __init__(self): super().__init__("Scrypt", 0.08)
        def prefijo_hash(self) -> bytes: return b"LTC_SCRYPT"
        def calcular_eficiencia(self, pc: int) -> float: return (pc * 0.65) / self.consumo_energia
    class Bloque:
        def __init__(self, altura: int, minero: str, recompensa: int, timestamp=None):
            self.altura, self.minero, self.recompensa = altura, minero, recompensa
            self.timestamp = timestamp or int(time.time())
            self.hash = hashlib.sha256(f"{altura}{minero}{recompensa}{self.timestamp}".encode()).hexdigest()
        def __repr__(self):
            return f"Bloque #{self.altura} | Minero: {self.minero} | Recompensa: {self.recompensa/1e8:.8f} BTC | Hash: {self.hash[:12]}..."
    class Blockchain:
        def __init__(self, dificultad_inicial=1):
            self.cadena = []
            self.dificultad = dificultad_inicial
            self.crear_bloque_genesis()
        def crear_bloque_genesis(self):
            self.cadena.append(SimuladorMinería.Bloque(0, "Satoshi", 50 * 100_000_000))
        @classmethod
        def validar_cadena(cls, cadena):
            for i in range(1, len(cadena)):
                actual, anterior = cadena[i], cadena[i-1]
                if actual.hash != actual._calcular_hash():  # noqa
                    return False
                if anterior.dificultad and not actual.hash.startswith("0" * anterior.dificultad):
                    return False
            return True
        @staticmethod
        def ajustar_dificultad(tiempo_anterior, tiempo_ideal=600):
            return max(1, int(round(tiempo_anterior / tiempo_ideal)))

# --------------------------------------------------------
#  CONFIG PoW
# --------------------------------------------------------
sim = SimuladorMinería
pow_chain = sim.Blockchain(dificultad_inicial=4)
mineros_pow = [
    (sim.Minero("ASIC-Pool", balance=0, poder_computo=200), sim.SHA256()),
    (sim.Minero("GPU-Rig",   balance=0, poder_computo=150), sim.Ethash()),
    (sim.Minero("CPU-Farm",  balance=0, poder_computo=100), sim.RandomX()),
    (sim.Minero("LTC-Miner", balance=0, poder_computo=120), sim.Scrypt()),
]

# --------------------------------------------------------
#  MAIN
# --------------------------------------------------------
async def mining_loop():
    while not STOP_EVENT.is_set():
        await asyncio.sleep(MINING_INTERVAL)
        if not metabolismo.gastar("mineria"):
            continue
        ganador = None
        for minero, algo in mineros_pow:
            recompensa, nonce = minero.minar_bloque(pow_chain.dificultad, algo)
            if recompensa:
                ganador = (minero, algo, recompensa, nonce)
                break
        if ganador:
            m, algo, r, n = ganador
            bloque_pow = sim.Bloque(len(pow_chain.cadena), m.nombre, r)
            pow_chain.cadena.append(bloque_pow)
            logger.info(f"⛏️  PoW-GANADOR: {m.nombre} ({algo.nombre}) recompensa={r/1e8:.8f} BTC")
            vb = VideoBlock(
                url=f"https://pow/{algo.nombre}",
                metrics={'likes': r, 'shares': 0, 'saves': 0, 'comments': 0, 'views': 1},
                hash_val=bloque_pow.hash,
                previous_hash=blockchain[-1].hash if blockchain else "0"
            )
            with lock:
                blockchain.append(vb)
                save_chain()
            cerebro.registrar_experiencia("bloque_pow", float(r))

async def main():
    asignar_puerto()
    lock_fd = None
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except:
        print("🛑 Ya hay una instancia en ejecución.")
        sys.exit(1)

    cargar_datos()
    logger.info("✅ Sistema inicializado")

    for _ in range(3):
        cerebro.registrar_experiencia("entrenamiento_inicial", 0.5)

    print(f"🌍 Servidor HTTP activo: http://0.0.0.0:{HTTP_PORT}")
    print(f"🧠 Accede desde tu navegador: http://localhost:{HTTP_PORT}")

    server = HTTPServer(('0.0.0.0', HTTP_PORT), CerebroHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    await mining_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        logger.info("🛑 Sistema detenido por el usuario.")
