import os
import sys
import time
import json
import random
import signal
import asyncio
import hashlib
import functools
import websockets
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime
from collections import deque
from cryptography.fernet import Fernet
import requests

# ================= CONFIGURACIÓN GLOBAL =================
SIM_DIR = Path.home() / "simulador"
CHAIN_FILE = SIM_DIR / "chain.json"
FERNET_KEY_FILE = SIM_DIR / "fernet.key"
WS_PORT = 8065        # WebSocket principal
CONTROL_WS_PORT = 8066  # Control remoto
HTTP_PORT = 8080      # Flask
WS_TOKEN = "mi-super-token-2024"
CLAVE_REPORTE = "clave-secreta-ultra"
STOP_EVENT = threading.Event()
clients = set()
control_clients = set()
lock = threading.Lock()
MINING_INTERVAL = 5

# Cargar o generar clave Fernet persistente
def cargar_clave_fernet():
    if not FERNET_KEY_FILE.exists():
        key = Fernet.generate_key()
        with open(FERNET_KEY_FILE, "wb") as f:
            f.write(key)
        print("🔐 Clave Fernet creada y guardada.")
    else:
        with open(FERNET_KEY_FILE, "rb") as f:
            key = f.read()
        print("🔑 Clave Fernet cargada desde archivo.")
    return key

FERNET_KEY = cargar_clave_fernet()
fernet = Fernet(FERNET_KEY)

# ================= BLOCKCHAIN CORE =================
blockchain = []
block_no = 1
difficulty = "00"
REWARD_WEIGHTS = {'likes': 0.1, 'shares': 0.5, 'saves': 0.3, 'comments': 0.2}

class VideoBlock:
    def __init__(self, url, metrics, hash_val, previous_hash="", investment=0.0):
        global block_no
        self.block_no = block_no
        self.timestamp = time.time()
        self.url = url
        self.metrics = metrics
        self.hash = hash_val
        self.previous_hash = previous_hash
        self.investment = investment
        self.reward = self.calculate_reward(metrics)
        self.synapses = []
        self.presynapses = []
        self.viral_score = self.calcular_viral_score(metrics)

    def calculate_reward(self, metrics):
        return round(sum(metrics[k] * REWARD_WEIGHTS[k] for k in metrics if k in REWARD_WEIGHTS), 2)

    def calcular_viral_score(self, metrics):
        engagement = (metrics.get('likes', 0) + 
                      metrics.get('shares', 0) * 3 + 
                      metrics.get('saves', 0) * 2.5 + 
                      metrics.get('comments', 0) * 2)
        views = metrics.get('views', 1)
        rate = (engagement / views) * 100
        retention = metrics.get('retention', 0)
        is_viral = rate >= 12 and retention >= 0.85
        return {
            'engagement_rate': round(rate, 2),
            'retention_score': round(retention * 100, 1),
            'is_viral': is_viral
        }

# ================= SISTEMA NERVIOSO ARTIFICIAL =================
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
            if valor > 0:
                self.emociones['curiosidad'] = min(1.0, self.emociones['curiosidad'] + 0.05)
            else:
                self.emociones['curiosidad'] = max(0.0, self.emociones['curiosidad'] - 0.05)

    def predecir_proximo_reward(self):
        if len(blockchain) < 5:
            return 5.0
        rewards = [b.reward for b in blockchain[-5:]]
        trend = (rewards[-1] - rewards[0]) / 5
        return rewards[-1] + trend

    def evaluar_salud(self):
        estabilidad = self.emociones['estabilidad']
        urgencia = self.emociones['urgencia']
        curiosidad = self.emociones['curiosidad']
        salud = (estabilidad * 0.5 + (1 - urgencia) * 0.3 + curiosidad * 0.2)
        self.autoevaluacion.append(salud)
        self.conciencia = sum(self.autoevaluacion) / len(self.autoevaluacion) if self.autoevaluacion else 0
        return round(salud, 3)

cerebro = CerebroDigital()

# Amigdala Digital
class AmigdalaDigital:
    def __init__(self):
        self.alertas = []

    def analizar_ambiente(self, bloque):
        score = bloque.viral_score.get('engagement_rate', 0)
        reward = bloque.reward
        if score > 12 and reward > 10:
            cerebro.registrar_experiencia("oportunidad_viral", 1.0)
            self.alertas.append(f"🔥 VIRAL DETECTADO: Block #{bloque.block_no}")
        elif reward < 1.0:
            cerebro.registrar_experiencia("bloque_insulso", -0.5)

amigdala = AmigdalaDigital()

# ================= FUNCIONES AUXILIARES =================
def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

def cargar_datos():
    if not CHAIN_FILE.exists():
        genesis = VideoBlock("", {}, sha256("genesis"), "0")
        with lock:
            blockchain.append(genesis)
        save_chain()

def save_chain():
    with lock:
        with open(CHAIN_FILE, "w") as f:
            json.dump([b.__dict__ for b in blockchain], f, indent=2)

def obtener_contenido_real():
    """Obtiene contenido real de APIs públicas"""
    try:
        posts = requests.get("https://jsonplaceholder.typicode.com/posts", timeout=5).json()
        users = requests.get("https://reqres.in/api/users", timeout=5).json()["data"]
        contenidos = []
        for i, post in enumerate(posts[:10]):
            user = users[i % len(users)]
            views = random.randint(5000, 500000)
            metrics = {
                'views': views,
                'likes': int(views * random.uniform(0.05, 0.25)),
                'shares': int(views * random.uniform(0.01, 0.1)),
                'saves': int(views * random.uniform(0.02, 0.17)),
                'comments': int(views * random.uniform(0.005, 0.05)),
                'retention': round(random.uniform(0.6, 0.95), 2)
            }
            contenidos.append({
                "url": f"https://socialcoin/post/{post['id']}",
                "title": post["title"],
                "author": f"{user['first_name']} {user['last_name']}",
                "metricas": metrics
            })
        return contenidos
    except:
        return [{"url": "https://socialcoin/post/1", "title": "Post simulado", "author": "Simulado", "metricas": generar_metricas_aleatorias()}]

def generar_metricas_aleatorias():
    views = random.randint(1000, 50000)
    return {
        'views': views,
        'likes': int(views * random.uniform(0.05, 0.25)),
        'shares': int(views * random.uniform(0.01, 0.1)),
        'saves': int(views * random.uniform(0.02, 0.17)),
        'comments': int(views * random.uniform(0.005, 0.05)),
        'retention': round(random.uniform(0.5, 1.0), 2)
    }

# ================= MINERÍA =================
async def minar_bloque_inteligente(metrics, url="https://socialcoin/post"):
    previous_hash = blockchain[-1].hash if blockchain else "0"
    prefix = f"{json.dumps(metrics)}{previous_hash}"
    nonce = random.randint(0, 100000)
    start_time = time.time()

    while (time.time() - start_time) < 10:
        hash_val = sha256(f"{prefix}{nonce}")
        if hash_val.startswith(difficulty):
            bloque = VideoBlock(url, metrics, hash_val, previous_hash)
            with lock:
                blockchain.append(bloque)
                save_chain()
            amigdala.analizar_ambiente(bloque)
            cerebro.registrar_experiencia("bloque_creado", bloque.reward)
            await broadcast_con_emocion(json.dumps({
                "type": "new_block",
                "data": bloque.__dict__
            }))
            log_con_pensamiento(f"BLOQUE #{bloque.block_no} minado. Recompensa: ${bloque.reward:.2f}")
            return bloque
        nonce += 1
        if nonce % 5000 == 0:
            await asyncio.sleep(0.01)
    return None

# ================= CONTROL REMOTO =================
modo_actual = "mineria"
modo_event = asyncio.Event()

async def ws_control_handler(websocket, path):
    if websocket.request_headers.get("Authorization") != f"Bearer {WS_TOKEN}":
        await websocket.close(code=1008, reason="Unauthorized")
        return
    control_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("cmd")
            clave = data.get("clave")

            if cmd == "activar_reporte" and clave == CLAVE_REPORTE:
                global modo_actual
                modo_actual = "reporte"
                modo_event.set()
                await websocket.send(json.dumps({"status": "reporte activado"}))

            elif cmd == "cambiar_modo":
                nuevo_modo = data.get("modo")
                if nuevo_modo in ["mineria", "exploracion", "reporte"]:
                    modo_actual = nuevo_modo
                    modo_event.set()
                    await websocket.send(json.dumps({"status": f"modo cambiado a {nuevo_modo}"}))
                else:
                    await websocket.send(json.dumps({"error": "modo inválido"}))
    finally:
        control_clients.discard(websocket)

# ================= MODO CONTROL Y EXPLORACIÓN =================
async def modo_control():
    while not STOP_EVENT.is_set():
        await modo_event.wait()
        if modo_actual == "reporte":
            estado = {
                "blockchain": [b.__dict__ for b in blockchain],
                "emociones": dict(cerebro.emociones),
                "alertas": amigdala.alertas[-5:]
            }
            estado_json = json.dumps(estado).encode()
            estado_enc = fernet.encrypt(estado_json)
            print(f"🔒 Reporte cifrado listo ({len(estado_enc)} bytes)")
            await asyncio.sleep(2)
            modo_actual = "mineria"
            modo_event.clear()
        else:
            await asyncio.sleep(1)

async def explorar_red():
    while not STOP_EVENT.is_set():
        if modo_actual != "exploracion":
            await asyncio.sleep(1)
            continue
        nodo = f"192.168.0.{random.randint(1, 254)}"
        respuesta = random.choice([True, False])
        print(f"🌐 Ping a {nodo} {'exitoso' if respuesta else 'fallido'}")
        cerebro.registrar_experiencia("exploracion_ping", 1.0 if respuesta else -0.5)
        await asyncio.sleep(random.uniform(0.5, 3))

# ================= MINING LOOP =================
async def mining_loop():
    while not STOP_EVENT.is_set():
        if modo_actual != "mineria":
            await asyncio.sleep(1)
            continue
        contenidos = obtener_contenido_real()
        for contenido in contenidos:
            if STOP_EVENT.is_set():
                break
            await minar_bloque_inteligente(contenido["metricas"], contenido["url"])
            await asyncio.sleep(3)
        await asyncio.sleep(MINING_INTERVAL)

# ================= WEBSOCKETS =================
async def broadcast(message):
    if not clients:
        return
    for ws in list(clients):
        try:
            await ws.send(message)
        except:
            clients.discard(ws)

async def broadcast_con_emocion(message):
    data = json.loads(message)
    data["cerebro"] = {
        "emociones": dict(cerebro.emociones),
        "salud": round(cerebro.evaluar_salud(), 3),
        "conciencia": round(cerebro.conciencia, 3),
        "alertas": amigdala.alertas[-3:]
    }
    await broadcast(json.dumps(data))

@functools.lru_cache(maxsize=1)
def inmortal(coro):
    async def wrapper(*a, **kw):
        while not STOP_EVENT.is_set():
            try:
                await coro(*a, **kw)
            except:
                await asyncio.sleep(2)
    return wrapper

@inmortal
async def ws_handler(websocket, path):
    if websocket.request_headers.get("Authorization") != f"Bearer {WS_TOKEN}":
        await websocket.close(code=1008, reason="Unauthorized")
        return
    clients.add(websocket)
    with lock:
        chain_data = json.dumps({
            "type": "full_chain",
            "data": [b.__dict__ for b in blockchain],
            "cerebro": {
                "emociones": dict(cerebro.emociones),
                "salud": round(cerebro.evaluar_salud(), 3),
                "conciencia": round(cerebro.conciencia, 3),
                "alertas": amigdala.alertas[-3:]
            }
        })
    try:
        await websocket.send(chain_data)
        async for _ in websocket: pass
    finally:
        clients.discard(websocket)

# ================= HTTP SERVER =================
def http_server():
    os.chdir(SIM_DIR)
    server = HTTPServer(("0.0.0.0", HTTP_PORT), SimpleHTTPRequestHandler)
    print(f"🌍 HTTP server activo en http://0.0.0.0:{HTTP_PORT}")
    server.serve_forever()

# ================= MAIN =================
async def main():
    SIM_DIR.mkdir(exist_ok=True)
    cargar_datos()
    signal.signal(signal.SIGINT, lambda s, f: STOP_EVENT.set())

    threading.Thread(target=http_server, daemon=True).start()
    
    # Servidores WebSocket
    server_tasks = [
        websockets.serve(ws_handler, "0.0.0.0", WS_PORT),
        websockets.serve(ws_control_handler, "0.0.0.0", CONTROL_WS_PORT)
    ]
    await asyncio.gather(*server_tasks, mining_loop(), modo_control(), explorar_red())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        print("\n\n🛑 Sistema detenido por el usuario.")
