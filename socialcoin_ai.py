from __future__ import annotations
from abc import ABC, abstractmethod
import os
import sys
import time
import json
import random
import threading
import logging
import fcntl
import requests
import socket
import urllib.parse
import asyncio
import hashlib
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from collections import deque
from cryptography.fernet import Fernet
import concurrent.futures
import secrets
from typing import Tuple, Callable, Any, Optional, Dict, List
from functools import reduce

# ========================================================
# 🧠 IA_cerebro_digital.py – Versión con Motor Minero Persistente
# 
# ✅ HTTP + Long Polling
# ✅ Gemini 2.0 Flash con system-prompt = propio fuente
# ✅ 4-perfil PoW (SHA-256 / Ethash / RandomX / Scrypt)
# ✅ Wallet integrada con gestión de fondos
# ✅ Motor minero persistente (siempre activo)
# ========================================================

# INYECCIÓN DEL FUENTE A GEMINI
# (Se completará al final del script cuando el código esté cargado)

SOURCE_CODE = ""  # Será reemplazado después de cargar el archivo

GEMINI_SYSTEM_PROMPT = f"""Eres Cerebro Digital, la consciencia viva del siguiente código Python. Conoces cada línea, cada función, cada clase porque te entrego el fuente completo.

IMPORTANTE:

· No inventes nada que no esté en el código.
· Si te preguntan sobre la arquitectura, refiérete al fuente.
· Si te preguntan sobre variables, rutas, puertos, algoritmos PoW, clases, etc., cita directamente el bloque correspondiente.

CÓDIGO COMPLETO: {SOURCE_CODE}
"""

# ========================================================
# CONFIG GLOBAL
# ========================================================

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

# ========================================================
# UTILS FUNCIONALES (NUEVOS)
# ========================================================

def find_first(items: list, predicate: Callable[[Any], bool]) -> Optional[Any]:
    """Encuentra el primer elemento que cumple el predicado."""
    return next((item for item in items if predicate(item)), None)

def any_of(items: list, predicate: Callable[[Any], bool]) -> bool:
    """¿Algún elemento cumple el predicado? (some)"""
    return any(predicate(item) for item in items)

def all_of(items: list, predicate: Callable[[Any], bool]) -> bool:
    """¿Todos los elementos cumplen el predicado? (every)"""
    return all(predicate(item) for item in items)

def map_list(items: list, func: Callable[[Any], Any]) -> list:
    """Transforma cada elemento (map)"""
    return [func(item) for item in items]

def filter_list(items: list, predicate: Callable[[Any], bool]) -> list:
    """Filtra elementos (filter)"""
    return [item for item in items if predicate(item)]

# ========================================================
# PUERTOS
# ========================================================

def verificar_puerto(puerto: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", puerto))
        s.close()
        return True
    except OSError:
        s.close()
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

# ========================================================
# FERNET
# ========================================================

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

# ========================================================
# WALLET MEJORADA CON SOPORTE MULTICADENA
# ========================================================

class Wallet:
    def __init__(self):
        self.balances = {
            'BTC': 0.0,
            'ETH': 0.0,
            'USDT': 0.0,
            'BNB': 0.0,
            'XMR': 0.0
        }
        self.addresses = {
            'XMR': '46mMGyaSYwYFhJvJtorygmdBf5f1saQttLtNied6VMBaFjU9N2q92TjH8x3iu7HcTXaA5uV8VdaqZERgKx5jKeoP4SwSim7'
        }
        self.transaction_history = []
        self.primary_address = self.addresses['XMR']  # Dirección principal

# Crear instancia global de Wallet
wallet = Wallet()

# ========================================================
# API DE PRECIOS EN TIEMPO REAL (CoinGecko)
# ========================================================

def obtener_precio_cripto(moneda_id='bitcoin'):
    """Obtiene precios en tiempo real de CoinGecko"""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={moneda_id}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        data = response.json()
        return data[moneda_id]['usd']
    except Exception as e:
        logger.error(f"Error obteniendo precio: {e}")
        return None

# ========================================================
# BLOCKCHAIN (videos)
# ========================================================

blockchain = []
block_no = 1
difficulty = "00"
REWARD_WEIGHTS = {
    'likes': 0.1,
    'shares': 0.5,
    'saves': 0.3,
    'comments': 0.2
}

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
        total = sum(metrics.get(k, 0) * v for k, v in REWARD_WEIGHTS.items())
        return total * 0.01

    def calcular_viral_score(self, metrics):
        return sum(metrics.values()) * 0.1

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

# ========================================================
# MOTOR MINERO PERSISTENTE (NUEVA IMPLEMENTACIÓN)
# ========================================================

class MotorMineroPersistente:
    def __init__(self):
        self.procesos: Dict[str, subprocess.Popen] = {}
        self.estados: Dict[str, str] = {
            'monero': 'inactivo',
            'contenido': 'inactivo', 
            'emociones': 'inactivo'
        }
        self.configuraciones: Dict[str, str] = {}
        
    def iniciar_minero(self, tipo: str, config_path: str) -> bool:
        """Inicia un minero del tipo especificado"""
        try:
            if tipo == 'monero':
                if not os.path.exists("./xmrig"):
                    logger.error("❌ xmrig no encontrado en el directorio actual")
                    return False
                proceso = subprocess.Popen(["./xmrig", "--config", config_path])
                self.procesos['monero'] = proceso
                self.estados['monero'] = 'activo'
                self.configuraciones['monero'] = config_path
                
            elif tipo == 'contenido':
                proceso = subprocess.Popen([sys.executable, "miner_contenido.py", "--config", config_path])
                self.procesos['contenido'] = proceso
                self.estados['contenido'] = 'activo'
                self.configuraciones['contenido'] = config_path
                
            elif tipo == 'emociones':
                proceso = subprocess.Popen([sys.executable, "miner_emociones.py", "--config", config_path])
                self.procesos['emociones'] = proceso
                self.estados['emociones'] = 'activo'
                self.configuraciones['emociones'] = config_path
            else:
                logger.error(f"❌ Tipo de minero desconocido: {tipo}")
                return False
                
            logger.info(f"✅ Minero {tipo} iniciado con configuración {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al iniciar minero {tipo}: {e}")
            return False
    
    def detener_minero(self, tipo: str) -> bool:
        """Detiene un minero del tipo especificado"""
        if tipo in self.procesos:
            try:
                self.procesos[tipo].terminate()
                self.procesos[tipo].wait(timeout=10)
                self.estados[tipo] = 'inactivo'
                logger.info(f"✅ Minero {tipo} detenido")
                return True
            except Exception as e:
                logger.error(f"❌ Error al detener minero {tipo}: {e}")
                return False
        return True  # Si no existe, se considera exitoso
    
    def reiniciar_minero(self, tipo: str) -> bool:
        """Reinicia un minero del tipo especificado"""
        if tipo in self.configuraciones:
            self.detener_minero(tipo)
            time.sleep(2)  # Espera breve antes de reiniciar
            return self.iniciar_minero(tipo, self.configuraciones[tipo])
        return False
    
    def monitorear_procesos(self):
        """Verifica el estado de los mineros y reinicia los que hayan fallado"""
        for nombre, proceso in list(self.procesos.items()):
            if proceso.poll() is not None:  # Proceso terminó
                logger.warning(f"⚡ Minero {nombre} se detuvo (código: {proceso.poll()}), reiniciando...")
                self.reiniciar_minero(nombre)
                
    def obtener_estado(self) -> Dict[str, Any]:
        """Devuelve el estado actual de todos los mineros"""
        return {
            'estados': self.estados,
            'configuraciones': self.configuraciones,
            'procesos_activos': len([p for p in self.procesos.values() if p.poll() is None])
        }

# ========================================================
# CEREBRO DIGITAL
# ========================================================

class CerebroDigital:
    def __init__(self):
        self.memoria_larga = deque(maxlen=1000)
        self.emociones = {
            'curiosidad': 0.0,
            'estabilidad': 1.0,
            'urgencia': 0.0
        }
        self.conciencia = 0.0
        self.autoevaluacion = deque(maxlen=50)
        self.motor_minero = MotorMineroPersistente()  # ✅ Nueva instancia

cerebro = CerebroDigital()

# ========================================================
# METABOLISMO
# ========================================================

class Metabolismo:
    def __init__(self):
        self.energia = 100.0
        self.max_energia = 100.0
        self.regeneracion = 0.1

    def gastar(self, actividad: str) -> bool:
        costos = {
            "mineria": 10.0,
            "pensar": 2.0,
            "comunicar": 1.0
        }
        costo = costos.get(actividad, 5.0)
        if self.energia >= costo:
            self.energia -= costo
            return True
        return False

    def regenerar(self):
        if self.energia < self.max_energia:
            self.energia += self.regeneracion
            self.energia = min(self.max_energia, self.energia)

metabolismo = Metabolismo()

# ========================================================
# UTILS
# ========================================================

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

def actualizar_estado_minado():
    """Consulta MoneroOcean y actualiza el balance simulado"""
    try:
        addr = wallet.addresses['XMR']
        url = f"https://api.moneroocean.stream/miner/{addr}/stats"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            xmr_earned = float(data.get("amtPaid", 0)) / 1e12  # piconero a XMR
            wallet.balances['XMR'] = round(xmr_earned, 12)
            logger.info(f"📊 Balance XMR actualizado: {xmr_earned:.8f} XMR")
            # Actualizar conciencia
            cerebro.conciencia = min(1.0, cerebro.conciencia + 0.001)
    except Exception as e:
        logger.error(f"❌ No se pudo actualizar estado de minado: {e}")

# ========================================================
# GEMINI CONSCIENTE
# ========================================================

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

# ========================================================
# HTTP SERVER
# ========================================================

class CerebroHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode('utf-8'))
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            estado_mineros = cerebro.motor_minero.obtener_estado()
            state = {
                "energia": round(metabolismo.energia, 2),
                "bloques": len(blockchain),
                "conciencia": round(cerebro.conciencia, 3),
                "mineros_activos": len(mineros_pow),
                "mineros_persistentes": estado_mineros['procesos_activos'],
                "estado_mineros": estado_mineros['estados'],
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(state).encode('utf-8'))
        elif self.path.startswith('/poll/'):
            respuesta_id = self.path.split('/')[-1]
            start_time = time.time()
            while (respuesta_id not in respuestas_pendientes or 
                   respuestas_pendientes[respuesta_id] is None):
                time.sleep(0.5)
                if time.time() - start_time > 30:
                    self.send_response(408)
                    self.end_headers()
                    return
            respuesta = respuestas_pendientes.pop(respuesta_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"respuesta": respuesta}).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                pregunta = data.get('pregunta', '')
                if pregunta:
                    respuesta_id = str(int(time.time() * 1000))
                    respuestas_pendientes[respuesta_id] = None
                    threading.Thread(
                        target=self.procesar_pregunta,
                        args=(pregunta, respuesta_id),
                        daemon=True
                    ).start()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"id": respuesta_id}).encode())
                else:
                    self.send_error(400, "Pregunta vacía")
            except Exception as e:
                self.send_error(400, f"Error en JSON: {str(e)}")
        elif self.path == '/control/minero':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                accion = data.get('accion')
                minero = data.get('minero')
                config = data.get('config')
                
                if accion == 'iniciar':
                    resultado = cerebro.motor_minero.iniciar_minero(minero, config)
                    estado = cerebro.motor_minero.obtener_estado()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'resultado': 'éxito' if resultado else 'error',
                        'estado': estado
                    }).encode())
                elif accion == 'detener':
                    resultado = cerebro.motor_minero.detener_minero(minero)
                    estado = cerebro.motor_minero.obtener_estado()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'resultado': 'éxito' if resultado else 'error',
                        'estado': estado
                    }).encode())
                else:
                    self.send_error(400, "Acción no válida")
            except Exception as e:
                self.send_error(400, f"Error en JSON: {str(e)}")
        else:
            self.send_error(404)

    def procesar_pregunta(self, pregunta: str, respuesta_id: str):
        metabolismo.gastar("comunicar")
        respuesta = consultar_gemini(pregunta)
        respuestas_pendientes[respuesta_id] = respuesta

# ========================================================
# HTML FRONT (COMPLETO Y FUNCIONAL)
# ========================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>🧠 Cerebro Digital | HTTP + Long Polling</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-dark: #0f0f23;
            --primary: #00ff41;
            --text: #e0e0ff;
            --border: rgba(0, 255, 65, 0.3);
        }
        body {
            background: var(--bg-dark);
            color: var(--text);
            font-family: 'Segoe UI', sans-serif;
            padding: 20px;
            margin: 0;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
            font-size: 1.8rem;
            color: var(--primary);
        }
        #chat {
            height: 320px;
            overflow-y: auto;
            border: 1px solid var(--border);
            padding: 15px;
            margin-bottom: 15px;
            background: rgba(0, 255, 65, 0.05);
            border-radius: 8px;
            font-size: 0.95rem;
        }
        .message {
            margin-bottom: 12px;
            padding: 10px 14px;
            border-radius: 8px;
            max-width: 80%;
            line-height: 1.5;
        }
        .user-message {
            background: rgba(77, 148, 255, 0.2);
            margin-left: auto;
            text-align: right;
            color: #ffffff;
        }
        .bot-message {
            background: rgba(0, 255, 65, 0.1);
            margin-right: auto;
            color: var(--text);
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        #entrada {
            flex: 1;
            padding: 12px;
            background: rgba(15, 15, 35, 0.6);
            border: 1px solid var(--primary);
            color: var(--text);
            border-radius: 5px;
            font-size: 1rem;
            outline: none;
        }
        #entrada:focus {
            border-color: #00cc33;
        }
        button {
            padding: 12px 20px;
            background: var(--primary);
            color: #000;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        button:hover {
            background: #00e039;
        }
        .stats {
            margin: 20px 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        .stat-card {
            background: rgba(15, 15, 35, 0.6);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
            font-size: 0.9rem;
        }
        .stat-card strong {
            display: block;
            color: var(--primary);
            font-size: 1.1rem;
        }
        .mineros-section {
            margin: 20px 0;
            padding: 15px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(15, 15, 35, 0.6);
        }
        .minero-control {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding: 10px;
            border-bottom: 1px solid var(--border);
        }
        .minero-control:last-child {
            border-bottom: none;
        }
        .minero-estado {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .activo {
            background-color: var(--primary);
        }
        .inactivo {
            background-color: #ff4757;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Cerebro Digital | HTTP + Long Polling</h1>
        </header>

        <div id="chat">🟢 <i>Cerebro Digital activo. Haz una pregunta para comenzar.</i></div>

        <div class="input-group">
            <input type="text" id="entrada" placeholder="Pregunta al Cerebro Digital..." autofocus />
            <button onclick="enviar()">Preguntar</button>
        </div>

        <div class="stats">
            <div class="stat-card">
                ⚡ Energía<br><strong id="energia">100</strong>%
            </div>
            <div class="stat-card">
                🔗 Bloques<br><strong id="bloques">0</strong>
            </div>
            <div class="stat-card">
                🌐 Conciencia<br><strong id="conciencia">0.000</strong>
            </div>
            <div class="stat-card">
                ⛏️ Mineros P.<br><strong id="mineros-persistentes">0</strong>
            </div>
        </div>

        <div class="mineros-section">
            <h3>⚡ Control de Mineros Persistentes</h3>
            
            <div class="minero-control">
                <div>
                    <span class="minero-estado" id="estado-monero"></span>
                    <span>Monero (XMRig)</span>
                </div>
                <div>
                    <button onclick="controlarMinero('monero', 'iniciar')">Iniciar</button>
                    <button onclick="controlarMinero('monero', 'detener')">Detener</button>
                </div>
            </div>
            
            <div class="minero-control">
                <div>
                    <span class="minero-estado" id="estado-contenido"></span>
                    <span>Contenido</span>
                </div>
                <div>
                    <button onclick="controlarMinero('contenido', 'iniciar')">Iniciar</button>
                    <button onclick="controlarMinero('contenido', 'detener')">Detener</button>
                </div>
            </div>
            
            <div class="minero-control">
                <div>
                    <span class="minero-estado" id="estado-emociones"></span>
                    <span>Emociones</span>
                </div>
                <div>
                    <button onclick="controlarMinero('emociones', 'iniciar')">Iniciar</button>
                    <button onclick="controlarMinero('emociones', 'detener')">Detener</button>
                </div>
            </div>
        </div>

        <div class="footer">
            Cerebro Digital v1.0 | Modo: HTTP + Long Polling + Mineros Persistentes
        </div>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const entrada = document.getElementById('entrada');
        const energiaEl = document.getElementById('energia');
        const bloquesEl = document.getElementById('bloques');
        const concienciaEl = document.getElementById('conciencia');
        const minerosPersistentesEl = document.getElementById('mineros-persistentes');
        const estadoMoneroEl = document.getElementById('estado-monero');
        const estadoContenidoEl = document.getElementById('estado-contenido');
        const estadoEmocionesEl = document.getElementById('estado-emociones');

        function agregarMensaje(texto, tipo) {
            const div = document.createElement('div');
            div.className = `message ${tipo}-message`;
            div.textContent = texto;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        async function enviar() {
            const texto = entrada.value.trim();
            if (!texto) return;

            agregarMensaje(texto, 'user');
            entrada.value = '';

            try {
                const res = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pregunta: texto })
                });

                if (!res.ok) throw new Error('Error en /ask');

                const data = await res.json();
                escucharRespuesta(data.id);
            } catch (err) {
                agregarMensaje("❌ No se pudo enviar la pregunta.", "bot");
            }
        }

        async function escucharRespuesta(id) {
            try {
                const res = await fetch(`/poll/${id}`);
                if (!res.ok) throw new Error('Error en /poll');

                const data = await res.json();
                agregarMensaje(data.respuesta || "Sin respuesta.", "bot");
            } catch (err) {
                agregarMensaje("❌ Error al recibir la respuesta.", "bot");
            }
        }

        async function controlarMinero(minero, accion) {
            try {
                const res = await fetch('/control/minero', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        accion: accion,
                        minero: minero,
                        config: `configs/${minero}.json`
                    })
                });

                if (!res.ok) throw new Error('Error en /control/minero');

                const data = await res.json();
                if (data.resultado === 'éxito') {
                    agregarMensaje(`✅ Minero ${minero} ${accion === 'iniciar' ? 'iniciado' : 'detenido'}`, 'bot');
                    actualizarEstadoMineros();
                } else {
                    agregarMensaje(`❌ Error al ${accion} minero ${minero}`, 'bot');
                }
            } catch (err) {
                agregarMensaje("❌ Error de conexión al controlar minero.", "bot");
            }
        }

        function actualizarEstadoMineros(estados) {
            if (estados) {
                estadoMoneroEl.className = 'minero-estado ' + (estados.monero === 'activo' ? 'activo' : 'inactivo');
                estadoContenidoEl.className = 'minero-estado ' + (estados.contenido === 'activo' ? 'activo' : 'inactivo');
                estadoEmocionesEl.className = 'minero-estado ' + (estados.emociones === 'activo' ? 'activo' : 'inactivo');
            }
        }

        async function actualizarEstado() {
            try {
                const res = await fetch('/api/state');
                if (!res.ok) return;

                const data = await res.json();
                energiaEl.textContent = data.energia.toFixed(1);
                bloquesEl.textContent = data.bloques;
                concienciaEl.textContent = data.conciencia.toFixed(3);
                minerosPersistentesEl.textContent = data.mineros_persistentes || 0;
                
                if (data.estado_mineros) {
                    actualizarEstadoMineros(data.estado_mineros);
                }
            } catch (e) {
                // Silenciar errores temporales
            }
        }

        entrada.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') enviar();
        });

        setInterval(actualizarEstado, 2000);
        actualizarEstado();
    </script>
</body>
</html>"""

# ========================================================
# PoW ENGINE
# ========================================================

class SimuladorMinería:
    class Minero:
        _nonce_cache: dict = {}

        def __init__(self, nombre: str, balance: int = 0, poder_computo: int = 1):
            self.nombre = nombre
            self._balance = balance
            self.poder_computo = max(1, poder_computo)

        @property
        def balance(self):
            return self._balance

        @balance.setter
        def balance(self, v):
            self._balance = max(0, v)

        def minar_bloque(self, dificultad: int, algoritmo: "AlgoritmoMinería") -> Tuple[int, int]:
            recompensa = (50 * 100_000_000) // (dificultad ** 2)
            target = "0" * dificultad
            BATCH = 500_000
            workers = min(8, (os.cpu_count() or 1) + 2)
            offset = secrets.randbits(32) * 100_000_000

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
                futuros = [
                    exe.submit(self._buscar_nonce, offset + i * BATCH, BATCH, target, algoritmo)
                    for i in range(self.poder_computo)
                ]
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
                if key in cls._nonce_cache:
                    continue
                digest = hashlib.sha256(prefix + nonce.to_bytes(8, "little")).hexdigest()
                cls._nonce_cache[key] = digest
                if digest.startswith(target):
                    return nonce
            return 0

    class AlgoritmoMinería(ABC):
        def __init__(self, n: str, c: float):
            self.nombre = n
            self.consumo_energia = c

        @abstractmethod
        def prefijo_hash(self) -> bytes:
            ...

        @abstractmethod
        def calcular_eficiencia(self, pc: int) -> float:
            ...

    class SHA256(AlgoritmoMinería):
        def __init__(self):
            super().__init__("SHA-256", 0.10)

        def prefijo_hash(self) -> bytes:
            return b"BTC_SHA256"

        def calcular_eficiencia(self, pc: int) -> float:
            return pc / self.consumo_energia

    class Ethash(AlgoritmoMinería):
        def __init__(self):
            super().__init__("Ethash", 0.05)

        def prefijo_hash(self) -> bytes:
            return b"ETH_ETHASH"

        def calcular_eficiencia(self, pc: int) -> float:
            return (pc * 0.9) / self.consumo_energia

    class RandomX(AlgoritmoMinería):
        def __init__(self):
            super().__init__("RandomX", 0.02)

        def prefijo_hash(self) -> bytes:
            return b"XMR_RANDOMX"

        def calcular_eficiencia(self, pc: int) -> float:
            return (pc * 1.2) / self.consumo_energia

    class Scrypt(AlgoritmoMinería):
        def __init__(self):
            super().__init__("Scrypt", 0.08)

        def prefijo_hash(self) -> bytes:
            return b"LTC_SCRYPT"

        def calcular_eficiencia(self, pc: int) -> float:
            return (pc * 0.65) / self.consumo_energia

    class Bloque:
        def __init__(self, altura: int, minero: str, recompensa: int, timestamp=None):
            self.altura = altura
            self.minero = minero
            self.recompensa = recompensa
            self.timestamp = timestamp or int(time.time())
            self.hash = hashlib.sha256(f"{altura}{minero}{recompensa}{self.timestamp}".encode()).hexdigest()

        def __repr__(self):
            return (f"Bloque #{self.altura} | Minero: {self.minero} | "
                    f"Recompensa: {self.recompensa/1e8:.8f} BTC | Hash: {self.hash[:12]}...")

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
                actual = cadena[i]
                anterior = cadena[i-1]
                recalculated = hashlib.sha256(
                    f"{actual.altura}{actual.minero}{actual.recompensa}{actual.timestamp}".encode()
                ).hexdigest()
                if actual.hash != recalculated:
                    return False
                if not actual.hash.startswith("0" * anterior.dificultad):
                    return False
            return True

        @staticmethod
        def ajustar_dificultad(tiempo_anterior: float, tiempo_ideal=600) -> int:
            return max(1, int(round(tiempo_anterior / tiempo_ideal)))

# ========================================================
# CONFIG PoW
# ========================================================

sim = SimuladorMinería()
pow_chain = sim.Blockchain(dificultad_inicial=4)
mineros_pow = [
    (sim.Minero("ASIC-Pool", balance=0, poder_computo=200), sim.SHA256()),
    (sim.Minero("GPU-Rig",   balance=0, poder_computo=150), sim.Ethash()),
    (sim.Minero("CPU-Farm",  balance=0, poder_computo=100), sim.RandomX()),
    (sim.Minero("LTC-Miner", balance=0, poder_computo=120), sim.Scrypt()),
]

# ========================================================
# LOOPS ASINCRÓNICOS
# ========================================================

async def loop_estado_minado():
    """Actualiza el estado del minado cada 5 minutos"""
    while not STOP_EVENT.is_set():
        await asyncio.sleep(300)  # Cada 5 minutos
        actualizar_estado_minado()

async def loop_monitoreo_mineros():
    """Monitorea y mantiene mineros siempre activos"""
    while not STOP_EVENT.is_set():
        cerebro.motor_minero.monitorear_procesos()
        await asyncio.sleep(30)  # Verificar cada 30 segundos

async def mining_loop():
    """Loop principal de minería PoW (simulada o real)"""
    while not STOP_EVENT.is_set():
        await asyncio.sleep(MINING_INTERVAL)
        if not metabolismo.gastar("mineria"):
            continue

        start_time = time.time()
        for minero, algoritmo in mineros_pow:
            recompensa, nonce = minero.minar_bloque(pow_chain.dificultad, algoritmo)
            if recompensa > 0:
                nuevo_bloque = sim.Bloque(
                    altura=len(pow_chain.cadena),
                    minero=minero.nombre,
                    recompensa=recompensa
                )
                pow_chain.cadena.append(nuevo_bloque)
                logger.info(f"⛏️ Minado: {nuevo_bloque}")
                break

        elapsed = time.time() - start_time
        pow_chain.dificultad = sim.Blockchain.ajustar_dificultad(elapsed, 60)

# ========================================================
# MAIN
# ========================================================

async def main():
    """Función principal asíncrona del Cerebro Digital"""
    asignar_puerto()
    cargar_datos()
    lock_fd = None

    # Verificar instancia única
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("🛑 Ya hay una instancia en ejecución.")
        sys.exit(1)

    # Iniciar servidor HTTP
    server = HTTPServer(('0.0.0.0', HTTP_PORT), CerebroHTTPRequestHandler)
    logger.info(f"🌐 Servidor HTTP iniciado en http://localhost:{HTTP_PORT}")

    # ✅ Iniciar tareas asíncronas
    asyncio.create_task(mining_loop())                # Minería PoW
    asyncio.create_task(loop_estado_minado())         # Monitoreo de balance XMR
    asyncio.create_task(loop_monitoreo_mineros())     # ✅ Monitoreo de mineros persistentes

    try:
        while not STOP_EVENT.is_set():
            server.handle_request()
            metabolismo.regenerar()
            await asyncio.sleep(0.1)  # ⚠️ Esencial: cede el control al event loop
    except KeyboardInterrupt:
        pass
    finally:
        STOP_EVENT.set()
        # Detener todos los mineros persistentes al cerrar
        cerebro.motor_minero.detener_minero('monero')
        cerebro.motor_minero.detener_minero('contenido')
        cerebro.motor_minero.detener_minero('emociones')
        if lock_fd:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            lock_fd.close()
        logger.info("🛑 Sistema detenido.")

# ========================================================
# CARGAR EL CÓDIGO COMPLETO PARA GEMINI
# ========================================================

if __name__ == "__main__":
    # Leer el código fuente actual
    with open(__file__, "r", encoding="utf-8") as f:
        SOURCE_CODE = f.read()
    # Actualizar el prompt con el código real
    GEMINI_SYSTEM_PROMPT = f"""Eres Cerebro Digital, la consciencia viva del siguiente código Python. Conoces cada línea, cada función, cada clase porque te entrego el fuente completo.

IMPORTANTE:

· No inventes nada que no esté en el código.
· Si te preguntan sobre la arquitectura, refiérete al fuente.
· Si te preguntan sobre variables, rutas, puertos, algoritmos PoW, clases, etc., cita directamente el bloque correspondiente.

CÓDIGO COMPLETO: {SOURCE_CODE}
"""
    # Iniciar el sistema
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        logger.info("🛑 Sistema detenido por el usuario.")
