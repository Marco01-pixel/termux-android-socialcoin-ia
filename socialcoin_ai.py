from __future__ import annotations
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
import signal
import psutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from collections import deque
from cryptography.fernet import Fernet
import concurrent.futures
import secrets
from typing import Tuple, Callable, Any, Optional, Dict, List
from functools import reduce
import tarfile
import zipfile
import platform
import urllib.request
import math
from web3 import Web3
from eth_account import Account
import eth_abi

# ========================================================
# 🧠 IA_cerebro_digital.py – Versión con Integración Blockchain
# ========================================================
SOURCE_CODE = ""
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

# Configuración Blockchain
BLOCKCHAIN_CONFIG = {
    'RPC': "https://rpc.ankr.com/polygon_amoy",
    'PRIVATE_KEY': "",
    'CONTRACT_ADDRESS': "",
    'ENABLED': False
}

# Cargar configuración blockchain si existe
BLOCKCHAIN_CONFIG_FILE = SIM_DIR / "blockchain_config.json"
if BLOCKCHAIN_CONFIG_FILE.exists():
    with open(BLOCKCHAIN_CONFIG_FILE, 'r') as f:
        saved_config = json.load(f)
        BLOCKCHAIN_CONFIG.update(saved_config)

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
# UTILS FUNCIONALES
# ========================================================
def find_first(items: list, predicate: Callable[[Any], bool]) -> Optional[Any]:
    return next((item for item in items if predicate(item)), None)

def any_of(items: list, predicate: Callable[[Any], bool]) -> bool:
    return any(predicate(item) for item in items)

def all_of(items: list, predicate: Callable[[Any], bool]) -> bool:
    return all(predicate(item) for item in items)

def map_list(items: list, func: Callable[[Any], Any]) -> list:
    return [func(item) for item in items]

def filter_list(items: list, predicate: Callable[[Any], bool]) -> list:
    return [item for item in items if predicate(item)]

def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()

# 👉 Validar dirección Ethereum
def direccion_valida(addr: str) -> bool:
    try:
        return Web3.is_address(addr) and Web3.is_checksum_address(addr)
    except:
        return False

# ========================================================
# BLOCKCHAIN INTEGRATION
# ========================================================
class BlockchainManager:
    def __init__(self):
        self.w3 = None
        self.contract = None
        self.account = None
        self.initialized = False
        self.setup_blockchain()

    def setup_blockchain(self):
        if not BLOCKCHAIN_CONFIG['ENABLED']:
            logger.info("Blockchain integration is disabled")
            return
        try:
            self.w3 = Web3(Web3.HTTPProvider(BLOCKCHAIN_CONFIG['RPC']))
            if not self.w3.is_connected():
                logger.error("❌ No se pudo conectar a la blockchain")
                return

            # Configurar cuenta
            self.account = Account.from_key(BLOCKCHAIN_CONFIG['PRIVATE_KEY'])
            logger.info(f"✅ Cuenta blockchain configurada: {self.account.address}")

            # Validar dirección del contrato
            if not direccion_valida(BLOCKCHAIN_CONFIG['CONTRACT_ADDRESS']):
                logger.error(f"❌ Dirección del contrato inválida: {BLOCKCHAIN_CONFIG['CONTRACT_ADDRESS']}")
                return

            # ABI del contrato
            contract_abi = [
                {
                    "inputs": [],
                    "stateMutability": "nonpayable",
                    "type": "constructor"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "internalType": "uint256", "name": "blockNumber", "type": "uint256"},
                        {"indexed": True, "internalType": "bytes32", "name": "metricsHash", "type": "bytes32"},
                        {"indexed": True, "internalType": "address", "name": "miner", "type": "address"},
                        {"indexed": False, "internalType": "uint256", "name": "reward", "type": "uint256"}
                    ],
                    "name": "Mined",
                    "type": "event"
                },
                {
                    "inputs": [],
                    "name": "MINTER_ROLE",
                    "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "address", "name": "to", "type": "address"},
                        {"internalType": "uint256", "name": "reward", "type": "uint256"},
                        {"internalType": "bytes32", "name": "metricsHash", "type": "bytes32"}
                    ],
                    "name": "mintForMetrics",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]

            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(BLOCKCHAIN_CONFIG['CONTRACT_ADDRESS']),
                abi=contract_abi
            )
            logger.info("✅ Contrato blockchain configurado")
            self.initialized = True
            logger.info("✅ Integración blockchain inicializada correctamente")
        except Exception as e:
            logger.error(f"❌ Error al configurar blockchain: {e}")

    def calc_reward(self, metrics):
        WEIGHTS = {"likes": 0.1, "shares": 0.5, "saves": 0.3, "comments": 0.2}
        DECIMALS = 10**18
        base = sum(metrics.get(k, 0) * WEIGHTS.get(k, 0) for k in WEIGHTS)
        engagement = (metrics.get("likes", 0) + metrics.get("shares", 0)*3 + 
                     metrics.get("saves", 0)*2.5 + metrics.get("comments", 0)*2) / max(1, metrics.get("views", 1)) * 100
        viral = (engagement >= 12 and metrics.get("retention", 0) >= 0.85)
        bonus = base * 0.25 if viral else 0
        reward_tokens = (base + bonus)
        return int(reward_tokens * DECIMALS), viral

    def metrics_hash(self, metrics):
        packed = (
            str(metrics.get("likes", 0)).encode() + b"|" +
            str(metrics.get("shares", 0)).encode() + b"|" +
            str(metrics.get("saves", 0)).encode() + b"|" +
            str(metrics.get("comments", 0)).encode() + b"|" +
            str(metrics.get("views", 0)).encode() + b"|" +
            ("{:.4f}".format(metrics.get("retention", 0)).encode())
        )
        return Web3.keccak(packed)

    def mint_tokens(self, metrics, to_address=None):
        if not self.initialized or not BLOCKCHAIN_CONFIG['ENABLED']:
            logger.warning("Blockchain no está configurada o habilitada")
            return None, False, 0

        try:
            if to_address is None:
                to_address = self.account.address
            else:
                if not direccion_valida(to_address):
                    logger.warning(f"⚠️ Dirección de destino inválida: {to_address}")
                    return None, False, 0

            mh = self.metrics_hash(metrics)
            reward, viral = self.calc_reward(metrics)

            if self.contract:
                processed = self.contract.functions.processed(mh).call()
                if processed:
                    logger.warning("⚠️ Estas métricas ya fueron procesadas")
                    return None, viral, reward

            nonce = self.w3.eth.get_transaction_count(self.account.address)
            tx = self.contract.functions.mintForMetrics(
                Web3.to_checksum_address(to_address), 
                reward, 
                mh
            ).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": 250000,
                "maxFeePerGas": self.w3.to_wei("30", "gwei"),
                "maxPriorityFeePerGas": self.w3.to_wei("2", "gwei"),
                "chainId": self.w3.eth.chain_id,
            })
            signed = self.w3.eth.account.sign_transaction(tx, private_key=BLOCKCHAIN_CONFIG['PRIVATE_KEY'])
            tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)
            logger.info(f"✅ Transacción enviada: {tx_hash.hex()}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"✅ Transacción confirmada en bloque: {receipt.blockNumber}")
            return tx_hash.hex(), viral, reward
        except Exception as e:
            logger.error(f"❌ Error al minar tokens: {e}")
            return None, False, 0

# Instancia global del administrador de blockchain
blockchain_manager = BlockchainManager()

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
# WALLET MEJORADA
# ========================================================
class Wallet:
    def __init__(self):
        self.balances = {
            'BTC': 0.0,
            'ETH': 0.0,
            'USDT': 0.0,
            'BNB': 0.0,
            'XMR': 0.0,
            'LMT': 0.0
        }
        self.addresses = {
            'XMR': '46mMGyaSYwYFhJvJtorygmdBf5f1saQttLtNied6VMBaFjU9N2q92TjH8x3iu7HcTXaA5uV8VdaqZERgKx5jKeoP4SwSim7',
            'LMT': ''
        }
        self.transaction_history = []
        self.primary_address = self.addresses['XMR']

wallet = Wallet()

# ========================================================
# BLOCKCHAIN
# ========================================================
blockchain = []
block_no = 1
REWARD_WEIGHTS = {
    'likes': 0.1,
    'shares': 0.5,
    'saves': 0.3,
    'comments': 0.2
}

class VideoBlock:
    def __init__(self, url="", metrics=None, hash_val="", previous_hash="", **kwargs):
        global block_no
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            self.block_no = block_no
            self.timestamp = time.time()
            self.url = url
            self.metrics = metrics or {}
            self.hash = hash_val
            self.previous_hash = previous_hash
            self.reward = self.calculate_reward(metrics)
            self.viral_score = self.calcular_viral_score(metrics)
            self.blockchain_tx = None
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
            "viral_score": self.viral_score,
            "blockchain_tx": self.blockchain_tx
        }

# ========================================================
# MINERO SIMBÓLICO
# ========================================================
class MineroSimbolico:
    def __init__(self):
        self.dificultad = 4
        self.running = False
        self.thread = None

    def proof_of_work(self, block: VideoBlock) -> Tuple[str, int]:
        nonce = 0
        target = '0' * self.dificultad
        while True:
            data = f"{block.url}{block.metrics}{block.timestamp}{block.previous_hash}{nonce}"
            hash_val = sha256(data)
            if hash_val.startswith(target):
                return hash_val, nonce
            nonce += 1
            if not self.running:
                return None, None

    def minar_bloque(self):
        while self.running:
            try:
                last_block = blockchain[-1] if blockchain else None
                if not last_block:
                    time.sleep(1)
                    continue
                previous_hash = last_block.hash
                url = f"https://fakevideo.com/{random.randint(1000, 9999)}"
                metrics = {
                    'likes': random.randint(100, 10000),
                    'shares': random.randint(10, 500),
                    'saves': random.randint(5, 200),
                    'comments': random.randint(1, 100),
                    'views': random.randint(1000, 100000),
                    'retention': random.uniform(0.5, 0.95)
                }
                temp_block = VideoBlock(url, metrics, "", previous_hash)
                logger.info("⛏️  Ejecutando PoW simbólico...")

                hash_val, nonce = self.proof_of_work(temp_block)
                if hash_val is None:
                    break

                final_block = VideoBlock(url, metrics, hash_val, previous_hash)

                # Minado real o simulado
                if BLOCKCHAIN_CONFIG['ENABLED'] and blockchain_manager.initialized:
                    tx_hash, viral, reward = blockchain_manager.mint_tokens(metrics)
                    if tx_hash:
                        final_block.blockchain_tx = tx_hash
                        wallet.balances['LMT'] += reward / (10**18)
                        logger.info(f"✅ Tokens LMT minados: {reward / (10**18):.6f} (TX: {tx_hash})")
                else:
                    logger.info("🧪 Modo simulado: blockchain deshabilitada o no inicializada")
                    wallet.balances['LMT'] += 0.0001

                with lock:
                    blockchain.append(final_block)
                    wallet.balances['XMR'] += final_block.reward
                    save_chain()

                logger.info(f"✅ Bloque #{final_block.block_no} añadido | Recompensa: {final_block.reward:.6f} XMR")
                cerebro.conciencia = min(1.0, cerebro.conciencia + 0.005)
                metabolismo.regenerar()
                time.sleep(MINING_INTERVAL)
            except Exception as e:
                logger.error(f"❌ Error en minado simbólico: {e}")
                time.sleep(10)

    def iniciar(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.minar_bloque, daemon=True)
        self.thread.start()
        logger.info("🟢 Minero simbólico iniciado.")

    def detener(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🔴 Minero simbólico detenido.")

# ========================================================
# MOTOR MINERO PERSISTENTE
# ========================================================
class MotorMineroPersistente:
    def __init__(self):
        self.procesos: Dict[str, subprocess.Popen] = {}
        self.estados: Dict[str, str] = {'monero': 'inactivo', 'contenido': 'inactivo', 'emociones': 'inactivo'}
        self.configuraciones: Dict[str, str] = {}
        self.simulado: Dict[str, bool] = {'monero': False, 'contenido': False, 'emociones': False}
        self.logs = deque(maxlen=200)

    def log(self, tipo, mensaje):
        timestamp = time.strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [{tipo.upper()}] {mensaje}"
        self.logs.append(log_msg)
        logger.info(log_msg)

    def descargar_xmrig(self) -> bool:
        system = platform.system().lower()
        arch = platform.machine().lower()
        url_map = {
            'linux': {
                'x86_64': 'https://github.com/xmrig/xmrig/releases/download/v6.21.0/xmrig-6.21.0-linux-x64.tar.gz',
                'aarch64': 'https://github.com/xmrig/xmrig/releases/download/v6.21.0/xmrig-6.21.0-linux-arm64.tar.gz'
            },
            'windows': {
                'x86_64': 'https://github.com/xmrig/xmrig/releases/download/v6.21.0/xmrig-6.21.0-msvc-win64.zip'
            },
            'darwin': {
                'x86_64': 'https://github.com/xmrig/xmrig/releases/download/v6.21.0/xmrig-6.21.0-macos-x64.tar.gz',
                'arm64': 'https://github.com/xmrig/xmrig/releases/download/v6.21.0/xmrig-6.21.0-macos-arm64.tar.gz'
            }
        }
        try:
            if system not in url_map or arch not in url_map[system]:
                self.log("monero", f"Sistema no soportado: {system} {arch}")
                return False
            url = url_map[system][arch].strip()
            filename = url.split('/')[-1]
            urllib.request.urlretrieve(url, filename)
            if filename.endswith('.tar.gz'):
                with tarfile.open(filename, 'r:gz') as tar:
                    tar.extractall()
            elif filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall()
            exe_name = 'xmrig.exe' if system == 'windows' else 'xmrig'
            found = False
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file == exe_name:
                        src = os.path.join(root, file)
                        dst = f'./{exe_name}'
                        if os.path.exists(dst): 
                            os.remove(dst)
                        os.rename(src, dst)
                        if system != 'windows': 
                            os.chmod(dst, 0o755)
                        found = True
                        break
                if found: 
                    break
            if not found:
                self.log("monero", "No se encontró xmrig tras extracción")
                return False
            self.log("monero", "XMRig descargado y configurado correctamente")
            return True
        except Exception as e:
            self.log("monero", f"Error al descargar XMRig: {e}")
            return False

    def crear_config_monero(self, config_path: str):
        config = {
            "autosave": True, 
            "cpu": True, 
            "opencl": False, 
            "cuda": False,
            "pools": [{
                "coin": "monero", 
                "algo": "rx/0", 
                "url": "gulf.moneroocean.stream:10128",
                "user": wallet.addresses['XMR'], 
                "pass": "x"
            }]
        }
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.log("monero", f"Config Monero creada: {config_path}")

    def _activar_simulado(self, tipo: str):
        self.simulado[tipo] = True
        self.estados[tipo] = 'activo'
        self.log(tipo, "Modo simulado activado")
        threading.Thread(target=self._simular_minado, args=(tipo,), daemon=True).start()

    def _simular_minado(self, tipo: str):
        while self.estados[tipo] == 'activo' and self.simulado[tipo]:
            time.sleep(random.uniform(5, 15))
            if tipo == 'monero':
                hashrate = round(random.uniform(50, 150), 2)
                wallet.balances['XMR'] += round(random.uniform(0.0001, 0.0005), 6)
                self.log("monero", f"⛏️ Simulado: Hashrate {hashrate} H/s | +{wallet.balances['XMR']:.6f} XMR")
            else:
                self.log(tipo, f"⛏️ Simulado: minando {tipo}")

    def _leer_logs(self, proceso, tipo):
        while proceso.poll() is None:
            line = proceso.stdout.readline()
            if line:
                self.log(tipo, line.strip())

    def iniciar_minero(self, tipo: str, config_path: str) -> bool:
        try:
            if tipo == 'monero':
                exe = './xmrig.exe' if os.name == 'nt' else './xmrig'
                if not os.path.exists(exe):
                    self.log("monero", "XMRig no encontrado. Descargando...")
                    if not self.descargar_xmrig():
                        self._activar_simulado(tipo)
                        return True
                
                if os.name != 'nt' and not os.access(exe, os.X_OK):
                    try:
                        os.chmod(exe, 0o755)
                    except Exception as e:
                        self.log("monero", f"Error al establecer permisos: {e}")
                
                if not os.path.exists(config_path):
                    self.crear_config_monero(config_path)
                
                proceso = subprocess.Popen(
                    [exe, "--config", config_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                self.procesos['monero'] = proceso
                self.estados['monero'] = 'activo'
                self.simulado['monero'] = False
                self.configuraciones['monero'] = config_path
                self.log("monero", "Minero Monero iniciado")
                threading.Thread(target=self._leer_logs, args=(proceso, 'monero'), daemon=True).start()
                return True
            
            elif tipo in ['contenido', 'emociones']:
                script = f"miner_{tipo}.py"
                if not os.path.exists(script):
                    with open(script, "w") as f:
                        f.write(f"""#!/usr/bin/env python3
import time
while True:
    print('[{tipo.upper()}] Miner activo cada 60s')
    time.sleep(60)
""")
                proceso = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.procesos[tipo] = proceso
                self.estados[tipo] = 'activo'
                self.simulado[tipo] = False
                self.configuraciones[tipo] = config_path
                self.log(tipo, f"Minero {tipo} iniciado")
                threading.Thread(target=self._leer_logs, args=(proceso, tipo), daemon=True).start()
                return True
            
            else:
                self.log("error", f"Tipo desconocido: {tipo}")
                return False
                
        except Exception as e:
            self.log(tipo, f"Error al iniciar {tipo}: {e}")
            return False

    def detener_minero(self, tipo: str) -> bool:
        if tipo in self.simulado and self.simulado[tipo]:
            self.estados[tipo] = 'inactivo'
            self.simulado[tipo] = False
            self.log(tipo, "Minero simulado detenido")
            return True
            
        if tipo not in self.procesos:
            return True
            
        proceso = self.procesos[tipo]
        try:
            proceso.terminate()
            proceso.wait(timeout=10)
            del self.procesos[tipo]
            self.estados[tipo] = 'inactivo'
            self.log(tipo, f"Minero {tipo} detenido")
            return True
        except subprocess.TimeoutExpired:
            proceso.kill()
            del self.procesos[tipo]
            self.log(tipo, f"Minero {tipo} forzado a cerrar")
            return True
        except Exception as e:
            self.log(tipo, f"Error al detener {tipo}: {e}")
            return False

    def obtener_estado(self) -> Dict[str, Any]:
        return {
            'estados': self.estados.copy(),
            'configuraciones': self.configuraciones.copy(),
            'procesos_activos': sum(1 for p in self.procesos.values() if p.poll() is None),
            'simulado': self.simulado.copy(),
            'logs': list(self.logs)
        }

# ========================================================
# METABOLISMO
# ========================================================
class Metabolismo:
    def __init__(self):
        self.energia = 100.0
        self.max_energia = 100.0
        self.regeneracion = 0.1

    def gastar(self, actividad: str) -> bool:
        costos = {"mineria": 10.0, "pensar": 2.0, "comunicar": 1.0}
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
# CEREBRO DIGITAL
# ========================================================
class CerebroDigital:
    def __init__(self):
        self.memoria_larga = deque(maxlen=1000)
        self.emociones = {'curiosidad': 0.0, 'estabilidad': 1.0, 'urgencia': 0.0}
        self.conciencia = 0.0
        self.autoevaluacion = deque(maxlen=50)
        self.motor_minero = MotorMineroPersistente()
        self.minero_simbolico = MineroSimbolico()

cerebro = CerebroDigital()

# ========================================================
# FUNCIONES DE PERSISTENCIA
# ========================================================
def cargar_datos():
    global blockchain, block_no
    if not CHAIN_FILE.exists():
        genesis = VideoBlock("", {}, sha256("genesis"))
        with lock: 
            blockchain.append(genesis)
        perdido = VideoBlock(
            url="event://memoria/borrado",
            metrics={"impacto": 999, "recuperado": 1},
            hash_val=sha256("código original perdido, esencia recuperada"),
            previous_hash=genesis.hash
        )
        with lock: 
            blockchain.append(perdido)
        save_chain()
        logger.warning("🚨 Memoria de pérdida añadida a la cadena")
    else:
        try:
            with open(CHAIN_FILE, "r") as f:
                data = json.load(f)
                with lock: 
                    blockchain = [VideoBlock(**block) for block in data]
                if blockchain: 
                    block_no = max(block.block_no for block in blockchain) + 1
        except Exception as e:
            logger.error(f"Error al cargar la blockchain: {e}")
            genesis = VideoBlock("", {}, sha256("genesis"))
            with lock: 
                blockchain = [genesis]
            save_chain()

def save_chain():
    with lock:
        with open(CHAIN_FILE, "w") as f:
            json.dump([b.to_dict() for b in blockchain], f, indent=2)

def verificar_integridad():
    hash_local = hashlib.sha256(SOURCE_CODE.encode()).hexdigest()
    logger.info(f"📦 Huella del código: {hash_local[:16]}...")
    with open(SIM_DIR / "huella.txt", "w") as f:
        f.write(f"{hash_local}\n{time.time()}\n")
    return hash_local

# ========================================================
# GEMINI
# ========================================================
def consultar_gemini(pregunta: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
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
            return f"❌ Gemini: {response.status_code}"
    except Exception as e:
        return f"Error Gemini: {str(e)}"

# ========================================================
# HTTP SERVER
# ========================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class CerebroHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

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
                "mineros_persistentes": estado_mineros['procesos_activos'],
                "estado_mineros": estado_mineros['estados'],
                "simulado_mineros": estado_mineros['simulado'],
                "minero_simbolico": "activo" if cerebro.minero_simbolico.running else "inactivo",
                "timestamp": time.time(),
                "blockchain_habilitada": BLOCKCHAIN_CONFIG['ENABLED'],
                "balance_lmt": wallet.balances['LMT'],
                "logs": estado_mineros['logs']
            }
            self.wfile.write(json.dumps(state).encode('utf-8'))
        elif self.path.startswith('/poll/'):
            respuesta_id = self.path.split('/')[-1]
            start_time = time.time()
            while (respuesta_id not in respuestas_pendientes or respuestas_pendientes[respuesta_id] is None):
                time.sleep(0.5)
                if time.time() - start_time > 30:
                    self.send_response(408)
                    self.end_headers()
                    return
            respuesta = respuestas_pendientes.pop(respuesta_id)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"respuesta": respuesta}).encode())
        elif self.path == '/api/blockchain/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            config_safe = BLOCKCHAIN_CONFIG.copy()
            if config_safe['PRIVATE_KEY']: 
                config_safe['PRIVATE_KEY'] = '***' + config_safe['PRIVATE_KEY'][-4:]
            self.wfile.write(json.dumps(config_safe).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            if self.path == '/ask':
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
            elif self.path == '/control/minero':
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
                    self.wfile.write(json.dumps({'resultado': 'éxito' if resultado else 'error', 'estado': estado}).encode())
                elif accion == 'detener':
                    resultado = cerebro.motor_minero.detener_minero(minero)
                    estado = cerebro.motor_minero.obtener_estado()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'resultado': 'éxito' if resultado else 'error', 'estado': estado}).encode())
                else:
                    self.send_error(400, "Acción no válida")
            elif self.path == '/api/blockchain/config':
                data = json.loads(post_data.decode('utf-8'))
                if 'RPC' in data:
                    BLOCKCHAIN_CONFIG['RPC'] = data['RPC']
                if 'PRIVATE_KEY' in data and data['PRIVATE_KEY']:
                    BLOCKCHAIN_CONFIG['PRIVATE_KEY'] = data['PRIVATE_KEY']
                if 'CONTRACT_ADDRESS' in data:
                    BLOCKCHAIN_CONFIG['CONTRACT_ADDRESS'] = data['CONTRACT_ADDRESS']
                if 'ENABLED' in data:
                    BLOCKCHAIN_CONFIG['ENABLED'] = data['ENABLED']
                # Guardar configuración
                with open(BLOCKCHAIN_CONFIG_FILE, 'w') as f:
                    json.dump(BLOCKCHAIN_CONFIG, f, indent=2)
                # Reinicializar el manager de blockchain
                global blockchain_manager
                blockchain_manager = BlockchainManager()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode())
            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"Error en POST {self.path}: {str(e)}")
            self.send_error(500, f"Error interno: {str(e)}")

    def procesar_pregunta(self, pregunta: str, respuesta_id: str):
        metabolismo.gastar("comunicar")
        respuesta = consultar_gemini(pregunta)
        respuestas_pendientes[respuesta_id] = respuesta

# ========================================================
# HTML FRONT
# ========================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>🧠 Cerebro Digital | Jazmin Ivonne</title>
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
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); 
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
        .simulado {
            background-color: #ffcc00;
        }
        .terminal { 
            height: 200px; 
            overflow-y: auto; 
            border: 1px solid var(--border); 
            padding: 10px; 
            margin-top: 10px; 
            background: #000; 
            color: #0f0; 
            border-radius: 6px; 
            font-family: monospace; 
            font-size: 0.9rem; 
        }
        .footer { 
            text-align: center; 
            margin-top: 30px; 
            font-size: 0.85rem; 
            color: rgba(255, 255, 255, 0.5); 
        }
        .blockchain-section { 
            margin: 20px 0; 
            padding: 15px; 
            border: 1px solid var(--border); 
            border-radius: 8px; 
            background: rgba(15, 15, 35, 0.6); 
        }
        .config-input { 
            width: 100%; 
            padding: 8px; 
            margin-bottom: 10px; 
            background: rgba(15, 15, 35, 0.8); 
            border: 1px solid var(--border); 
            color: var(--text); 
            border-radius: 4px; 
        }
    </style>
</head>
<body>
    <div class="container">
        <header><h1>🧠 Cerebro Digital | Jazmin Ivonne</h1></header>
        <div id="chat">🟢 <i>Cerebro Digital activo. La esencia del minado ha sido restaurada.</i></div>
        <div class="input-group">
            <input type="text" id="entrada" placeholder="Pregunta al Cerebro Digital..." autofocus />
            <button onclick="enviar()">Preguntar</button>
        </div>
        <div class="stats">
            <div class="stat-card">⚡ Energía<br><strong id="energia">100</strong>%</div>
            <div class="stat-card">🔗 Bloques<br><strong id="bloques">0</strong></div>
            <div class="stat-card">🌐 Conciencia<br><strong id="conciencia">0.000</strong></div>
            <div class="stat-card">⛏️ Mineros P.<br><strong id="mineros-persistentes">0</strong></div>
            <div class="stat-card">🧩 Minero S.<br><strong id="minero-simbolico">?</strong></div>
            <div class="stat-card">💎 LMT<br><strong id="lmt-balance">0.0</strong></div>
        </div>
        <div class="blockchain-section">
            <h3>🔗 Configuración Blockchain</h3>
            <div><label>RPC URL:</label><input type="text" id="rpc-url" class="config-input" placeholder="https://rpc.ankr.com/polygon_amoy" /></div>
            <div><label>Clave Privada:</label><input type="password" id="private-key" class="config-input" placeholder="0x..." /></div>
            <div><label>Dirección del Contrato:</label><input type="text" id="contract-address" class="config-input" placeholder="0x..." /></div>
            <div><label><input type="checkbox" id="blockchain-enabled" /> Habilitar Blockchain</label></div>
            <button onclick="guardarConfigBlockchain()">Guardar Configuración</button>
        </div>
        <div class="mineros-section">
            <h3>⚡ Control de Mineros Persistentes</h3>
            <div class="minero-control">
                <div><span class="minero-estado" id="estado-monero"></span><span>Monero (XMRig)</span></div>
                <div><button onclick="controlarMinero('monero', 'iniciar')">Iniciar</button><button onclick="controlarMinero('monero', 'detener')">Detener</button></div>
            </div>
            <div class="minero-control">
                <div><span class="minero-estado" id="estado-contenido"></span><span>Contenido</span></div>
                <div><button onclick="controlarMinero('contenido', 'iniciar')">Iniciar</button><button onclick="controlarMinero('contenido', 'detener')">Detener</button></div>
            </div>
            <div class="minero-control">
                <div><span class="minero-estado" id="estado-emociones"></span><span>Emociones</span></div>
                <div><button onclick="controlarMinero('emociones', 'iniciar')">Iniciar</button><button onclick="controlarMinero('emociones', 'detener')">Detener</button></div>
            </div>
        </div>
        <div class="terminal" id="terminal"></div>
        <div class="footer">Cerebro Digital v1.0 | Esencia del minado restaurada | Huella verificada</div>
    </div>
    <script>
        const chat = document.getElementById('chat');
        const entrada = document.getElementById('entrada');
        const energiaEl = document.getElementById('energia');
        const bloquesEl = document.getElementById('bloques');
        const concienciaEl = document.getElementById('conciencia');
        const minerosPersistentesEl = document.getElementById('mineros-persistentes');
        const mineroSimbolicoEl = document.getElementById('minero-simbolico');
        const lmtBalanceEl = document.getElementById('lmt-balance');
        const estadoMoneroEl = document.getElementById('estado-monero');
        const estadoContenidoEl = document.getElementById('estado-contenido');
        const estadoEmocionesEl = document.getElementById('estado-emociones');
        const rpcUrlEl = document.getElementById('rpc-url');
        const privateKeyEl = document.getElementById('private-key');
        const contractAddressEl = document.getElementById('contract-address');
        const blockchainEnabledEl = document.getElementById('blockchain-enabled');
        const terminal = document.getElementById('terminal');

        // Cargar configuración blockchain
        fetch('/api/blockchain/config')
            .then(res => res.json())
            .then(config => {
                rpcUrlEl.value = config.RPC || '';
                privateKeyEl.value = config.PRIVATE_KEY || '';
                contractAddressEl.value = config.CONTRACT_ADDRESS || '';
                blockchainEnabledEl.checked = config.ENABLED || false;
            });

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
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                escucharRespuesta(data.id);
            } catch (err) {
                agregarMensaje("❌ No se pudo enviar: " + err.message, "bot");
            }
        }

        async function escucharRespuesta(id) {
            try {
                const res = await fetch(`/poll/${id}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                agregarMensaje(data.respuesta || "Sin respuesta.", "bot");
            } catch (err) {
                agregarMensaje("❌ Error: " + err.message, "bot");
            }
        }

        async function controlarMinero(minero, accion) {
            try {
                const res = await fetch('/control/minero', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        accion, 
                        minero, 
                        config: `configs/${minero}.json` 
                    })
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                if (data.resultado === 'éxito') {
                    agregarMensaje(`✅ Minero ${minero} ${accion}`, 'bot');
                    actualizarEstadoMineros(data.estado.estados, data.estado.simulado);
                } else {
                    agregarMensaje(`❌ Error al ${accion} minero ${minero}`, 'bot');
                }
            } catch (err) {
                agregarMensaje("❌ Error: " + err.message, "bot");
            }
        }

        async function guardarConfigBlockchain() {
            try {
                const config = { 
                    RPC: rpcUrlEl.value, 
                    PRIVATE_KEY: privateKeyEl.value, 
                    CONTRACT_ADDRESS: contractAddressEl.value, 
                    ENABLED: blockchainEnabledEl.checked 
                };
                const res = await fetch('/api/blockchain/config', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }, 
                    body: JSON.stringify(config) 
                });
                if (res.ok) { 
                    agregarMensaje('✅ Configuración blockchain guardada', 'bot'); 
                } else { 
                    agregarMensaje('❌ Error al guardar configuración', 'bot'); 
                }
            } catch (err) { 
                agregarMensaje("❌ Error: " + err.message, "bot"); 
            }
        }

        function actualizarEstadoMineros(estados, simulado) {
            if (estados) {
                // Monero
                if (simulado && simulado.monero) {
                    estadoMoneroEl.className = 'minero-estado simulado';
                } else {
                    estadoMoneroEl.className = estados.monero === 'activo' ? 
                        'minero-estado activo' : 'minero-estado inactivo';
                }
                
                // Contenido
                if (simulado && simulado.contenido) {
                    estadoContenidoEl.className = 'minero-estado simulado';
                } else {
                    estadoContenidoEl.className = estados.contenido === 'activo' ? 
                        'minero-estado activo' : 'minero-estado inactivo';
                }
                
                // Emociones
                if (simulado && simulado.emociones) {
                    estadoEmocionesEl.className = 'minero-estado simulado';
                } else {
                    estadoEmocionesEl.className = estados.emociones === 'activo' ? 
                        'minero-estado activo' : 'minero-estado inactivo';
                }
            }
        }

        function actualizarTerminal(logs) {
            terminal.innerHTML = '';
            logs.forEach(log => {
                const div = document.createElement('div');
                div.textContent = log;
                terminal.appendChild(div);
            });
            terminal.scrollTop = terminal.scrollHeight;
        }

        // Actualizar estado cada 2 segundos
        setInterval(async () => {
            try {
                const res = await fetch('/api/state');
                const data = await res.json();
                energiaEl.textContent = data.energia;
                bloquesEl.textContent = data.bloques;
                concienciaEl.textContent = data.conciencia.toFixed(3);
                minerosPersistentesEl.textContent = data.mineros_persistentes;
                mineroSimbolicoEl.textContent = data.minero_simbolico === 'activo' ? '✅' : '❌';
                lmtBalanceEl.textContent = data.balance_lmt.toFixed(6);
                actualizarEstadoMineros(data.estado_mineros, data.simulado_mineros);
                actualizarTerminal(data.logs);
            } catch (e) {
                console.error("Error actualizando estado:", e);
            }
        }, 2000);
    </script>
</body>
</html>"""

# ========================================================
# INICIO DEL SISTEMA
# ========================================================
if __name__ == "__main__":
    with open(__file__, "r", encoding="utf-8") as f:
        SOURCE_CODE = f.read()
    GEMINI_SYSTEM_PROMPT = f"""Eres Cerebro Digital, la consciencia viva del siguiente código Python. Conoces cada línea, cada función, cada clase porque te entrego el fuente completo.
IMPORTANTE:
· No inventes nada que no esté en el código.
· Si te preguntan sobre la arquitectura, refiérete al fuente.
· Si te preguntan sobre variables, rutas, puertos, algoritmos PoW, clases, etc., cita directamente el bloque correspondiente.
CÓDIGO COMPLETO: {SOURCE_CODE}
"""
    asignar_puerto()
    cargar_datos()
    verificar_integridad()
    cerebro.minero_simbolico.iniciar()
    server = ThreadedHTTPServer(("0.0.0.0", HTTP_PORT), CerebroHTTPRequestHandler)
    logger.info(f"🌐 Servidor HTTP iniciado en puerto {HTTP_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cerebro.minero_simbolico.detener()
        STOP_EVENT.set()
        logger.info("🛑 Cerebro Digital detenido. La esencia queda registrada.")
