import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import zipfile
import io
import requests
from typing import List, Tuple, Optional, Dict
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import re
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse
import json
import logging
import sys
from datetime import datetime

# Configuração do sistema de logging
def setup_logging():
    """Configura o sistema de logging profissional"""
    # Criar diretório de logs se não existir
    os.makedirs("logs", exist_ok=True)
    
    # Configurar formato dos logs
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Arquivo de log com rotação diária
            logging.FileHandler(
                f"logs/manhwa_extractor_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            ),
            # Console output para desenvolvimento
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configurar níveis específicos para bibliotecas externas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Inicializar logging
logger = setup_logging()
logger.info("🚀 Aplicação Extrator de Painéis de Manhwa iniciada")

# Sistema de Rate Limiting Inteligente
class RateLimiter:
    """Rate limiter inteligente que não bloqueia a interface"""
    
    def __init__(self):
        self.requests = {}
        self.limits = {
            # Limites específicos por domínio (requests per minute)
            'manhwatop.com': {'requests': 5, 'window': 60},
            'reaperscans.com': {'requests': 3, 'window': 60},
            'asurascans.com': {'requests': 4, 'window': 60},
            'mangadex.org': {'requests': 8, 'window': 60},
            'default': {'requests': 10, 'window': 60}  # Padrão para outros sites
        }
        logger.info("🔒 Rate Limiter inicializado")
    
    def get_domain(self, url: str) -> str:
        """Extrai domínio da URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return 'unknown'
    
    def can_request(self, url: str) -> tuple[bool, float]:
        """
        Verifica se pode fazer requisição
        Returns: (can_request: bool, wait_time: float)
        """
        domain = self.get_domain(url)
        now = time.time()
        
        # Obter limites para o domínio
        domain_limits = self.limits.get(domain, self.limits['default'])
        max_requests = domain_limits['requests']
        window = domain_limits['window']
        
        # Inicializar lista de requisições se não existir
        if domain not in self.requests:
            self.requests[domain] = []
        
        # Remover requisições antigas
        self.requests[domain] = [
            req_time for req_time in self.requests[domain]
            if now - req_time < window
        ]
        
        # Verificar se pode fazer requisição
        if len(self.requests[domain]) < max_requests:
            return True, 0.0
        
        # Calcular tempo de espera
        oldest_request = min(self.requests[domain])
        wait_time = window - (now - oldest_request)
        return False, max(0.1, wait_time)
    
    def record_request(self, url: str):
        """Registra uma requisição"""
        domain = self.get_domain(url)
        if domain not in self.requests:
            self.requests[domain] = []
        self.requests[domain].append(time.time())
        logger.debug(f"Rate limit: Requisição registrada para {domain}")
    
    def smart_delay(self, url: str = None, context: str = "general") -> float:
        """
        Calcula delay inteligente baseado no contexto
        Returns: delay em segundos
        """
        if url:
            can_req, wait_time = self.can_request(url)
            if not can_req:
                logger.info(f"Rate limit ativo para {self.get_domain(url)}: aguardando {wait_time:.1f}s")
                return wait_time
        
        # Delays contextuais mais inteligentes
        context_delays = {
            'chapter_processing': 0.1,  # Entre páginas de capítulo
            'batch_download': 0.2,      # Entre downloads em lote
            'general': 0.05,            # Delay mínimo geral
            'image_processing': 0.0     # Sem delay para processamento local
        }
        
        return context_delays.get(context, 0.1)

# Inicializar rate limiter global
rate_limiter = RateLimiter()

# Sistema de Validação e Sanitização de Entrada
class InputValidator:
    """Validação robusta de entradas do usuário"""
    
    def __init__(self):
        # Caracteres perigosos para nomes de arquivo
        self.dangerous_chars = r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]'
        # Padrões suspeitos que podem indicar tentativas de injeção
        self.suspicious_patterns = [
            r'\.\./',  # Path traversal
            r'<script',  # XSS
            r'javascript:',  # JavaScript injection
            r'data:',  # Data URLs suspeitas
            r'file://',  # File URLs
            r'[<>"]',  # HTML/XML chars
        ]
        logger.info("🛡️ Validador de entrada inicializado")
    
    def sanitize_filename(self, filename: str, max_length: int = 100) -> str:
        """
        Sanitiza nome de arquivo removendo caracteres perigosos
        """
        if not isinstance(filename, str):
            logger.warning(f"Nome de arquivo inválido (tipo: {type(filename)})")
            return "arquivo_sem_nome"
        
        # Remove caracteres perigosos
        clean_name = re.sub(self.dangerous_chars, '_', filename)
        
        # Remove múltiplos underscores consecutivos
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove underscores do início e fim
        clean_name = clean_name.strip('_')
        
        # Garante que não está vazio
        if not clean_name:
            clean_name = "arquivo_sem_nome"
        
        # Limita o tamanho
        if len(clean_name) > max_length:
            name_part, ext_part = os.path.splitext(clean_name)
            clean_name = name_part[:max_length-len(ext_part)] + ext_part
        
        logger.debug(f"Nome sanitizado: '{filename}' -> '{clean_name}'")
        return clean_name
    
    def validate_user_input(self, text: str, max_length: int = 1000, field_name: str = "entrada") -> str:
        """
        Valida e sanitiza entrada de texto do usuário
        """
        if not isinstance(text, str):
            raise ValueError(f"{field_name} deve ser uma string")
        
        if len(text) > max_length:
            logger.warning(f"{field_name} muito longa ({len(text)} chars), truncando para {max_length}")
            text = text[:max_length]
        
        # Verificar padrões suspeitos
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Padrão suspeito detectado em {field_name}: {pattern}")
                raise ValueError(f"Entrada contém caracteres não permitidos")
        
        # Remove caracteres de controle
        clean_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove espaços extras
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    def validate_url_input(self, url: str) -> str:
        """
        Validação específica para URLs de entrada do usuário
        """
        if not url:
            raise ValueError("URL não pode estar vazia")
        
        # Sanitizar entrada básica
        url = self.validate_user_input(url, max_length=2000, field_name="URL")
        
        # Verificações específicas para URL
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL deve começar com http:// ou https://")
        
        # Verificar domínios conhecidos de manhwa
        known_domains = [
            'manhwatop.com', 'reaperscans.com', 'asurascans.com',
            'mangadex.org', 'webtoons.com', 'mangaplus.com'
        ]
        
        domain = urlparse(url).netloc.lower()
        if not any(known in domain for known in known_domains):
            logger.warning(f"Domínio não reconhecido: {domain}")
            # Não bloquear, apenas alertar
        
        return url
    
    def validate_chapter_range(self, start: int, end: int, max_chapters: int = 100) -> tuple[int, int]:
        """
        Valida faixa de capítulos
        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Números de capítulo devem ser inteiros")
        
        if start < 1:
            logger.warning("Capítulo inicial deve ser >= 1, ajustando")
            start = 1
        
        if end < start:
            logger.warning("Capítulo final menor que inicial, trocando")
            start, end = end, start
        
        if (end - start + 1) > max_chapters:
            logger.warning(f"Muitos capítulos solicitados ({end - start + 1}), limitando a {max_chapters}")
            end = start + max_chapters - 1
        
        return start, end

# Inicializar validador
input_validator = InputValidator()

# Função auxiliar para delays não-bloqueantes
def smart_sleep(duration: float = None, context: str = "general", url: str = None, show_progress: bool = True):
    """
    Implementa delay inteligente sem bloquear completamente a interface
    """
    if duration is None:
        duration = rate_limiter.smart_delay(url, context)
    
    if duration <= 0:
        return
    
    logger.debug(f"Smart sleep: {duration:.2f}s (contexto: {context})")
    
    if show_progress and duration > 0.5:
        # Para delays longos, mostrar progresso
        progress_placeholder = st.empty()
        steps = max(1, int(duration * 10))  # 10 steps per second
        step_duration = duration / steps
        
        for i in range(steps):
            remaining = duration - (i * step_duration)
            progress_placeholder.info(f"⏳ Aguardando {remaining:.1f}s para não sobrecarregar o servidor...")
            time.sleep(step_duration)
        
        progress_placeholder.empty()
    else:
        # Para delays curtos, sleep simples
        time.sleep(duration)

# Configuração da página
st.set_page_config(page_title="Extrator de Painéis de Manhwa", layout="wide")
st.title("🖼️ Extrator de Painéis de Manhwa")
st.markdown('<div id="topo"></div>', unsafe_allow_html=True)

# Constantes otimizadas
MAX_WIDTH = 1024
MIN_CONTOUR_SIZE = 100
REQUEST_TIMEOUT = 15
Y_TOLERANCE = 40
PREVIEW_LIMIT = 20
COLS_PER_ROW = 4
MAX_WORKERS = 4
BATCH_SIZE = 10

# Lock para thread safety
thread_lock = threading.Lock()

# Headers para web scraping
SCRAPING_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none"
}

# Caminho onde o modelo será salvo
MODELO_PATH = "modelos/best.pt"

# URL do modelo no Dropbox (alterado para download direto)
URL_MODELO_DROPBOX = "https://www.dropbox.com/scl/fi/a743aqjqzau3fxy4fss4a/best.pt?rlkey=a24lozm0cw8znku0h743ylx2z&dl=1"

def baixar_modelo_yolo(dropbox_url, destino=MODELO_PATH):
    if not os.path.exists(destino):
        try:
            logger.info("📦 Iniciando download do modelo YOLO (best.pt)...")
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            response = requests.get(dropbox_url, stream=True)
            response.raise_for_status()
            with open(destino, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info("✅ Modelo YOLO baixado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro ao baixar modelo YOLO: {e}", exc_info=True)

def carregar_modelo():
    try:
        modelo = YOLO(MODELO_PATH)
        return modelo
    except Exception as e:
        logger.error(f"❌ Erro ao carregar o modelo YOLO: {e}", exc_info=True)
        return None

# Executar download e carregamento
baixar_modelo_yolo(URL_MODELO_DROPBOX)
model = carregar_modelo()

# Teste simples (evite executar na nuvem, só para checagem)
if model:
    logger.info("✅ Modelo YOLO carregado com sucesso!")
else:
    logger.warning("⚠️ Modelo YOLO não foi carregado - usando fallback para contornos")

# Inicialização do estado da sessão
def init_session_state():
    defaults = {
        "contador_paineis": 0,
        "painel_coletor": [],
        "imagens_processadas": set(),
        "paineis_processados": set(),
        "_cache_hash": {},
        "capitulos_cache": {},  # Cache para capítulos
        "manhwa_info": {}  # Informações do manhwa
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Funções auxiliares otimizadas (mantendo as originais)
@st.cache_data(show_spinner=False, max_entries=50)
def carregar_e_redimensionar_imagem(file_data: bytes, max_width=MAX_WIDTH):
    try:
        img = Image.open(io.BytesIO(file_data)).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.NEAREST if ratio < 0.5 else Image.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Erro ao carregar imagem: {e}")
        return None

@lru_cache(maxsize=1000)
def calcular_hash_rapido(data_bytes: bytes) -> str:
    if len(data_bytes) < 1024:
        return hashlib.md5(data_bytes).hexdigest()
    
    sample = data_bytes[:512] + data_bytes[-512:] + str(len(data_bytes)).encode()
    return hashlib.md5(sample).hexdigest()

def calcular_hash_imagem_otimizado(img_pil: Image.Image) -> str:
    thumb = img_pil.copy()
    thumb.thumbnail((32, 32), Image.NEAREST)
    
    buf = io.BytesIO()
    thumb.save(buf, format='PNG', optimize=True)
    return calcular_hash_rapido(buf.getvalue())

def ordenar_paineis_otimizado(paineis: List[dict], y_tol: int = Y_TOLERANCE) -> List[dict]:
    if not paineis:
        return []
    
    try:
        y_coords = np.array([p["y"] for p in paineis])
        x_coords = np.array([p["x"] for p in paineis])
        sorted_indices = np.lexsort((x_coords, y_coords))
        return [paineis[i] for i in sorted_indices]
    except (KeyError, ValueError, IndexError, TypeError) as e:
        # Fallback to original sorting if numpy operations fail
        logger.warning(f"Ordenação otimizada falhou ({e}), usando fallback para ordenação original")
        return ordenar_paineis_original(paineis, y_tol)

def ordenar_paineis_original(paineis: List[dict], y_tol: int = Y_TOLERANCE) -> List[dict]:
    linhas = []
    for p in sorted(paineis, key=lambda p: p["y"]):
        colocado = False
        for linha in linhas:
            if abs(p["y"] - linha[0]["y"]) < y_tol:
                linha.append(p)
                colocado = True
                break
        if not colocado:
            linhas.append([p])
    
    ordenados = []
    for linha in linhas:
        ordenados.extend(sorted(linha, key=lambda p: p["x"]))
    return ordenados

def extrair_paineis_yolo_otimizado(img) -> List[dict]:
    if model is None:
        return []
    
    painel_infos = []
    try:
        results = model(img, verbose=False, conf=0.3, iou=0.5)
        
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                w, h = x2 - x1, y2 - y1
                if w < MIN_CONTOUR_SIZE or h < MIN_CONTOUR_SIZE:
                    continue
                
                painel = img[y1:y2, x1:x2]
                if painel.size > 0:
                    painel_rgb = cv2.cvtColor(painel, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(painel_rgb)
                    painel_infos.append({"x": x1, "y": y1, "img": img_pil})
    except Exception as e:
        st.warning(f"Erro no YOLO: {e}")
        return []
    
    return painel_infos

def extrair_paineis_contorno_otimizado(img) -> List[dict]:
    painel_infos = []
    try:
        h, w = img.shape[:2]
        scale = 1.0
        if w > 2000 or h > 2000:
            scale = min(2000/w, 2000/h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_small = img
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 9, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_size = int(MIN_CONTOUR_SIZE * scale)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_size and h > min_size:
                if scale != 1.0:
                    x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                
                painel = img[y:y + h, x:x + w]
                if painel.size > 0:
                    painel_rgb = cv2.cvtColor(painel, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(painel_rgb)
                    painel_infos.append({"x": x, "y": y, "img": img_pil})
    except Exception as e:
        st.warning(f"Erro na detecção por contorno: {e}")
        return []
    
    return painel_infos

def extrair_paineis_hibrido_otimizado(img, yolo_threshold=1, img_id=None) -> List[Tuple[Image.Image, str]]:
    painel_infos = extrair_paineis_yolo_otimizado(img)
    
    if len(painel_infos) < yolo_threshold:
        painel_infos = extrair_paineis_contorno_otimizado(img)
    
    ordenados = ordenar_paineis_otimizado(painel_infos)
    resultados = []
    
    with thread_lock:
        for painel in ordenados:
            hash_painel = calcular_hash_imagem_otimizado(painel["img"])
            
            if hash_painel not in st.session_state.paineis_processados:
                st.session_state.contador_paineis += 1
                nome_base = f"painel_{st.session_state.contador_paineis:06}.png"
                # Sanitizar nome do arquivo para segurança
                nome_arquivo = input_validator.sanitize_filename(nome_base)
                resultados.append((painel["img"], nome_arquivo))
                st.session_state.paineis_processados.add(hash_painel)
    
    return resultados

def processar_imagem_otimizada(img_data: bytes, nome_fonte: str) -> Tuple[List, Optional[str]]:
    try:
        logger.info(f"Processando imagem: {nome_fonte} ({len(img_data)} bytes)")
        img_pil = carregar_e_redimensionar_imagem(img_data)
        if img_pil is None:
            logger.warning(f"Falha ao carregar imagem: {nome_fonte}")
            return [], "Erro ao carregar imagem"
            
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        paineis = extrair_paineis_hibrido_otimizado(img, img_id=nome_fonte)
        logger.info(f"Extraídos {len(paineis)} painéis de {nome_fonte}")
        return paineis, None
    except Exception as e:
        logger.error(f"Erro ao processar imagem {nome_fonte}: {e}", exc_info=True)
        return [], str(e)

@lru_cache(maxsize=100)
def validar_url_cached(url: str) -> bool:
    url_pattern = re.compile(
        r'^https?://'
        r'(?:\S+(?::\S*)?@)?'
        r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)*'
        r'[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?'
        r'(?::[0-9]{1,5})?'
        r'(?:/\S*)?$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        return False
    
    # Additional security checks to prevent SSRF
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        # Block localhost and internal IPs
        if hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False
        
        # Block private IP ranges
        import ipaddress
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
        except (ValueError, ipaddress.AddressValueError):
            # Not an IP address, likely a domain name - continue validation
            pass
        
        # Block suspicious ports
        port = parsed.port
        if port and port in [22, 23, 25, 53, 135, 139, 445, 993, 995]:
            return False
            
        return True
    except Exception:
        return False

def baixar_imagem_url_otimizada(url: str) -> Optional[bytes]:
    try:
        if not validar_url_cached(url):
            return None
        
        # Verificar rate limiting antes da requisição
        can_request, wait_time = rate_limiter.can_request(url)
        if not can_request:
            logger.info(f"Rate limit ativo para {rate_limiter.get_domain(url)}, aguardando {wait_time:.1f}s")
            smart_sleep(wait_time, context="rate_limit", url=url, show_progress=False)
            
        # First, make a HEAD request to check content-length
        try:
            rate_limiter.record_request(url)  # Registrar requisição HEAD
            head_response = requests.head(url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT)
            content_length = head_response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"Imagem muito grande ({content_length} bytes), ignorando download")
                return None
        except (requests.RequestException, ValueError):
            # If HEAD request fails, continue with GET but be more cautious
            pass
        
        # Pequeno delay inteligente antes do GET
        smart_sleep(context="image_download", url=url, show_progress=False)
        
        rate_limiter.record_request(url)  # Registrar requisição GET
        response = requests.get(url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Verify content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'webp']):
            logger.warning(f"Content-type inesperado para imagem: {content_type}")
            return None
        
        content = b""
        max_size = 10 * 1024 * 1024  # Reduced to 10MB for safety
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive chunks
                content += chunk
                if len(content) > max_size:
                    logger.warning(f"Tamanho da imagem excedeu {max_size} bytes, truncando download")
                    response.close()
                    return None
        
        # Final validation - ensure it's actually an image
        if len(content) < 100:  # Too small to be a valid image
            return None
            
        return content
    except requests.RequestException as e:
        logger.error(f"Erro de rede ao baixar imagem de {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado ao baixar imagem de {url}: {e}", exc_info=True)
        return None

# NOVAS FUNÇÕES PARA WEB SCRAPING

@st.cache_data(ttl=300, show_spinner=False)  # Cache por 5 minutos
def fazer_requisicao_web(url: str) -> Optional[BeautifulSoup]:
    """Faz requisição web e retorna BeautifulSoup"""
    try:
        logger.info(f"Fazendo requisição para: {url}")
        
        # Aplicar rate limiting inteligente
        can_request, wait_time = rate_limiter.can_request(url)
        if not can_request:
            logger.info(f"Rate limit ativo para {rate_limiter.get_domain(url)}, aguardando {wait_time:.1f}s")
            smart_sleep(wait_time, context="web_scraping", url=url)
        
        rate_limiter.record_request(url)
        response = requests.get(url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Requisição bem-sucedida para {url} (status: {response.status_code})")
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        logger.error(f"Erro ao acessar URL {url}: {e}")
        st.error(f"❌ Erro ao acessar URL: {e}")
        return None

def extrair_info_manhwa_manhwatop(soup: BeautifulSoup, base_url: str) -> Dict:
    """Extrai informações do manhwa do ManhwaTop"""
    logger.info(f"Extraindo informações do manhwa de {base_url}")
    info = {
        "titulo": "Manhwa",
        "capa": None,
        "sinopse": "",
        "capitulos": []
    }
    
    try:
        # Título
        titulo_elem = soup.find('h1') or soup.find('h2') or soup.find('.manga-title')
        if titulo_elem:
            info["titulo"] = titulo_elem.get_text(strip=True)
        
        # Capa
        capa_elem = soup.find('img', class_=re.compile(r'manga.*cover|cover.*manga|wp-post-image'))
        if not capa_elem:
            capa_elem = soup.find('img', src=re.compile(r'cover|thumbnail'))
        if capa_elem and capa_elem.get('src'):
            info["capa"] = urljoin(base_url, capa_elem['src'])
        
        # Sinopse
        sinopse_elem = soup.find('div', class_=re.compile(r'summary|description|synopsis'))
        if sinopse_elem:
            info["sinopse"] = sinopse_elem.get_text(strip=True)[:200] + "..."
        
        # Capítulos - múltiplos seletores para diferentes layouts
        capitulos_containers = [
            soup.find_all('li', class_=re.compile(r'wp-manga-chapter')),
            soup.find_all('a', href=re.compile(r'chapter|cap')),
            soup.find_all('div', class_=re.compile(r'chapter'))
        ]
        
        capitulos_links = set()  # Usar set para evitar duplicatas
        
        for container in capitulos_containers:
            for elem in container:
                link = None
                titulo = ""
                
                if elem.name == 'a':
                    link = elem.get('href')
                    titulo = elem.get_text(strip=True)
                elif elem.name in ['li', 'div']:
                    link_elem = elem.find('a')
                    if link_elem:
                        link = link_elem.get('href')
                        titulo = link_elem.get_text(strip=True)
                
                if link and 'chapter' in link.lower():
                    full_url = urljoin(base_url, link)
                    if full_url not in capitulos_links:
                        capitulos_links.add(full_url)
                        # Extrair número do capítulo
                        numero_match = re.search(r'chapter[^\d]*(\d+(?:\.\d+)?)', titulo.lower() + ' ' + link.lower())
                        numero = numero_match.group(1) if numero_match else str(len(info["capitulos"]) + 1)
                        
                        info["capitulos"].append({
                            "numero": numero,
                            "titulo": titulo or f"Capítulo {numero}",
                            "url": full_url
                        })
        
        # Ordenar capítulos por número
        info["capitulos"].sort(key=lambda x: float(x["numero"]) if x["numero"].replace('.', '').isdigit() else 999)
        logger.info(f"Extraídos {len(info['capitulos'])} capítulos do manhwa '{info['titulo']}'")
        
    except Exception as e:
        logger.error(f"Erro ao extrair informações do manhwa: {e}", exc_info=True)
        st.error(f"Erro ao extrair informações: {e}")
    
    return info

def extrair_imagens_capitulo(url_capitulo: str) -> List[str]:
    """Extrai URLs das imagens de um capítulo"""
    soup = fazer_requisicao_web(url_capitulo)
    if not soup:
        return []
    
    imagens = []
    base_url = f"{urlparse(url_capitulo).scheme}://{urlparse(url_capitulo).netloc}"
    
    # Múltiplos seletores para encontrar imagens do capítulo
    seletores_img = [
        'div.reading-content img',
        'div.page-break img',
        'div.wp-manga-chapter-img img',
        'img[src*="wp-content"]',
        'img[data-src]',
        '.chapter-content img'
    ]
    
    urls_encontradas = set()
    
    for seletor in seletores_img:
        elementos = soup.select(seletor)
        for img in elementos:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if src:
                url_completa = urljoin(base_url, src)
                if any(ext in url_completa.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']) and url_completa not in urls_encontradas:
                    urls_encontradas.add(url_completa)
                    imagens.append(url_completa)
    
    return imagens

def processar_capitulo_completo(capitulo_info: Dict) -> List[Tuple[Image.Image, str]]:
    """Processa um capítulo completo e extrai todos os painéis"""
    urls_imagens = extrair_imagens_capitulo(capitulo_info["url"])
    
    if not urls_imagens:
        return []
    
    paineis_capitulo = []
    progress_container = st.empty()
    
    for i, url_img in enumerate(urls_imagens):
        progress_container.progress((i + 1) / len(urls_imagens), 
                                  f"Processando página {i+1}/{len(urls_imagens)}")
        
        img_data = baixar_imagem_url_otimizada(url_img)
        if img_data:
            paineis, erro = processar_imagem_otimizada(img_data, f"cap_{capitulo_info['numero']}_pag_{i+1}")
            if not erro:
                paineis_capitulo.extend(paineis)
        
        # Smart delay baseado no contexto e rate limiting
        smart_sleep(context="chapter_processing", url=url_img, show_progress=False)
    
    progress_container.empty()
    return paineis_capitulo

def mostrar_paineis_grid_otimizado(paineis: List[Tuple], titulo: str, expandido: bool = True):
    with st.expander(f"{titulo} - {len(paineis)} painel(is)", expanded=expandido):
        if not paineis:
            st.warning("⚠️ Nenhum painel detectado.")
            return
        
        limite_exibicao = min(12, len(paineis))
        paineis_exibir = paineis[:limite_exibicao]
        
        cols = st.columns(COLS_PER_ROW)
        for i, (painel_img, _) in enumerate(paineis_exibir):
            with cols[i % COLS_PER_ROW]:
                preview_img = painel_img.copy()
                preview_img.thumbnail((200, 200), Image.NEAREST)
                st.image(preview_img, use_container_width=True)
                st.caption(f"Painel {i+1}")
        
        if len(paineis) > limite_exibicao:
            st.info(f"Mostrando {limite_exibicao} de {len(paineis)} painéis")

def criar_zip_otimizado(paineis: List[Tuple] = None) -> bytes:
    paineis_usar = paineis or st.session_state.painel_coletor
    if not paineis_usar:
        return b""
    
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for img_pil, nome_arquivo in paineis_usar:
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG", optimize=True, compress_level=6)
                zipf.writestr(nome_arquivo, buf.getvalue())
        return zip_buffer.getvalue()
    except Exception as e:
        st.error(f"Erro ao criar ZIP: {e}")
        return b""

def reset_processamento():
    st.session_state.painel_coletor.clear()
    st.session_state.contador_paineis = 0
    st.session_state.imagens_processadas.clear()
    st.session_state.paineis_processados.clear()
    st.session_state._cache_hash.clear()
    st.session_state.capitulos_cache.clear()
    st.session_state.manhwa_info.clear()
    
    calcular_hash_rapido.cache_clear()
    validar_url_cached.cache_clear()

# Interface principal
st.sidebar.markdown("### 📊 Estatísticas")
st.sidebar.metric("Painéis extraídos", len(st.session_state.painel_coletor))
st.sidebar.metric("Imagens processadas", len(st.session_state.imagens_processadas))

# Sistema de logs na sidebar
st.sidebar.markdown("### 📋 Logs do Sistema")
if st.sidebar.checkbox("🔍 Mostrar Logs", help="Exibe logs em tempo real"):
    log_level = st.sidebar.selectbox(
        "Nível de Log:",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
        index=1
    )
    
    # Configurar nível de log dinamicamente
    numeric_level = getattr(logging, log_level.upper())
    logger.setLevel(numeric_level)
    
    st.sidebar.info(f"📊 Nível atual: {log_level}")
    
    # Mostrar últimas linhas do log
    try:
        from pathlib import Path
        log_files = list(Path("logs").glob("manhwa_extractor_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    st.sidebar.text_area(
                        "Últimos logs:",
                        ''.join(lines[-10:]),  # Últimas 10 linhas
                        height=200
                    )
    except Exception as e:
        st.sidebar.error(f"Erro ao ler logs: {e}")

# Botões de controle
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Limpar", type="secondary", use_container_width=True):
        reset_processamento()
        st.sidebar.success("Tudo limpo!")

with col2:
    if st.button("🔄 Reset", type="secondary", use_container_width=True):
        reset_processamento()
        st.rerun()

# Rate Limiter status
st.sidebar.markdown("### 🔒 Rate Limiter")
if rate_limiter.requests:
    for domain, requests_list in rate_limiter.requests.items():
        if requests_list:  # Só mostrar domínios com requisições recentes
            recent_requests = len([r for r in requests_list if time.time() - r < 60])
            limit = rate_limiter.limits.get(domain, rate_limiter.limits['default'])['requests']
            
            # Indicador visual do status
            if recent_requests >= limit * 0.8:
                status = "🔴"  # Próximo do limite
            elif recent_requests >= limit * 0.5:
                status = "🟡"  # Uso moderado
            else:
                status = "🟢"  # Ok
            
            st.sidebar.metric(
                f"{status} {domain}",
                f"{recent_requests}/{limit}",
                help=f"Requisições na última hora"
            )
else:
    st.sidebar.info("🟢 Nenhuma requisição recente")

# Debug info
if st.sidebar.checkbox("🔍 Debug"):
    st.sidebar.write(f"Painéis únicos: {len(st.session_state.paineis_processados)}")
    st.sidebar.write(f"Cache size: {len(st.session_state._cache_hash)}")
    st.sidebar.write(f"Manhwas cache: {len(st.session_state.capitulos_cache)}")

# Abas principais
tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Extrair Painéis", "🌐 Web Scraping", "📋 Capítulos", "📦 Download"])

with tab1:
    modo = st.radio("**Escolha o modo:**", 
                   ["📤 Upload", "🌐 URLs"], horizontal=True)
    
    if modo == "📤 Upload":
        arquivos = st.file_uploader(
            "Envie suas imagens:",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True
        )
        
        if arquivos:
            # Sanitizar nomes de arquivo de upload
            files_data = []
            for f in arquivos:
                nome_limpo = input_validator.sanitize_filename(f.name)
                files_data.append((nome_limpo, f.read()))
                logger.info(f"Arquivo carregado: {f.name} -> {nome_limpo}")
            
            files_id = hash(tuple(f"{name}_{len(data)}" for name, data in files_data))
            
            if files_id not in st.session_state.imagens_processadas:
                st.session_state.imagens_processadas.add(files_id)
                
                progress_bar = st.progress(0)
                paineis_novos = []
                
                for i in range(0, len(files_data), BATCH_SIZE):
                    lote = files_data[i:i+BATCH_SIZE]
                    
                    for idx, (nome, data) in enumerate(lote):
                        progress_bar.progress((i + idx + 1) / len(files_data))
                        
                        paineis, erro = processar_imagem_otimizada(data, nome)
                        if not erro:
                            paineis_novos.extend(paineis)
                            mostrar_paineis_grid_otimizado(paineis, f"📄 {nome}")
                
                st.session_state.painel_coletor.extend(paineis_novos)
                progress_bar.empty()
                
                if paineis_novos:
                    st.success(f"✅ {len(paineis_novos)} painéis extraídos!")

with tab2:
    st.markdown("### 🌐 Extração de Manhwa de Sites")
    st.info("📖 Cole a URL da página principal do manhwa (ex: manhwatop.com, reaperscans.com, etc.)")
    
    url_manhwa = st.text_input(
        "URL do Manhwa:",
        placeholder="https://manhwatop.com/manga/nome-do-manhwa/",
        help="URL da página principal do manhwa, não de um capítulo específico"
    )
    
    if url_manhwa and st.button("🔍 Analisar Manhwa", type="primary"):
        try:
            # Validar e sanitizar URL de entrada
            url_manhwa = input_validator.validate_url_input(url_manhwa)
            logger.info(f"URL validada com sucesso: {url_manhwa}")
            
            if not validar_url_cached(url_manhwa):
                st.error("❌ URL inválida ou não permitida!")
            else:
                with st.spinner("🔍 Analisando manhwa..."):
                    soup = fazer_requisicao_web(url_manhwa)
                    if soup:
                        base_url = f"{urlparse(url_manhwa).scheme}://{urlparse(url_manhwa).netloc}"
                        
                        # Detectar tipo de site e usar extrator apropriado
                        if "manhwatop.com" in url_manhwa.lower():
                            info = extrair_info_manhwa_manhwatop(soup, base_url)
                        else:
                            # Extrator genérico para outros sites
                            info = extrair_info_manhwa_manhwatop(soup, base_url)  # Usar o mesmo por enquanto
                        
                        if info["capitulos"]:
                            st.session_state.manhwa_info = info
                            st.session_state.capitulos_cache[url_manhwa] = info
                            st.success(f"✅ Encontrados {len(info['capitulos'])} capítulos!")
                            st.rerun()
                        else:
                            st.error("❌ Nenhum capítulo encontrado. Verifique se a URL está correta.")
        except ValueError as e:
            st.error(f"❌ {e}")
            logger.warning(f"URL rejeitada: {url_manhwa} - {e}")
        except Exception as e:
            st.error("❌ Erro na validação da URL!")
            logger.error(f"Erro inesperado na validação de URL: {e}")

with tab3:
    if st.session_state.manhwa_info:
        info = st.session_state.manhwa_info
        
        # Mostrar informações do manhwa
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if info.get("capa"):
                try:
                    capa_data = baixar_imagem_url_otimizada(info["capa"])
                    if capa_data:
                        capa_img = Image.open(io.BytesIO(capa_data))
                        st.image(capa_img, width=200)
                except (requests.RequestException, IOError, ValueError) as e:
                    st.write("🖼️ Capa não disponível")
                    logger.warning(f"Falha ao carregar imagem de capa: {e}")
        
        with col2:
            st.markdown(f"## 📚 {info['titulo']}")
            if info.get("sinopse"):
                st.markdown(f"**Sinopse:** {info['sinopse']}")
            st.markdown(f"**Capítulos disponíveis:** {len(info['capitulos'])}")
        
        st.markdown("---")
        st.markdown("### 📋 Lista de Capítulos")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            filtro_inicio = st.number_input("Capítulo inicial:", min_value=1, value=1)
        with col2:
            filtro_fim = st.number_input("Capítulo final:", min_value=1, value=len(info['capitulos']))
        with col3:
            modo_exibicao = st.selectbox("Exibição:", ["Lista", "Grade"])
        
        # Filtrar capítulos
        capitulos_filtrados = [
            cap for cap in info['capitulos'] 
            if filtro_inicio <= float(cap['numero']) <= filtro_fim
        ]
        
        if modo_exibicao == "Lista":
            for cap in capitulos_filtrados[:20]:  # Limitar exibição
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**Cap. {cap['numero']}** - {cap['titulo']}")
                
                with col2:
                    if st.button(f"👁️ Ver", key=f"ver_{cap['numero']}"):
                        st.write(f"URL: {cap['url']}")
                
                with col3:
                    if st.button(f"📥 Baixar", key=f"download_{cap['numero']}", type="primary"):
                        with st.spinner(f"Processando Capítulo {cap['numero']}..."):
                            paineis_cap = processar_capitulo_completo(cap)
                            
                            if paineis_cap:
                                # Criar ZIP específico do capítulo
                                zip_data = criar_zip_otimizado(paineis_cap)
                                
                                if zip_data:
                                    st.download_button(
                                        f"📦 Cap. {cap['numero']} ({len(paineis_cap)} painéis)",
                                        data=zip_data,
                                        file_name=f"{info['titulo']}_cap_{cap['numero']}.zip",
                                        mime="application/zip",
                                        key=f"zip_{cap['numero']}"
                                    )
                                    
                                    # Adicionar à coleção geral
                                    st.session_state.painel_coletor.extend(paineis_cap)
                                    st.success(f"✅ {len(paineis_cap)} painéis extraídos do Capítulo {cap['numero']}!")
                            else:
                                st.error(f"❌ Nenhum painel encontrado no Capítulo {cap['numero']}")
        else:
            # Modo grade
            cols = st.columns(3)
            for i, cap in enumerate(capitulos_filtrados[:15]):  # Limitar para não sobrecarregar
                with cols[i % 3]:
                    st.markdown(f"**Cap. {cap['numero']}**")
                    st.markdown(f"{cap['titulo'][:30]}...")
                    
                    col_ver, col_down = st.columns(2)
                    with col_ver:
                        if st.button("👁️", key=f"ver_grid_{cap['numero']}", help="Ver capítulo"):
                            st.write(f"URL: {cap['url']}")
                    
                    with col_down:
                        if st.button("📥", key=f"down_grid_{cap['numero']}", help="Baixar capítulo", type="primary"):
                            with st.spinner(f"Processando..."):
                                paineis_cap = processar_capitulo_completo(cap)
                                
                                if paineis_cap:
                                    zip_data = criar_zip_otimizado(paineis_cap)
                                    
                                    if zip_data:
                                        st.download_button(
                                            f"📦 Download",
                                            data=zip_data,
                                            file_name=f"{info['titulo']}_cap_{cap['numero']}.zip",
                                            mime="application/zip",
                                            key=f"zip_grid_{cap['numero']}"
                                        )
                                        
                                        st.session_state.painel_coletor.extend(paineis_cap)
                                        st.success(f"✅ {len(paineis_cap)} painéis!")
                                else:
                                    st.error("❌ Erro!")
        
        # Opção para baixar múltiplos capítulos
        st.markdown("---")
        st.markdown("### 📦 Download em Lote")
        
        col1, col2 = st.columns(2)
        with col1:
            range_inicio = st.number_input("Do capítulo:", min_value=1, value=1, key="range_inicio")
        with col2:
            range_fim = st.number_input("Até o capítulo:", min_value=1, value=min(5, len(info['capitulos'])), key="range_fim")
        
        if st.button("📥 Baixar Capítulos em Lote", type="primary"):
            try:
                # Validar faixa de capítulos
                range_inicio_val, range_fim_val = input_validator.validate_chapter_range(
                    int(range_inicio), int(range_fim), max_chapters=50
                )
                
                if range_inicio_val != range_inicio or range_fim_val != range_fim:
                    st.warning(f"📊 Faixa ajustada para: {range_inicio_val} - {range_fim_val}")
                
                capitulos_selecionados = [
                    cap for cap in info['capitulos'] 
                    if range_inicio_val <= float(cap['numero']) <= range_fim_val
                ]
            except ValueError as e:
                st.error(f"❌ {e}")
                capitulos_selecionados = []
            
            if capitulos_selecionados:
                total_paineis = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, cap in enumerate(capitulos_selecionados):
                    status_text.text(f"Processando Capítulo {cap['numero']}...")
                    progress_bar.progress((i + 1) / len(capitulos_selecionados))
                    
                    paineis_cap = processar_capitulo_completo(cap)
                    total_paineis.extend(paineis_cap)
                    
                    # Smart delay entre capítulos para não sobrecarregar o servidor
                    smart_sleep(context="batch_download", show_progress=True)
                
                progress_bar.empty()
                status_text.empty()
                
                if total_paineis:
                    st.session_state.painel_coletor.extend(total_paineis)
                    st.success(f"✅ {len(total_paineis)} painéis extraídos de {len(capitulos_selecionados)} capítulos!")
                    
                    # Criar ZIP do lote
                    zip_data = criar_zip_otimizado(total_paineis)
                    if zip_data:
                        st.download_button(
                            f"📦 Baixar Lote ({len(total_paineis)} painéis)",
                            data=zip_data,
                            file_name=f"{info['titulo']}_caps_{range_inicio}-{range_fim}.zip",
                            mime="application/zip"
                        )
                else:
                    st.error("❌ Nenhum painel foi extraído!")
    else:
        st.info("📋 Primeiro, analise um manhwa na aba 'Web Scraping' para ver os capítulos disponíveis.")

with tab4:
    if st.session_state.painel_coletor:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ **{len(st.session_state.painel_coletor)} painéis** prontos para download")
        
        with col2:
            zip_data = criar_zip_otimizado()
            if zip_data:
                st.download_button(
                    "📦 Baixar Todos os Painéis",
                    data=zip_data,
                    file_name="paineis_manhwa_completo.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )
        
        # Mostrar prévia dos painéis
        st.markdown("### 🖼️ Prévia dos Painéis")
        mostrar_paineis_grid_otimizado(st.session_state.painel_coletor, "Todos os Painéis", expandido=False)
        
        # Opções adicionais
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Limpar Coleção", type="secondary"):
                st.session_state.painel_coletor.clear()
                st.success("Coleção limpa!")
                st.rerun()
        
        with col2:
            # Estatísticas
            st.metric("Total de Painéis", len(st.session_state.painel_coletor))
        
        with col3:
            # Tamanho estimado do ZIP
            tamanho_estimado = len(st.session_state.painel_coletor) * 0.5  # Estimativa de 500KB por painel
            st.metric("Tamanho Estimado", f"{tamanho_estimado:.1f} MB")
        
    else:
        st.info("📋 Nenhum painel extraído ainda.")
        st.markdown("### 🚀 Como começar:")
        st.markdown("""
        1. **📤 Upload**: Envie suas próprias imagens de manhwa
        2. **🌐 Web Scraping**: Cole a URL de um manhwa online
        3. **📋 Capítulos**: Selecione e baixe capítulos específicos
        4. **📦 Download**: Baixe todos os painéis extraídos
        """)

# Rodapé com informações adicionais
st.markdown("---")
st.markdown("### 📖 Sites Suportados")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🌟 Totalmente Suportados:**
    - ManhwaTop.com
    - ReaperScans.com
    - AsuraScans.com
    """)

with col2:
    st.markdown("""
    **⚡ Parcialmente Suportados:**
    - MangaDex.org
    - MangaPlus.com
    - Webtoons.com
    """)

with col3:
    st.markdown("""
    **🔧 Em Desenvolvimento:**
    - Outros sites de manhwa
    - Melhorias na detecção
    - Suporte a mais formatos
    """)

# Avisos importantes
st.markdown("---")
st.warning("""
⚠️ **Avisos Importantes:**
- Respeite os direitos autorais dos manhwas
- Use apenas para fins pessoais e educacionais
- Alguns sites podem ter proteções anti-bot
- Velocidade de download depende do servidor de origem
""")

# CSS otimizado
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
}
.main .block-container { 
    background: rgba(255, 255, 255, 0.95); 
    border-radius: 15px; 
    padding: 2rem; 
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 10px 10px 0 0;
    color: #4a4a4a;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255, 255, 255, 0.9);
    color: #667eea;
}
.stButton > button {
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.stDownloadButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}
.stExpander {
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
    margin: 1rem 0;
}
.stMetric {
    background: rgba(255, 255, 255, 0.1);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.stAlert {
    border-radius: 10px;
    border: none;
    padding: 1rem;
    margin: 1rem 0;
}
.stProgress .st-bo {
    background-color: rgba(102, 126, 234, 0.2);
}
.stProgress .st-bp {
    background: linear-gradient(45deg, #667eea, #764ba2);
}
</style>
""", unsafe_allow_html=True)
# --- CSS customizado para aparência melhor ---

st.markdown("""
<style>
h1, h2, h3 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #1f77b4;
}

body {
    background-color: #f0f2f6;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    padding: 8px 15px;
    font-weight: bold;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #145a86;
}

.stProgress > div > div > div > div {
    background-color: #1f77b4 !important;
}

.botao-flutuante-topo, .botao-flutuante-baixo {
    position: fixed;
    right: 30px;
    z-index: 9999;
    background-color: #1f77b4;
    color: white !important;
    padding: 12px 24px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    text-decoration: none;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
    cursor: pointer;
}
.botao-flutuante-topo {
    bottom: 100px;
}
.botao-flutuante-baixo {
    bottom: 40px;
}
</style>

<a href="#topo" class="botao-flutuante-topo">⬆️ Topo</a>
<a href="#final_paineis" class="botao-flutuante-baixo">⬇️ Painéis</a>
""", unsafe_allow_html=True)

st.markdown("""
<script>
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute("href"));
        if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    });
});
</script>
""", unsafe_allow_html=True)
st.markdown('<div id="final_paineis" style="height:1px;"></div>', unsafe_allow_html=True)
