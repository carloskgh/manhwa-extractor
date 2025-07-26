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
import asyncio
import aiohttp
import aiofiles
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- Download automático do modelo best.pt, se necessário ---
def baixar_best_pt_if_needed():
    modelo_dir = "modelos"
    modelo_path = os.path.join(modelo_dir, "best.pt")
    url = "https://www.dropbox.com/scl/fi/a743aqjqzau3fxy4fss4a/best.pt?rlkey=a24lozm0cw8znku0h743ylx2z&st=c4t06y2d&dl=1"
    if not os.path.exists(modelo_path):
        try:
            os.makedirs(modelo_dir, exist_ok=True)
            print(f"Baixando modelo YOLO best.pt de {url} ...")
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(modelo_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Modelo best.pt baixado com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar best.pt: {e}")
    else:
        print("Modelo best.pt já existe.")

baixar_best_pt_if_needed()
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

# Sistema de Operações Assíncronas
MAX_WORKERS = 4  # Constante para workers assíncronos

class AsyncOperationsManager:
    """Gerenciador de operações assíncronas para melhor performance"""
    
    def __init__(self):
        self.session = None
        # Usar valor padrão na inicialização, configuração será aplicada depois
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)  # Limite de conexões simultâneas
        self.process_pool = ProcessPoolExecutor(max_workers=2)  # Para processamento pesado
        self.config_manager = None  # Será definido depois
        logger.info(f"🚀 Gerenciador de operações assíncronas inicializado - Workers padrão: {MAX_WORKERS}")
    
    def set_config_manager(self, config_manager):
        """Define o config manager e ajusta configurações"""
        self.config_manager = config_manager
        # Reconfigurar semáforo com valor do config
        max_workers = config_manager.get('scraping', 'max_workers', MAX_WORKERS)
        self.semaphore = asyncio.Semaphore(max_workers)
        logger.info(f"🔧 Async manager reconfigurado - Workers: {max_workers}")
    
    async def get_session(self):
        """Obtém ou cria sessão HTTP assíncrona"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            # Usar config_manager se disponível, senão usar padrão
            max_workers = self.config_manager.get('scraping', 'max_workers', MAX_WORKERS) if self.config_manager else MAX_WORKERS
            connector = aiohttp.TCPConnector(limit=max_workers, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                headers=SCRAPING_HEADERS,
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    async def close_session(self):
        """Fecha a sessão HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Sessão HTTP fechada")
    
    async def download_image_async(self, url: str) -> Optional[bytes]:
        """Download assíncrono de imagem com rate limiting e cache"""
        start_time = time.time()
        metrics_collector.active_downloads += 1
        
        try:
            # Verificar cache primeiro
            cached_image = intelligent_cache.get(url, 'image')
            if cached_image:
                logger.debug(f"Imagem obtida do cache: {url}")
                metrics_collector.record_request(time.time() - start_time, True, "cache_hit")
                return cached_image
            
            # Verificar rate limiting
            can_request, wait_time = rate_limiter.can_request(url)
            if not can_request:
                logger.info(f"Rate limit ativo para {rate_limiter.get_domain(url)}, aguardando {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            
            if not validar_url_cached(url):
                metrics_collector.record_request(time.time() - start_time, False, "validation_failed")
                return None
            
            async with self.semaphore:  # Limitar conexões simultâneas
                session = await self.get_session()
                
                # HEAD request para verificar tamanho
                try:
                    rate_limiter.record_request(url)
                    async with session.head(url) as response:
                        content_length = response.headers.get('content-length')
                        if content_length and int(content_length) > 10 * 1024 * 1024:
                            logger.warning(f"Imagem muito grande ({content_length} bytes), ignorando download")
                            return None
                except Exception:
                    pass  # Continuar com GET se HEAD falhar
                
                # GET request para baixar
                await asyncio.sleep(rate_limiter.smart_delay(url, "image_download"))
                rate_limiter.record_request(url)
                
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    # Verificar content-type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'webp']):
                        logger.warning(f"Content-type inesperado para imagem: {content_type}")
                        return None
                    
                    # Download com limite de tamanho
                    content = b""
                    max_size = 10 * 1024 * 1024
                    
                    async for chunk in response.content.iter_chunked(8192):
                        content += chunk
                        if len(content) > max_size:
                            logger.warning(f"Tamanho da imagem excedeu {max_size} bytes, truncando download")
                            return None
                    
                    if len(content) < 100:
                        return None
                    
                    logger.debug(f"Download assíncrono concluído: {url} ({len(content)} bytes)")
                    
                    # Armazenar no cache
                    intelligent_cache.set(url, content, 'image')
                    
                    metrics_collector.record_request(time.time() - start_time, True, "download_success")
                    return content
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout no download de {url}")
            metrics_collector.record_request(time.time() - start_time, False, "timeout")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Erro de rede ao baixar imagem de {url}: {e}")
            metrics_collector.record_request(time.time() - start_time, False, "network_error")
            return None
        except Exception as e:
            logger.error(f"Erro inesperado ao baixar imagem de {url}: {e}", exc_info=True)
            metrics_collector.record_request(time.time() - start_time, False, "unexpected_error")
            return None
        finally:
            metrics_collector.active_downloads -= 1
    
    async def download_multiple_images(self, urls: List[str], progress_callback=None) -> List[Tuple[str, Optional[bytes]]]:
        """Download assíncrono de múltiplas imagens"""
        logger.info(f"Iniciando download assíncrono de {len(urls)} imagens")
        
        async def download_with_progress(i, url):
            result = await self.download_image_async(url)
            if progress_callback:
                progress_callback(i + 1, len(urls))
            return url, result
        
        # Executar downloads em paralelo
        tasks = [download_with_progress(i, url) for i, url in enumerate(urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar resultados válidos
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Erro no download assíncrono: {result}")
            else:
                valid_results.append(result)
        
        logger.info(f"Download assíncrono concluído: {len(valid_results)}/{len(urls)} sucessos")
        return valid_results
    
    async def process_image_async(self, img_data: bytes, nome_fonte: str) -> Tuple[List, Optional[str]]:
        """Processamento assíncrono de imagem em processo separado com cache"""
        start_time = time.time()
        metrics_collector.active_processing += 1
        
        try:
            # Verificar cache primeiro
            cache_key = f"{nome_fonte}_{hashlib.md5(img_data).hexdigest()}"
            cached_result = intelligent_cache.get(cache_key, 'processed_image')
            if cached_result:
                logger.debug(f"Resultado de processamento obtido do cache: {nome_fonte}")
                processing_time = time.time() - start_time
                paineis, erro = cached_result
                panels_count = len(paineis) if not erro else 0
                metrics_collector.record_processing(processing_time, panels_count, nome_fonte)
                return cached_result
            
            logger.info(f"Processando imagem assíncrona: {nome_fonte} ({len(img_data)} bytes)")
            
            # Executar processamento pesado em processo separado
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.process_pool,
                self._process_image_sync,
                img_data,
                nome_fonte
            )
            
            # Armazenar resultado no cache
            intelligent_cache.set(cache_key, result, 'processed_image')
            
            # Registrar métricas de processamento
            paineis, erro = result
            processing_time = time.time() - start_time
            panels_count = len(paineis) if not erro else 0
            metrics_collector.record_processing(processing_time, panels_count, nome_fonte)
            
            return result
        except Exception as e:
            logger.error(f"Erro no processamento assíncrono de {nome_fonte}: {e}", exc_info=True)
            processing_time = time.time() - start_time
            metrics_collector.record_processing(processing_time, 0, nome_fonte)
            return [], str(e)
        finally:
            metrics_collector.active_processing -= 1
    
    def _process_image_sync(self, img_data: bytes, nome_fonte: str) -> Tuple[List, Optional[str]]:
        """Processamento síncrono de imagem (executado em processo separado)"""
        try:
            img_pil = carregar_e_redimensionar_imagem(img_data)
            if img_pil is None:
                logger.warning(f"Falha ao carregar imagem: {nome_fonte}")
                return [], "Erro ao carregar imagem"
                
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            paineis = extrair_paineis_hibrido_otimizado(img, img_id=nome_fonte)
            logger.info(f"Extraídos {len(paineis)} painéis de {nome_fonte}")
            return paineis, None
        except Exception as e:
            logger.error(f"Erro no processamento síncrono de {nome_fonte}: {e}")
            return [], str(e)
    
    def cleanup(self):
        """Limpeza de recursos"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            logger.info("Process pool encerrado")

# Sistema de Configuração Externa (MOVIDO PARA CÁ)
class ConfigManager:
    """Gerenciador de configurações externas"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "app": {
                "name": "Extrator de Painéis de Manhwa",
                "version": "2.0.0",
                "debug": False,
                "max_upload_size_mb": 50,
                "supported_formats": ["jpg", "jpeg", "png", "webp"]
            },
            "yolo": {
                "model_name": "best.pt",
                "confidence_threshold": 0.25,
                "device": "cpu",
                "max_det": 300
            },
            "opencv": {
                "max_width": 1024,
                "min_contour_size": 100,
                "blur_threshold": 100,
                "area_threshold": 1000
            },
            "scraping": {
                "request_timeout": 15,
                "max_retries": 3,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "max_workers": 4
            },
            "rate_limits": {
                "manhwatop.com": {"requests": 5, "window": 60},
                "reaperscans.com": {"requests": 3, "window": 60},
                "asurascans.com": {"requests": 4, "window": 60},
                "mangadex.org": {"requests": 8, "window": 60},
                "default": {"requests": 10, "window": 60}
            },
            "cache": {
                "max_memory_mb": 200,
                "ttl": {
                    "image": 86400,
                    "webpage": 21600,
                    "chapter_list": 43200,
                    "processed_image": 604800,
                    "default": 7200
                }
            },
            "logging": {
                "level": "INFO",
                "file_rotation": True,
                "max_file_size_mb": 10,
                "backup_count": 5
            },
            "ui": {
                "theme": "dark",
                "sidebar_state": "expanded",
                "show_debug": False,
                "auto_refresh_metrics": True,
                "metrics_refresh_interval": 10
            },
            "alerts": {
                "success_rate_threshold": 80,
                "memory_threshold_gb": 1.0,
                "response_time_threshold_ms": 5000,
                "enable_notifications": True
            }
        }
        self.config = self._load_config()
        self._apply_env_overrides()
        logger.info(f"⚙️ Configuração carregada - Arquivo: {config_file}")
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo ou cria padrão"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge com configuração padrão para ter todas as chaves
                    return self._deep_merge(self.default_config, loaded_config)
            else:
                # Criar arquivo de configuração padrão
                self.save_config(self.default_config)
                return self.default_config.copy()
        except Exception as e:
            logger.warning(f"Erro ao carregar configuração: {e}, usando padrão")
            return self.default_config.copy()
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Merge profundo de dicionários"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _apply_env_overrides(self):
        """Aplica overrides de variáveis de ambiente"""
        env_mappings = {
            'MANHWA_DEBUG': ('app', 'debug'),
            'MANHWA_LOG_LEVEL': ('logging', 'level'),
            'MANHWA_THEME': ('ui', 'theme'),
            'MANHWA_CACHE_SIZE': ('cache', 'max_memory_mb'),
            'MANHWA_MAX_WORKERS': ('scraping', 'max_workers'),
            'MANHWA_REQUEST_TIMEOUT': ('scraping', 'request_timeout')
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    # Tentar converter tipos apropriados
                    if key in ['debug', 'file_rotation', 'show_debug', 'auto_refresh_metrics', 'enable_notifications']:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif key in ['max_memory_mb', 'max_workers', 'request_timeout', 'metrics_refresh_interval']:
                        value = int(value)
                    elif key in ['memory_threshold_gb']:
                        value = float(value)
                    
                    self.config[section][key] = value
                    logger.info(f"Override de env aplicado: {env_var} = {value}")
                except Exception as e:
                    logger.warning(f"Erro ao aplicar override {env_var}: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """Obtém valor de configuração"""
        try:
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)
        except Exception:
            return default
    
    def set(self, section: str, key: str, value):
        """Define valor de configuração"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save_config(self, config: dict = None):
        """Salva configuração no arquivo"""
        try:
            config_to_save = config or self.config
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuração salva em {self.config_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar configuração: {e}")
    
    def reset_to_default(self):
        """Restaura configuração padrão"""
        self.config = self.default_config.copy()
        self.save_config()
        logger.info("Configuração restaurada para padrão")
    
    def export_config(self) -> str:
        """Exporta configuração como JSON string"""
        return json.dumps(self.config, indent=2, ensure_ascii=False)
    
    def render_config_editor(self):
        """Renderiza editor de configuração na interface"""
        st.markdown("### ⚙️ Editor de Configuração")
        
        # Tabs para diferentes seções
        sections = list(self.config.keys())
        selected_section = st.selectbox("Seção:", sections)
        
        if selected_section:
            st.markdown(f"#### 📋 {selected_section.title()}")
            
            section_config = self.config[selected_section]
            updated = False
            
            for key, value in section_config.items():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**{key}:**")
                
                with col2:
                    if isinstance(value, bool):
                        new_value = st.checkbox(f"", value=value, key=f"{selected_section}_{key}")
                    elif isinstance(value, int):
                        new_value = st.number_input(f"", value=value, key=f"{selected_section}_{key}")
                    elif isinstance(value, float):
                        new_value = st.number_input(f"", value=value, format="%.2f", key=f"{selected_section}_{key}")
                    elif isinstance(value, dict):
                        new_value = st.text_area(f"", value=json.dumps(value, indent=2), key=f"{selected_section}_{key}")
                        try:
                            new_value = json.loads(new_value)
                        except:
                            st.error("JSON inválido")
                            new_value = value
                    else:
                        new_value = st.text_input(f"", value=str(value), key=f"{selected_section}_{key}")
                    
                    if new_value != value:
                        self.config[selected_section][key] = new_value
                        updated = True
            
            # Botões de controle
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💾 Salvar Alterações"):
                    self.save_config()
                    st.success("Configuração salva!")
            
            with col2:
                if st.button("📤 Exportar JSON"):
                    st.text_area("Configuração JSON:", self.export_config(), height=300)
            
            with col3:
                if st.button("🔄 Resetar Padrão"):
                    self.reset_to_default()
                    st.success("Configuração resetada!")
    
    # Inicializar gerenciador de configuração primeiro
    config_manager = ConfigManager()

# Inicializar gerenciador assíncrono
async_manager = AsyncOperationsManager()

# Configurar async_manager com config_manager
async_manager.set_config_manager(config_manager)

# Sistema de Monitoramento e Métricas
from dataclasses import dataclass, field
from collections import deque
import psutil

@dataclass
class PerformanceMetrics:
    """Métricas de performance da aplicação"""
    total_requests: int = 0
    failed_requests: int = 0
    total_images_processed: int = 0
    total_panels_extracted: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_download_time: float = 0.0
    avg_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))

class MetricsCollector:
    """Coletor de métricas em tempo real"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.active_downloads = 0
        self.active_processing = 0
        logger.info("📊 Sistema de métricas inicializado")
    
    def record_request(self, duration: float, success: bool, request_type: str = "download"):
        """Registra métricas de requisição"""
        self.metrics.total_requests += 1
        if not success:
            self.metrics.failed_requests += 1
        
        self.metrics.response_times.append(duration)
        if self.metrics.response_times:
            self.metrics.avg_download_time = sum(self.metrics.response_times) / len(self.metrics.response_times)
        
        logger.debug(f"Métrica registrada: {request_type} - {duration:.2f}s - {'✅' if success else '❌'}")
    
    def record_processing(self, duration: float, panels_extracted: int, image_name: str):
        """Registra métricas de processamento"""
        self.metrics.total_images_processed += 1
        self.metrics.total_panels_extracted += panels_extracted
        
        self.metrics.processing_times.append(duration)
        if self.metrics.processing_times:
            self.metrics.avg_processing_time = sum(self.metrics.processing_times) / len(self.metrics.processing_times)
        
        logger.debug(f"Processamento registrado: {image_name} - {duration:.2f}s - {panels_extracted} painéis")
    
    def record_cache_event(self, hit: bool):
        """Registra evento de cache"""
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
    
    def update_memory_usage(self):
        """Atualiza uso de memória"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.metrics.peak_memory_mb:
                self.metrics.peak_memory_mb = memory_mb
        except Exception:
            pass  # psutil pode não estar disponível
    
    def get_success_rate(self) -> float:
        """Calcula taxa de sucesso"""
        if self.metrics.total_requests == 0:
            return 0.0
        return (self.metrics.total_requests - self.metrics.failed_requests) / self.metrics.total_requests
    
    def get_cache_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache"""
        total = self.metrics.cache_hits + self.metrics.cache_misses
        if total == 0:
            return 0.0
        return self.metrics.cache_hits / total
    
    def get_uptime(self) -> float:
        """Calcula tempo de execução em segundos"""
        return time.time() - self.metrics.start_time
    
    def get_throughput(self) -> float:
        """Calcula throughput (painéis por minuto)"""
        uptime_minutes = self.get_uptime() / 60
        if uptime_minutes == 0:
            return 0.0
        return self.metrics.total_panels_extracted / uptime_minutes
    
    def display_dashboard(self):
        """Exibe dashboard completo de métricas"""
        st.markdown("### 📊 Dashboard de Performance")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            theme_manager.create_animated_metric(
                "Taxa de Sucesso", 
                f"{self.get_success_rate():.1%}",
                f"{self.metrics.total_requests} reqs" if self.metrics.total_requests > 0 else None
            )
        
        with col2:
            theme_manager.create_animated_metric(
                "Painéis Extraídos", 
                str(self.metrics.total_panels_extracted),
                f"{self.metrics.total_images_processed} imgs"
            )
        
        with col3:
            theme_manager.create_animated_metric(
                "Cache Hit Rate", 
                f"{self.get_cache_hit_rate():.1%}",
                "Ótimo" if self.get_cache_hit_rate() > 0.8 else "Baixo"
            )
        
        with col4:
            theme_manager.create_animated_metric(
                "Throughput", 
                f"{self.get_throughput():.1f}/min",
                f"{self.get_uptime()/60:.1f}min uptime"
            )
        
        # Métricas de tempo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Tempo Médio Download", 
                f"{self.metrics.avg_download_time:.2f}s",
                delta="Rápido" if self.metrics.avg_download_time < 2.0 else "Lento"
            )
        
        with col2:
            st.metric(
                "Tempo Médio Processamento", 
                f"{self.metrics.avg_processing_time:.2f}s",
                delta="Eficiente" if self.metrics.avg_processing_time < 5.0 else "Lento"
            )
        
        with col3:
            self.update_memory_usage()
            st.metric(
                "Pico de Memória", 
                f"{self.metrics.peak_memory_mb:.1f}MB",
                delta="Normal" if self.metrics.peak_memory_mb < 500 else "Alto"
            )
        
        # Operações em tempo real
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Downloads Ativos", 
                self.active_downloads,
                help="Número de downloads em execução"
            )
        
        with col2:
            st.metric(
                "Processamentos Ativos", 
                self.active_processing,
                help="Número de imagens sendo processadas"
            )
        
        # Gráfico de tempos de resposta (últimos 50)
        if len(self.metrics.response_times) > 5:
            st.markdown("#### 📈 Tempos de Resposta Recentes")
            chart_data = list(self.metrics.response_times)[-50:]
            st.line_chart(chart_data)
        
        # Alertas automáticos
        self._show_performance_alerts()
    
    def _show_performance_alerts(self):
        """Mostra alertas de performance"""
        alerts = []
        
        if self.get_success_rate() < 0.8 and self.metrics.total_requests > 10:
            alerts.append("🔴 Taxa de sucesso baixa (<80%)")
        
        if self.metrics.avg_download_time > 5.0:
            alerts.append("🟡 Downloads lentos (>5s)")
        
        if self.metrics.avg_processing_time > 10.0:
            alerts.append("🟡 Processamento lento (>10s)")
        
        if self.metrics.peak_memory_mb > 1000:
            alerts.append("🔴 Alto uso de memória (>1GB)")
        
        if self.get_cache_hit_rate() < 0.3 and (self.metrics.cache_hits + self.metrics.cache_misses) > 10:
            alerts.append("🟡 Cache pouco eficiente (<30%)")
        
        if alerts:
            st.markdown("#### ⚠️ Alertas de Performance")
            for alert in alerts:
                st.warning(alert)
    
    def get_summary_stats(self) -> Dict:
        """Retorna estatísticas resumidas"""
        return {
            "success_rate": self.get_success_rate(),
            "total_panels": self.metrics.total_panels_extracted,
            "avg_download": self.metrics.avg_download_time,
            "avg_processing": self.metrics.avg_processing_time,
            "cache_rate": self.get_cache_hit_rate(),
            "throughput": self.get_throughput(),
            "uptime": self.get_uptime(),
            "memory_peak": self.metrics.peak_memory_mb
        }

# Inicializar coletor de métricas
metrics_collector = MetricsCollector()

# Sistema de Cache Inteligente
from pathlib import Path
import pickle
from typing import Any

class IntelligentCache:
    """Sistema de cache inteligente com persistência e invalidação automática"""
    
    def __init__(self, cache_dir: str = "cache", max_memory_mb: int = 200):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_cache = {}  # Cache em memória para itens pequenos
        self.cache_index = {}   # Índice com metadados dos itens
        self.hit_count = 0
        self.miss_count = 0
        
        # Configurações de TTL por tipo
        self.ttl_config = {
            'image': 3600 * 24,      # Imagens: 24 horas
            'webpage': 3600 * 6,     # Páginas web: 6 horas  
            'chapter_list': 3600 * 12, # Lista de capítulos: 12 horas
            'processed_image': 3600 * 24 * 7, # Imagens processadas: 7 dias
            'default': 3600 * 2      # Padrão: 2 horas
        }
        
        self._load_index()
        self._cleanup_expired()
        logger.info(f"💾 Cache inteligente inicializado - Diretório: {cache_dir}")
    
    def _get_cache_key(self, data: Any, cache_type: str = "default") -> str:
        """Gera chave única para o cache"""
        if isinstance(data, str):
            data_str = data
        elif isinstance(data, bytes):
            data_str = hashlib.md5(data).hexdigest()
        else:
            data_str = str(data)
        
        key_data = f"{cache_type}_{data_str}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _get_file_path(self, key: str) -> Path:
        """Obtém caminho do arquivo de cache"""
        return self.cache_dir / f"{key}.cache"
    
    def _load_index(self):
        """Carrega índice do cache"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logger.debug(f"Índice do cache carregado: {len(self.cache_index)} itens")
        except Exception as e:
            logger.warning(f"Erro ao carregar índice do cache: {e}")
            self.cache_index = {}
    
    def _save_index(self):
        """Salva índice do cache"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Erro ao salvar índice do cache: {e}")
    
    def _cleanup_expired(self):
        """Remove itens expirados do cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.cache_index.items():
            if current_time > metadata.get('expires_at', 0):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_item(key)
        
        if expired_keys:
            logger.info(f"Cache cleanup: {len(expired_keys)} itens expirados removidos")
    
    def _remove_item(self, key: str):
        """Remove item do cache"""
        # Remover da memória
        self.memory_cache.pop(key, None)
        
        # Remover arquivo
        file_path = self._get_file_path(key)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Erro ao remover arquivo de cache {key}: {e}")
        
        # Remover do índice
        self.cache_index.pop(key, None)
    
    def _get_memory_usage(self) -> int:
        """Calcula uso atual de memória do cache"""
        total_size = 0
        for key, data in self.memory_cache.items():
            try:
                total_size += len(pickle.dumps(data))
            except:
                pass
        return total_size
    
    def _enforce_memory_limit(self):
        """Aplica limite de memória removendo itens menos usados"""
        if self._get_memory_usage() <= self.max_memory_bytes:
            return
        
        # Ordenar por último acesso e remover os mais antigos
        sorted_items = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        for key, _ in sorted_items:
            if key in self.memory_cache:
                del self.memory_cache[key]
                logger.debug(f"Item removido da memória por limite: {key}")
                
                if self._get_memory_usage() <= self.max_memory_bytes * 0.8:
                    break
    
    def get(self, data: Any, cache_type: str = "default") -> Any:
        """Recupera item do cache"""
        key = self._get_cache_key(data, cache_type)
        current_time = time.time()
        
        # Verificar se existe e não expirou
        if key not in self.cache_index:
            self.miss_count += 1
            metrics_collector.record_cache_event(False)
            return None
        
        metadata = self.cache_index[key]
        if current_time > metadata.get('expires_at', 0):
            self._remove_item(key)
            self.miss_count += 1
            metrics_collector.record_cache_event(False)
            return None
        
        # Tentar recuperar da memória primeiro
        if key in self.memory_cache:
            self.hit_count += 1
            metadata['last_access'] = current_time
            metadata['hit_count'] = metadata.get('hit_count', 0) + 1
            metrics_collector.record_cache_event(True)
            logger.debug(f"Cache hit (memória): {cache_type}")
            return self.memory_cache[key]
        
        # Recuperar do disco
        file_path = self._get_file_path(key)
        if not file_path.exists():
            self._remove_item(key)
            self.miss_count += 1
            metrics_collector.record_cache_event(False)
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Colocar na memória se couber
            if self._get_memory_usage() < self.max_memory_bytes:
                self.memory_cache[key] = data
                self._enforce_memory_limit()
            
            self.hit_count += 1
            metadata['last_access'] = current_time
            metadata['hit_count'] = metadata.get('hit_count', 0) + 1
            metrics_collector.record_cache_event(True)
            logger.debug(f"Cache hit (disco): {cache_type}")
            return data
            
        except Exception as e:
            logger.warning(f"Erro ao ler cache {key}: {e}")
            self._remove_item(key)
            self.miss_count += 1
            metrics_collector.record_cache_event(False)
            return None
    
    def set(self, data: Any, value: Any, cache_type: str = "default") -> bool:
        """Armazena item no cache"""
        key = self._get_cache_key(data, cache_type)
        current_time = time.time()
        ttl = self.ttl_config.get(cache_type, self.ttl_config['default'])
        
        try:
            # Metadados do item
            metadata = {
                'created_at': current_time,
                'last_access': current_time,
                'expires_at': current_time + ttl,
                'cache_type': cache_type,
                'hit_count': 0,
                'size_bytes': len(pickle.dumps(value))
            }
            
            # Salvar no disco
            file_path = self._get_file_path(key)
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Colocar na memória se couber
            if metadata['size_bytes'] < self.max_memory_bytes // 10:  # Máximo 10% da memória por item
                self.memory_cache[key] = value
                self._enforce_memory_limit()
            
            # Atualizar índice
            self.cache_index[key] = metadata
            self._save_index()
            
            logger.debug(f"Item adicionado ao cache: {cache_type} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"Erro ao salvar no cache: {e}")
            return False
    
    def invalidate_type(self, cache_type: str):
        """Invalida todos os itens de um tipo específico"""
        keys_to_remove = [
            key for key, metadata in self.cache_index.items()
            if metadata.get('cache_type') == cache_type
        ]
        
        for key in keys_to_remove:
            self._remove_item(key)
        
        if keys_to_remove:
            self._save_index()
            logger.info(f"Cache invalidado: {len(keys_to_remove)} itens do tipo '{cache_type}'")
    
    def clear_all(self):
        """Limpa todo o cache"""
        # Limpar memória
        self.memory_cache.clear()
        
        # Remover arquivos
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Erro ao remover {cache_file}: {e}")
        
        # Limpar índice
        self.cache_index.clear()
        self._save_index()
        
        logger.info("Cache completamente limpo")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        # Calcular tamanhos
        memory_usage = self._get_memory_usage()
        disk_usage = sum(
            metadata.get('size_bytes', 0) 
            for metadata in self.cache_index.values()
        )
        
        # Estatísticas por tipo
        type_stats = {}
        for metadata in self.cache_index.values():
            cache_type = metadata.get('cache_type', 'unknown')
            if cache_type not in type_stats:
                type_stats[cache_type] = {'count': 0, 'size': 0}
            type_stats[cache_type]['count'] += 1
            type_stats[cache_type]['size'] += metadata.get('size_bytes', 0)
        
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_items': len(self.cache_index),
            'memory_items': len(self.memory_cache),
            'memory_usage_mb': memory_usage / 1024 / 1024,
            'disk_usage_mb': disk_usage / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'type_stats': type_stats
        }
    
    def display_dashboard(self):
        """Exibe dashboard do cache - VERSÃO ULTRA-SEGURA"""
        try:
            stats = self.get_stats()
            
            st.markdown("### 💾 Dashboard do Cache")
            
            # Métricas principais - VERSÃO SEGURA
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                hit_rate = stats.get('hit_rate', 0)
                st.metric(
                    "Hit Rate",
                    f"{hit_rate:.1%}",
                    delta="Excelente" if hit_rate > 0.8 else "Baixo"
                )
            
            with col2:
                total_items = stats.get('total_items', 0)
                memory_items = stats.get('memory_items', 0)
                st.metric(
                    "Total de Itens",
                    total_items,
                    delta=f"{memory_items} em memória"
                )
            
            with col3:
                memory_usage = stats.get('memory_usage_mb', 0)
                max_memory = stats.get('max_memory_mb', 200)
                st.metric(
                    "Uso de Memória",
                    f"{memory_usage:.1f}MB",
                    delta=f"/{max_memory:.0f}MB"
                )
            
            with col4:
                disk_usage = stats.get('disk_usage_mb', 0)
                st.metric(
                    "Uso de Disco",
                    f"{disk_usage:.1f}MB",
                    delta=f"{total_items} itens"
                )
        except Exception as e:
            st.error(f"Erro ao exibir métricas do cache: {e}")
            logger.error(f"Erro no display_dashboard do cache: {e}", exc_info=True)
        
        # Estatísticas por tipo
        if stats['type_stats']:
            st.markdown("#### 📊 Cache por Tipo")
            for cache_type, type_data in stats['type_stats'].items():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"📁 {cache_type.title()}",
                        type_data['count'],
                        help=f"Número de itens do tipo {cache_type}"
                    )
                with col2:
                    st.metric(
                        "Tamanho",
                        f"{type_data['size'] / 1024 / 1024:.1f}MB",
                        help=f"Espaço usado pelo tipo {cache_type}"
                    )
        
        # Botões de controle
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Limpar Cache Expirado"):
                self._cleanup_expired()
                st.success("Cache expirado removido!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Limpar Tudo", type="secondary"):
                self.clear_all()
                st.success("Cache completamente limpo!")
                st.rerun()
        
        with col3:
            if st.button("📊 Atualizar Stats"):
                st.rerun()

# Inicializar cache inteligente
intelligent_cache = IntelligentCache()

# Wrapper para executar operações assíncronas no Streamlit
def run_async(coro):
    """Executa corrotina assíncrona de forma compatível com Streamlit"""
    try:
        # Tentar usar loop existente
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Se já há um loop rodando, criar novo thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # Criar novo loop se necessário
        return asyncio.run(coro)

def process_chapter_async(capitulo_info: Dict, progress_container=None) -> List[Tuple[Image.Image, str]]:
    """Versão assíncrona do processamento de capítulo"""
    
    async def _process_async():
        urls_imagens = extrair_imagens_capitulo(capitulo_info["url"])
        
        if not urls_imagens:
            return []
        
        paineis_capitulo = []
        
        # Callback de progresso thread-safe
        def update_progress(current, total):
            if progress_container:
                progress_container.progress(current / total, 
                                          f"Processando página {current}/{total}")
        
        # Download assíncrono de múltiplas imagens
        download_results = await async_manager.download_multiple_images(
            urls_imagens, 
            progress_callback=update_progress
        )
        
        # Processar imagens em paralelo
        tasks = []
        for i, (url, img_data) in enumerate(download_results):
            if img_data:
                task = async_manager.process_image_async(
                    img_data, 
                    f"cap_{capitulo_info['numero']}_pag_{i+1}"
                )
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Erro no processamento assíncrono: {result}")
                else:
                    paineis, erro = result
                    if not erro:
                        paineis_capitulo.extend(paineis)
        
        return paineis_capitulo
    
    return run_async(_process_async())

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
st.set_page_config(
    page_title="🖼️ Extrator de Painéis de Manhwa",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sistema de Temas e Interface Otimizada
class ThemeManager:
    """Gerenciador de temas e interface otimizada"""
    
    def __init__(self):
        self.themes = {
            'dark': {
                'name': '🌙 Tema Escuro',
                'primary_color': '#FF4B4B',
                'background_color': '#0E1117',
                'secondary_background': '#262730',
                'text_color': '#FAFAFA',
                'accent_color': '#FF6B6B'
            },
            'light': {
                'name': '☀️ Tema Claro', 
                'primary_color': '#FF4B4B',
                'background_color': '#FFFFFF',
                'secondary_background': '#F0F2F6',
                'text_color': '#262730',
                'accent_color': '#FF6B6B'
            },
            'cyberpunk': {
                'name': '🎮 Cyberpunk',
                'primary_color': '#00FF41',
                'background_color': '#000000',
                'secondary_background': '#1A1A1A',
                'text_color': '#00FF41',
                'accent_color': '#FF0080'
            },
            'ocean': {
                'name': '🌊 Oceano',
                'primary_color': '#4FC3F7',
                'background_color': '#0D47A1',
                'secondary_background': '#1565C0',
                'text_color': '#E3F2FD',
                'accent_color': '#81C784'
            }
        }
        self.current_theme = self._load_theme_preference()
        logger.info(f"🎨 Gerenciador de temas inicializado - Tema: {self.current_theme}")
    
    def _load_theme_preference(self) -> str:
        """Carrega preferência de tema do usuário"""
        try:
            if 'user_theme' not in st.session_state:
                st.session_state.user_theme = 'dark'
            return st.session_state.user_theme
        except:
            return 'dark'
    
    def _save_theme_preference(self, theme: str):
        """Salva preferência de tema"""
        st.session_state.user_theme = theme
        self.current_theme = theme
    
    def apply_custom_css(self):
        """Aplica CSS customizado baseado no tema"""
        theme = self.themes[self.current_theme]
        
        css = f"""
        <style>
        /* Animações e transições */
        .element-container {{
            transition: all 0.3s ease-in-out;
        }}
        
        .stButton > button {{
            transition: all 0.2s ease-in-out;
            border-radius: 10px;
            border: 2px solid {theme['primary_color']};
            background: linear-gradient(45deg, {theme['primary_color']}, {theme['accent_color']});
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
            scale: 1.05;
        }}
        
        .stButton > button:active {{
            transform: translateY(0px);
            scale: 0.98;
        }}
        
        /* Métricas animadas */
        .metric-card {{
            background: linear-gradient(135deg, {theme['secondary_background']}, {theme['background_color']});
            padding: 1rem;
            border-radius: 15px;
            border: 1px solid {theme['primary_color']}40;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            border-color: {theme['primary_color']};
        }}
        
        /* Progress bars animadas */
        .stProgress > div > div > div {{
            background: linear-gradient(45deg, {theme['primary_color']}, {theme['accent_color']});
            border-radius: 10px;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 0.8; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.8; }}
        }}
        
        /* Alertas estilizados */
        .stAlert {{
            border-radius: 12px;
            border-left: 5px solid {theme['primary_color']};
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ transform: translateX(-100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        
        /* Badges de status */
        .status-badge {{
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            animation: glow 2s ease-in-out infinite alternate;
        }}
        
        .status-online {{
            background: linear-gradient(45deg, #4CAF50, #8BC34A);
            color: white;
            box-shadow: 0 0 10px #4CAF5050;
        }}
        
        .status-warning {{
            background: linear-gradient(45deg, #FF9800, #FFC107);
            color: white;
            box-shadow: 0 0 10px #FF980050;
        }}
        
        .status-error {{
            background: linear-gradient(45deg, #F44336, #E91E63);
            color: white;
            box-shadow: 0 0 10px #F4433650;
        }}
        
        @keyframes glow {{
            from {{ box-shadow: 0 0 5px currentColor; }}
            to {{ box-shadow: 0 0 20px currentColor, 0 0 30px currentColor; }}
        }}
        
        /* Loading spinner customizado */
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid {theme['primary_color']}30;
            border-radius: 50%;
            border-top-color: {theme['primary_color']};
            animation: spin 1s ease-in-out infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Cards de informação */
        .info-card {{
            background: linear-gradient(135deg, {theme['secondary_background']}, {theme['background_color']});
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid {theme['primary_color']}40;
            margin: 1rem 0;
            box-shadow: 0 6px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .info-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 35px rgba(0,0,0,0.15);
            border-color: {theme['primary_color']};
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def render_theme_selector(self):
        """Renderiza seletor de tema na sidebar"""
        st.sidebar.markdown("### 🎨 Personalização")
        
        theme_names = [self.themes[key]['name'] for key in self.themes.keys()]
        theme_keys = list(self.themes.keys())
        
        current_index = theme_keys.index(self.current_theme)
        
        selected_theme = st.sidebar.selectbox(
            "Escolha o tema:",
            options=theme_names,
            index=current_index,
            help="Selecione um tema para personalizar a interface"
        )
        
        # Encontrar a chave do tema selecionado
        selected_key = None
        for key, theme in self.themes.items():
            if theme['name'] == selected_theme:
                selected_key = key
                break
        
        if selected_key and selected_key != self.current_theme:
            self._save_theme_preference(selected_key)
            st.sidebar.success(f"Tema alterado para {selected_theme}!")
            st.rerun()
    
    def create_animated_metric(self, label: str, value: str, delta: str = None):
        """Cria métrica animada customizada"""
        delta_color = "normal"
        if delta:
            if "Excelente" in delta or "Ótimo" in delta or "Rápido" in delta:
                delta_color = "normal" 
            elif "Baixo" in delta or "Lento" in delta:
                delta_color = "inverse"
        
        st.markdown(f'''
        <div class="metric-card">
            <div style="font-size: 0.8rem; color: gray;">{label}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">{value}</div>
            {f'<div style="font-size: 0.8rem; color: {"#00FF00" if delta_color == "normal" else "#FF6B6B"};">{delta}</div>' if delta else ''}
        </div>
        ''', unsafe_allow_html=True)
    
    def create_status_badge(self, text: str, status: str = "online"):
        """Cria badge de status animado"""
        return f'<span class="status-badge status-{status}">{text}</span>'
    
    def create_loading_spinner(self, text: str = "Carregando..."):
        """Cria spinner de loading customizado"""
        return f'''
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="loading-spinner"></div>
            <span>{text}</span>
        </div>
        '''



# Inicializar gerenciador de temas
theme_manager = ThemeManager()

# Aplicar tema e CSS customizado
theme_manager.apply_custom_css()

st.title("🖼️ Extrator de Painéis de Manhwa")
st.markdown('<div id="topo"></div>', unsafe_allow_html=True)

# Constantes otimizadas
MAX_WIDTH = 1024
MIN_CONTOUR_SIZE = 100
REQUEST_TIMEOUT = 15
Y_TOLERANCE = 40
PREVIEW_LIMIT = 20
COLS_PER_ROW = 4
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
        "imagens_processadas": [],  # Corrigido de set() para list()
        "paineis_processados": [],  # Corrigido de set() para list()
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

def fazer_requisicao_web(url: str) -> Optional[BeautifulSoup]:
    """Faz requisição web e retorna BeautifulSoup com cache inteligente"""
    try:
        # Verificar cache primeiro
        cached_content = intelligent_cache.get(url, 'webpage')
        if cached_content:
            logger.debug(f"Requisição web obtida do cache: {url}")
            return BeautifulSoup(cached_content, 'html.parser')
        
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
        
        # Armazenar no cache
        intelligent_cache.set(url, response.content, 'webpage')
        
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        logger.error(f"Erro ao acessar URL {url}: {e}")
        st.error(f"❌ Erro ao acessar URL: {e}")
        return None

def extrair_info_manhwa_manhwatop(soup: BeautifulSoup, base_url: str) -> Dict:
    """Extrai informações do manhwa do ManhwaTop com cache"""
    # Verificar cache primeiro
    cached_info = intelligent_cache.get(base_url, 'chapter_list')
    if cached_info:
        logger.debug(f"Informações do manhwa obtidas do cache: {base_url}")
        return cached_info
    
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
        
        # Armazenar no cache
        intelligent_cache.set(base_url, info, 'chapter_list')
        
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
    """Processa um capítulo completo e extrai todos os painéis (versão otimizada assíncrona)"""
    progress_container = st.empty()
    
    try:
        # Usar versão assíncrona para melhor performance
        paineis_capitulo = process_chapter_async(capitulo_info, progress_container)
        logger.info(f"Processamento assíncrono do capítulo {capitulo_info['numero']} concluído: {len(paineis_capitulo)} painéis")
        return paineis_capitulo
    except Exception as e:
        logger.error(f"Erro no processamento assíncrono do capítulo {capitulo_info['numero']}: {e}")
        st.error(f"Erro no processamento: {e}")
        return []
    finally:
        progress_container.empty()

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
            
            # Criar badge animado
            badge_status = "error" if recent_requests >= limit * 0.8 else ("warning" if recent_requests >= limit * 0.5 else "online")
            badge_html = theme_manager.create_status_badge(
                f"{domain}: {recent_requests}/{limit}",
                badge_status
            )
            st.sidebar.markdown(f"{status} {badge_html}", unsafe_allow_html=True)
else:
    st.sidebar.info("🟢 Nenhuma requisição recente")

# Monitoramento de operações assíncronas
st.sidebar.markdown("### ⚡ Operações Assíncronas")
if hasattr(async_manager, 'session') and async_manager.session:
    if async_manager.session.closed:
        status = "🔴 Sessão fechada"
    else:
        status = "🟢 Sessão ativa"
    st.sidebar.metric("Status HTTP", status)
    
    # Mostrar estatísticas do semáforo
    if hasattr(async_manager, 'semaphore'):
        available = async_manager.semaphore._value
        total = MAX_WORKERS
        used = total - available
        st.sidebar.metric(
            "Conexões simultâneas", 
            f"{used}/{total}",
            help="Número de downloads em paralelo"
        )
else:
    st.sidebar.info("💤 Sessão assíncrona inativa")

# Estatísticas do cache
st.sidebar.markdown("### 💾 Cache Inteligente")
cache_stats = intelligent_cache.get_stats()
if cache_stats['total_items'] > 0:
    st.sidebar.metric(
        "Hit Rate",
        f"{cache_stats['hit_rate']:.1%}",
        delta="Ótimo" if cache_stats['hit_rate'] > 0.8 else "Baixo"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Itens", cache_stats['total_items'])
    with col2:
        st.metric("Memória", f"{cache_stats['memory_usage_mb']:.1f}MB")
    
    # Botão de limpeza rápida
    if st.sidebar.button("🧹 Limpar Cache"):
        intelligent_cache._cleanup_expired()
        st.sidebar.success("Cache limpo!")
        st.rerun()
else:
    st.sidebar.info("📭 Cache vazio")

# Seletor de temas
theme_manager.render_theme_selector()

# Debug info
if st.sidebar.checkbox("🔍 Debug"):
    st.sidebar.write(f"Painéis únicos: {len(st.session_state.paineis_processados)}")
    st.sidebar.write(f"Cache size: {len(st.session_state._cache_hash)}")
    st.sidebar.write(f"Manhwas cache: {len(st.session_state.capitulos_cache)}")

# Abas principais
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🖼️ Extrair Painéis", "🌐 Web Scraping", "📋 Capítulos", "📦 Download", "📊 Métricas", "⚙️ Configurações"])

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
                
                # Versão assíncrona para uploads múltiplos
                async def process_uploads_async():
                    tasks = []
                    for nome, data in files_data:
                        task = async_manager.process_image_async(data, nome)
                        tasks.append((nome, task))
                    
                    for i, (nome, task) in enumerate(tasks):
                        progress_bar.progress((i + 1) / len(tasks))
                        
                        try:
                            paineis, erro = await task
                            if not erro:
                                paineis_novos.extend(paineis)
                                # Mostrar preview após processamento
                                if paineis:
                                    mostrar_paineis_grid_otimizado(paineis, f"📄 {nome}")
                        except Exception as e:
                            logger.error(f"Erro no processamento assíncrono de {nome}: {e}")
                    
                    return paineis_novos
                
                # Executar processamento assíncrono
                try:
                    paineis_novos = run_async(process_uploads_async())
                    st.session_state.painel_coletor.extend(paineis_novos)
                except Exception as e:
                    logger.error(f"Erro no processamento assíncrono de uploads: {e}")
                    st.error(f"Erro no processamento: {e}")
                finally:
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

with tab5:
    st.markdown("# 📊 Dashboard de Performance")
    st.markdown("Monitoramento em tempo real da aplicação")
    
    # Botões de controle
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Atualizar Métricas", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("📊 Exportar Relatório"):
            stats = metrics_collector.get_summary_stats()
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                "📥 Baixar JSON",
                data=stats_json,
                file_name=f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", help="Atualiza métricas automaticamente")
    
    # Dashboard principal
    metrics_collector.display_dashboard()
    
    # Dashboard do cache (TEMPORARIAMENTE DESABILITADO PARA DEBUG)
    try:
        intelligent_cache.display_dashboard()
    except Exception as e:
        st.error(f"⚠️ Erro no dashboard de cache: {e}")
        st.info("💡 Cache funcionando normalmente, apenas dashboard com problema de exibição")
        logger.error(f"Erro no dashboard de cache: {e}", exc_info=True)
    
    # Seção de configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        st.markdown("#### Limites de Alerta")
        
        col1, col2 = st.columns(2)
        with col1:
            success_threshold = st.slider("Taxa de sucesso mínima (%)", 50, 100, 80)
            download_threshold = st.slider("Tempo máximo de download (s)", 1, 20, 5)
        
        with col2:
            processing_threshold = st.slider("Tempo máximo de processamento (s)", 1, 30, 10)
            memory_threshold = st.slider("Limite de memória (MB)", 100, 2000, 1000)
        
        if st.button("💾 Salvar Configurações"):
            st.success("Configurações salvas!")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(10)
        st.rerun()

with tab6:
    st.markdown("# ⚙️ Configurações do Sistema")
    
    st.markdown("""
    <div class="info-card">
        <h3>🎛️ Painel de Configuração</h3>
        <p>Configure todos os aspectos da aplicação através desta interface intuitiva. 
        As configurações são salvas automaticamente e persistem entre sessões.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Editor de configuração
    config_manager.render_config_editor()
    
    st.markdown("---")
    
    # Seção de variáveis de ambiente
    st.markdown("### 🌍 Variáveis de Ambiente")
    st.markdown("""
    Você pode configurar a aplicação usando variáveis de ambiente:
    
    - `MANHWA_DEBUG=true` - Ativa modo debug
    - `MANHWA_LOG_LEVEL=DEBUG` - Define nível de log  
    - `MANHWA_THEME=cyberpunk` - Define tema padrão
    - `MANHWA_CACHE_SIZE=500` - Tamanho do cache (MB)
    - `MANHWA_MAX_WORKERS=8` - Número de workers
    - `MANHWA_REQUEST_TIMEOUT=30` - Timeout de requisições
    """)
    
    # Informações do sistema
    st.markdown("### 💻 Informações do Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Aplicação:**
        - Nome: {config_manager.get('app', 'name')}
        - Versão: {config_manager.get('app', 'version')}
        - Debug: {config_manager.get('app', 'debug')}
        
        **Cache:**
        - Tamanho máximo: {config_manager.get('cache', 'max_memory_mb')}MB
        - TTL imagens: {config_manager.get('cache', 'ttl', {}).get('image', 0)}s
        """)
    
    with col2:
        st.markdown(f"""
        **Scraping:**
        - Timeout: {config_manager.get('scraping', 'request_timeout')}s
        - Max workers: {config_manager.get('scraping', 'max_workers')}
        - User agent: {config_manager.get('scraping', 'user_agent', 'N/A')[:50]}...
        
        **Alertas:**
        - Taxa sucesso min: {config_manager.get('alerts', 'success_rate_threshold')}%
        - Limite memória: {config_manager.get('alerts', 'memory_threshold_gb')}GB
        """)
    
    # Backup e restauração
    st.markdown("### 💾 Backup e Restauração")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Backup Configuração"):
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "config": config_manager.config,
                "version": config_manager.get('app', 'version')
            }
            
            st.download_button(
                "💾 Download Backup",
                json.dumps(backup_data, indent=2, ensure_ascii=False),
                file_name=f"manhwa_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader(
            "📤 Restaurar Configuração",
            type=['json'],
            help="Faça upload de um arquivo de backup de configuração"
        )
        
        if uploaded_config:
            try:
                backup_data = json.load(uploaded_config)
                if 'config' in backup_data:
                    config_manager.config = backup_data['config']
                    config_manager.save_config()
                    st.success("✅ Configuração restaurada com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Arquivo de backup inválido")
            except Exception as e:
                st.error(f"❌ Erro ao restaurar: {e}")
    
    with col3:
        if st.button("🔄 Resetar Tudo"):
            if st.checkbox("⚠️ Confirmar reset completo"):
                config_manager.reset_to_default()
                # Limpar cache também
                intelligent_cache.clear_all()
                st.success("✅ Sistema resetado para padrão!")
                st.rerun()
    
    # Status das configurações
    st.markdown("### 📊 Status das Configurações")
    
    config_status = {
        "Arquivo de config": "✅ Carregado" if os.path.exists("config.json") else "❌ Não encontrado",
        "Logs habilitados": "✅ Sim" if config_manager.get('logging', 'level') != 'DISABLED' else "❌ Não",
        "Cache ativo": "✅ Sim" if config_manager.get('cache', 'max_memory_mb', 0) > 0 else "❌ Não",
        "Rate limiting": "✅ Ativo" if config_manager.get('rate_limits') else "❌ Inativo",
        "Tema customizado": "✅ Sim" if config_manager.get('ui', 'theme') != 'default' else "❌ Padrão"
    }
    
    for item, status in config_status.items():
        st.markdown(f"- **{item}**: {status}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #FF4B4B, #FF6B6B); 
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: bold;">
        🚀 Configuração Profissional • Sistema Enterprise-Ready • Manhwa Extractor v2.0
    </div>
    """, unsafe_allow_html=True)

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


