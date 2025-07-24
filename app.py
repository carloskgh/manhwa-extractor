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

def baixar_modelo_yolo(dropbox_url, destino="modelos/best.pt"):
    if not os.path.exists(destino):
        try:
            st.warning("📦 Baixando modelo YOLO (best.pt)...")
            # Corrigir o link para download direto
            url_download = dropbox_url.replace("?dl=0", "?dl=1")
            response = requests.get(url_download, stream=True)
            response.raise_for_status()

            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            st.success("✅ Modelo YOLO baixado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao baixar o modelo: {e}")
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

# URL do modelo no Dropbox
URL_MODELO_DROPBOX = "https://www.dropbox.com/scl/fi/a743aqjqzau3fxy4fss4a/best.pt?rlkey=a24lozm0cw8znku0h743ylx2z&dl=0"

# Baixar automaticamente se necessário
baixar_modelo_yolo(URL_MODELO_DROPBOX)

# Carregar modelo normalmente
model = carregar_modelo("best.pt")

# Carregamento do modelo otimizado
@st.cache_resource
def carregar_modelo("best.pt"):
    try:
        modelo = YOLO("best.pt")
        modelo.overrides['verbose'] = False
        modelo.overrides['device'] = 'cpu'
        
        dummy_test = np.zeros((64, 64, 3), dtype=np.uint8)
        modelo(dummy_test, verbose=False)
        return modelo
    except FileNotFoundError:
        st.error("❌ Arquivo 'best.pt' não encontrado. Usando apenas detecção por contorno.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        st.info("💡 Usando apenas detecção por contorno...")
        return None

model = carregar_modelo("best.pt")

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
    except:
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
                nome_arquivo = f"painel_{st.session_state.contador_paineis:06}.png"
                resultados.append((painel["img"], nome_arquivo))
                st.session_state.paineis_processados.add(hash_painel)
    
    return resultados

def processar_imagem_otimizada(img_data: bytes, nome_fonte: str) -> Tuple[List, Optional[str]]:
    try:
        img_pil = carregar_e_redimensionar_imagem(img_data)
        if img_pil is None:
            return [], "Erro ao carregar imagem"
            
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        paineis = extrair_paineis_hibrido_otimizado(img, img_id=nome_fonte)
        return paineis, None
    except Exception as e:
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
    
    return bool(url_pattern.match(url))

def baixar_imagem_url_otimizada(url: str) -> Optional[bytes]:
    try:
        if not validar_url_cached(url):
            return None
            
        response = requests.get(url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > 50 * 1024 * 1024:
                raise ValueError("Imagem muito grande")
        
        return content
    except Exception as e:
        return None

# NOVAS FUNÇÕES PARA WEB SCRAPING

@st.cache_data(ttl=300, show_spinner=False)  # Cache por 5 minutos
def fazer_requisicao_web(url: str) -> Optional[BeautifulSoup]:
    """Faz requisição web e retorna BeautifulSoup"""
    try:
        response = requests.get(url, headers=SCRAPING_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        st.error(f"❌ Erro ao acessar URL: {e}")
        return None

def extrair_info_manhwa_manhwatop(soup: BeautifulSoup, base_url: str) -> Dict:
    """Extrai informações do manhwa do ManhwaTop"""
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
        
    except Exception as e:
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
        
        time.sleep(0.5)  # Pequena pausa para não sobrecarregar o servidor
    
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
            files_data = [(f.name, f.read()) for f in arquivos]
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
        if not validar_url_cached(url_manhwa):
            st.error("❌ URL inválida!")
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
                except:
                    st.write("🖼️ Capa não disponível")
        
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
            capitulos_selecionados = [
                cap for cap in info['capitulos'] 
                if range_inicio <= float(cap['numero']) <= range_fim
            ]
            
            if capitulos_selecionados:
                total_paineis = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, cap in enumerate(capitulos_selecionados):
                    status_text.text(f"Processando Capítulo {cap['numero']}...")
                    progress_bar.progress((i + 1) / len(capitulos_selecionados))
                    
                    paineis_cap = processar_capitulo_completo(cap)
                    total_paineis.extend(paineis_cap)
                    
                    time.sleep(1)  # Pausa entre capítulos
                
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
