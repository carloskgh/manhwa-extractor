# 🖼️ Manhwa Panel Extractor (YOLOv8 + Contorno)

Este projeto em Python usa uma **abordagem híbrida com YOLOv8 e detecção por contorno** para extrair painéis de capítulos de manhwa a partir de imagens ou URLs. Desenvolvido com Streamlit, ele oferece uma interface simples para upload, scraping e exportação dos painéis em `.zip`.

## 🚀 Funcionalidades

- 📥 Upload de imagens ou scraping de capítulos inteiros via URL
- 🔍 Detecção automática de painéis com YOLOv8 e fallback com contornos (OpenCV)
- 📦 Exportação em `.zip` com todos os painéis extraídos
- 🌐 Suporte a scraping de sites como ManhwaTop, ReaperScans e mais

## 🧠 Tecnologias Usadas

- YOLOv8 (`ultralytics`)
- OpenCV para detecção por contornos
- Streamlit para interface
- BeautifulSoup para scraping

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/manhwa-extractor.git
cd manhwa-extractor
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Adicione o peso do YOLOv8 (ex: `best.pt`) na pasta `modelos/`.

4. Rode o aplicativo:

```bash
streamlit run app.py
```

## 📁 Estrutura

```
app.py              # Aplicativo principal
requirements.txt    # Dependências do projeto
README.md           # Instruções e documentação
modelos/            # Pesos do modelo YOLOv8 (ex: best.pt)
```

## 📌 Observações

- Se o modelo YOLO não estiver presente ou falhar, a detecção por contorno será usada como fallback.
- O arquivo `best.pt` **não está incluído** por questões de licença/tamanho — adicione manualmente.

## 📜 Licença

MIT

---

Desenvolvido por [Seu Nome ou GitHub](https://github.com/seu-usuario)
