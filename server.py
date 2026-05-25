# -*- coding: utf-8 -*-
"""
Servidor FastAPI e WebSocket que orquestra a interface web do LLM Pessoal.
Gerencia o treinamento em segundo plano, serve os arquivos estáticos
e responde a requisições de geração de texto (chat) e tokenização.
"""

import os
import urllib.request
import threading
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tokenizer import BPETokenizer
from model import GPTModel
from trainer import LLMTrainer

app = FastAPI(title="LLM Pessoal Workbench")

# Diretório para arquivos estáticos
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Instâncias globais
tokenizer = BPETokenizer(vocab_size=1024) # Vocabulário padrão menor para treino super rápido
trainer = None
training_thread = None
loop = None # Loop de eventos asyncio principal
output_dir = "." # Diretório de saída customizável (Google Drive, USB, etc.)

# Gerenciador de Conexões WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Garante a existência do dataset com fallback offline
def ensure_dataset(out_dir="."):
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(out_dir, "dataset.txt")
    if os.path.exists(dataset_path):
        return dataset_path
        
    print(f"[Server] dataset.txt não encontrado em {out_dir}. Tentando baixar do Project Gutenberg...")
    url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt" # Dom Casmurro
    try:
        urllib.request.urlretrieve(url, dataset_path)
        print("[Server] Dom Casmurro baixado com sucesso!")
    except Exception as e:
        print(f"[Server] Falha ao baixar livro ({e}). Usando corpus em português embutido...")
        # Corpus de fallback em português
        fallback_text = """
Noites de Junho
Noite de junho... Corre o vento frio
Pelas frestas das portas gemedoras;
Ao longe escuto o sussurrar do rio,
E o sino bate as horas passadoras.

Eu gosto destas noites friorentas,
Desta mudez, deste silêncio mudo,
Onde a alma sonha imagens mais violentas,
E o pensamento quer abranger tudo.

O amor é um fogo que arde sem se ver,
é ferida que dói, e não se sente;
é um contentamento descontente,
é dor que desatina sem doer.

É um não querer mais que bem querer;
é um andar solitário entre a gente;
é nunca contentar-se de contente;
é um cuidar que ganha em se perder.

Machado de Assis era um escritor brasileiro de grande renome.
Ele escreveu Dom Casmurro, Memórias Póstumas de Brás Cubas, Quincas Borba, e outros clássicos.
Capitu tinha olhos de ressaca, olhos de cigana oblíqua e dissimulada.
O amor e a dúvida caminham juntos nas páginas desse livro inesquecível.
Treinar um modelo de inteligência artificial requer arquiteturas matemáticas limpas.
O Transformer revolucionou o processamento de linguagem natural no mundo.
Agora estamos rodando nosso próprio LLM pessoal criado do zero!
Sem LLaMA, sem APIs pagas, sem mistérios. Apenas matemática pura e código Python!
""" * 100 # Multiplica para ter tamanho mínimo de dados para treino coerente
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(fallback_text)
        print("[Server] Corpus em português de fallback criado.")
    return dataset_path

@app.on_event("startup")
async def startup_event():
    global loop, trainer, tokenizer, output_dir
    loop = asyncio.get_event_loop()
    ensure_dataset(output_dir)
    
    # Se já existir um tokenizador salvo, carrega
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            tokenizer.load(tokenizer_path)
        except Exception:
            pass

# Modelos Pydantic para validação
class ModelConfigSchema(BaseModel):
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    vocab_size: int = 1024
    learning_rate: float = 0.001
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 1000
    eval_interval: int = 100
    output_dir: str = ""

class ChatPromptSchema(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 10
    top_p: float = 0.9

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Envia status inicial de conexão
        await websocket.send_json({
            "type": "status",
            "is_training": trainer.is_training if trainer else False,
            "is_paused": trainer.is_paused if trainer else False,
            "vocab_size": len(tokenizer.vocab)
        })
        while True:
            # Mantém conexão ativa
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Funções auxiliares para callbacks de treinamento (executados em threads secundárias)
def thread_safe_step_callback(data):
    if loop:
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), loop)

def thread_safe_log_callback(message):
    if loop:
        asyncio.run_coroutine_threadsafe(manager.broadcast({
            "type": "log",
            "message": message
        }), loop)

@app.post("/api/start-train")
async def start_training(config: ModelConfigSchema):
    global trainer, training_thread, tokenizer, output_dir
    
    if trainer and trainer.is_training:
        raise HTTPException(status_code=400, detail="O treinamento já está em execução.")
        
    output_dir = config.output_dir if config.output_dir else "."
    if output_dir != ".":
        os.makedirs(output_dir, exist_ok=True)
        
    dataset_path = ensure_dataset(output_dir)
    
    # Tenta carregar o tokenizador do diretório de destino se ele já existir
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        try:
            tokenizer.load(tokenizer_path)
        except Exception:
            pass
            
    # Se o tamanho de vocabulário mudou, recria o tokenizador
    if len(tokenizer.vocab) != config.vocab_size:
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        
    model_config = {
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "max_seq_len": config.block_size
    }
    
    trainer = LLMTrainer(
        model_config=model_config,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        output_dir=output_dir
    )
    
    # Vincula os callbacks thread-safe
    trainer.on_step_callback = thread_safe_step_callback
    trainer.on_log_callback = thread_safe_log_callback
    
    def run_in_thread():
        try:
            trainer.train(
                max_iters=config.max_iters,
                batch_size=config.batch_size,
                block_size=config.block_size,
                learning_rate=config.learning_rate,
                eval_interval=config.eval_interval
            )
        except Exception as e:
            thread_safe_log_callback(f"ERRO CRÍTICO no treinamento: {str(e)}")
            
    training_thread = threading.Thread(target=run_in_thread, daemon=True)
    training_thread.start()
    
    return {"status": "training_started"}

@app.post("/api/stop-train")
async def stop_training():
    global trainer
    if not trainer or not trainer.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento em execução para parar.")
    trainer.should_stop = True
    return {"status": "stop_requested"}

@app.post("/api/pause-train")
async def pause_training():
    global trainer
    if not trainer or not trainer.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento em execução para pausar.")
    trainer.is_paused = not trainer.is_paused
    return {"status": "paused" if trainer.is_paused else "resumed"}

@app.post("/api/chat")
async def chat_generate(payload: ChatPromptSchema):
    global trainer, tokenizer, output_dir
    
    # Se não houver trainer com modelo carregado, tenta carregar checkpoint existente
    model = None
    if trainer and trainer.model:
        model = trainer.model
    else:
        checkpoint_path = os.path.join(output_dir, "model_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            try:
                # Inicializa um trainer temporário apenas para carregar o modelo
                trainer = LLMTrainer({}, tokenizer, os.path.join(output_dir, "dataset.txt"), output_dir=output_dir)
                trainer.load_model(checkpoint_path)
                model = trainer.model
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")
            
    if model is None:
        raise HTTPException(
            status_code=400, 
            detail="Nenhum modelo treinado ou carregado. Treine o modelo primeiro no painel principal!"
        )
        
    try:
        # Codifica o prompt
        prompt_ids = tokenizer.encode(payload.prompt)
        if not prompt_ids:
            prompt_ids = [0] # Fallback se vazio
            
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        # Gera tokens
        generated_tensor = model.generate(
            prompt_tensor,
            max_new_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p
        )
        
        # Decodifica de volta
        generated_list = generated_tensor[0].tolist()
        # Divide entre o prompt original e o texto recém-gerado
        prompt_len = len(prompt_ids)
        response_text = tokenizer.decode(generated_list[prompt_len:])
        full_text = tokenizer.decode(generated_list)
        
        return {
            "prompt": payload.prompt,
            "response": response_text,
            "full_text": full_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a geração: {str(e)}")

@app.post("/api/tokenize")
async def tokenize_visualizer(payload: dict):
    text = payload.get("text", "")
    if not text:
        return {"tokens": []}
        
    encoded_ids = tokenizer.encode(text)
    
    # Mapeia cada token ID para sua string decodificada e cor para o frontend destacar
    tokens_meta = []
    # Usaremos uma paleta de cores harmoniosas para destacar os tokens consecutivos
    colors = [
        "rgba(147, 51, 234, 0.2)",  # Roxo
        "rgba(59, 130, 246, 0.2)",  # Azul
        "rgba(16, 185, 129, 0.2)",  # Verde
        "rgba(245, 158, 11, 0.2)",  # Laranja
        "rgba(236, 72, 153, 0.2)"   # Rosa
    ]
    
    for i, token_id in enumerate(encoded_ids):
        # Para decodificar de forma individual
        # É mais seguro buscar direto no vocabulário do BPE se disponível
        token_bytes = tokenizer.vocab.get(token_id, b"")
        token_str = token_bytes.decode("utf-8", errors="replace")
        
        # Substitui caracteres especiais para renderização HTML limpa
        display_str = token_str.replace(" ", "•").replace("\n", "↵\n")
        
        tokens_meta.append({
            "id": token_id,
            "text": display_str,
            "color": colors[i % len(colors)]
        })
        
    return {
        "text": text,
        "token_count": len(encoded_ids),
        "tokens": tokens_meta
    }

# Rota curinga para servir o index.html como fallback ou arquivo principal
@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Erro: static/index.html não encontrado!</h1>")

# Monta o diretório de arquivos estáticos para CSS e JS
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
