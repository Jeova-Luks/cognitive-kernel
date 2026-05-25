# -*- coding: utf-8 -*-
"""
Gerenciador e Loop de Treinamento do LLM Pessoal.
Executa o treinamento em segundo plano, calcula métricas, salva checkpoints
e fornece callbacks em tempo real para o servidor WebSocket.
"""

import os
import time
import math
import torch
from model import GPTModel

class LLMTrainer:
    def __init__(self, model_config, tokenizer, dataset_path, device=None, output_dir="."):
        self.model_config = model_config # Dicionário de hiperparâmetros
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Garante que o diretório de saída exista
        if self.output_dir and self.output_dir != ".":
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Escolhe o melhor dispositivo disponível (GPU se CUDA estiver disponível, senão CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.optimizer = None
        
        # Estados de controle de execução
        self.is_training = False
        self.should_stop = False
        self.is_paused = False
        
        # Histórico de perdas para gráficos
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Dados carregados na RAM
        self.train_data = None
        self.val_data = None
        
        # Callbacks para o servidor web transmitir via WebSocket
        self.on_step_callback = None
        self.on_log_callback = None

    def log(self, message):
        print(f"[Trainer] {message}")
        if self.on_log_callback:
            self.on_log_callback(message)

    def prepare_data(self):
        """Carrega e tokeniza o corpus de dados."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Arquivo de dataset não encontrado em: {self.dataset_path}")
            
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        self.log(f"Carregando dataset de {len(text):,} caracteres...")
        
        # Treina ou atualiza o tokenizador se ele estiver vazio
        if len(self.tokenizer.vocab) <= 256:
            self.log("Treinando tokenizador com o dataset fornecido...")
            self.tokenizer.train(text, verbose=False)
            # Salva o tokenizador na pasta do projeto
            tokenizer_path = os.path.join(self.output_dir, "tokenizer.json")
            self.tokenizer.save(tokenizer_path)
            
        self.log("Codificando dataset completo...")
        token_ids = self.tokenizer.encode(text)
        self.log(f"Dataset convertido em {len(token_ids):,} tokens.")
        
        # Split treino (90%) e validação (10%)
        n = int(0.9 * len(token_ids))
        self.train_data = torch.tensor(token_ids[:n], dtype=torch.long)
        self.val_data = torch.tensor(token_ids[n:], dtype=torch.long)
        self.log(f"Dados de Treino: {len(self.train_data):,} tokens. Validação: {len(self.val_data):,} tokens.")

    def get_batch(self, split, batch_size, block_size):
        """Busca um lote (batch) de inputs X e targets Y apropriados."""
        data = self.train_data if split == "train" else self.val_data
        
        # Sorteia offsets aleatórios no corpus de dados
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        # Move dados para a GPU se disponível
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters, batch_size, block_size):
        """Calcula a média de perda para treino e validação de forma estável."""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y = self.get_batch(split, batch_size, block_size)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_lr(self, it, max_iters, learning_rate, warmup_iters):
        """Calcula a taxa de aprendizado com decaimento de cosseno e aquecimento (warmup)."""
        # 1) Warmup linear nas primeiras iterações
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) Se ultrapassar o limite, retorna o mínimo tolerado
        if it > max_iters:
            return learning_rate * 0.1
        # 3) Decaimento de cosseno no intervalo principal
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * (0.1 + 0.9 * coeff)

    def train(self, max_iters=2000, batch_size=32, block_size=128, learning_rate=1e-3, 
              eval_interval=100, eval_iters=20, warmup_iters=100, clip_grad=1.0):
        """Executa o loop principal de otimização em PyTorch."""
        self.is_training = True
        self.should_stop = False
        self.is_paused = False
        
        self.log(f"Iniciando treinamento no dispositivo: {self.device}")
        
        # Inicializa o modelo se ainda não existir
        if self.model is None:
            self.model = GPTModel(
                vocab_size=len(self.tokenizer.vocab),
                n_embd=self.model_config.get("n_embd", 256),
                n_head=self.model_config.get("n_head", 8),
                n_layer=self.model_config.get("n_layer", 6),
                max_seq_len=block_size
            )
        self.model.to(self.device)
        self.model.train()
        
        # Inicializa o otimizador AdamW com decaimento de peso (weight decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            weight_decay=0.01
        )
        
        # Garante dados carregados
        if self.train_data is None:
            self.prepare_data()

        # Variáveis de benchmark de velocidade
        t0 = time.time()
        running_tokens_per_sec = 0.0
        
        step = 0
        while step < max_iters:
            # Gerenciamento de Pausa
            if self.is_paused:
                time.sleep(0.5)
                continue
                
            # Gerenciamento de Parada Solicitada
            if self.should_stop:
                self.log("Treinamento interrompido pelo usuário.")
                break

            # Ajusta taxa de aprendizado dinâmica (Cosine Decay)
            lr = self.get_lr(step, max_iters, learning_rate, warmup_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Carrega lote de dados
            x, y = self.get_batch("train", batch_size, block_size)
            
            # Forward pass e computação de perda (loss)
            t_forward_start = time.time()
            logits, loss = self.model(x, y)
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping para conter explosão de gradientes
            if clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                
            self.optimizer.step()
            
            # Medição de velocidade (tokens por segundo)
            t_forward_end = time.time()
            dt = t_forward_end - t_forward_start
            tokens_in_batch = batch_size * block_size
            tokens_per_sec = tokens_in_batch / max(dt, 1e-6)
            
            # Suavização da métrica de tokens por segundo
            if running_tokens_per_sec == 0:
                running_tokens_per_sec = tokens_per_sec
            else:
                running_tokens_per_sec = 0.9 * running_tokens_per_sec + 0.1 * tokens_per_sec

            # Avaliação periódica no conjunto de validação e logs
            if step % eval_interval == 0 or step == max_iters - 1:
                t_eval_start = time.time()
                losses = self.estimate_loss(eval_iters, batch_size, block_size)
                t_eval_end = time.time()
                
                self.train_loss_history.append((step, losses["train"]))
                self.val_loss_history.append((step, losses["val"]))
                
                # Gera uma amostra rápida do que o modelo consegue fazer neste momento
                # Pega as primeiras palavras do dataset como semente de prompt
                seed_prompt = "O "
                prompt_ids = torch.tensor([self.tokenizer.encode(seed_prompt)], dtype=torch.long, device=self.device)
                
                generated_ids = self.model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=10)
                generated_text = self.tokenizer.decode(generated_ids[0].tolist())
                
                self.log(
                    f"Passo {step}/{max_iters} | "
                    f"Perda Treino: {losses['train']:.4f} | Perda Val: {losses['val']:.4f} | "
                    f"LR: {lr:.2e} | Velocidade: {running_tokens_per_sec:.0f} tok/s"
                )
                self.log(f"Amostra Gerada: \"{generated_text}\"")
                
                # Dispara o callback informando o painel web
                if self.on_step_callback:
                    self.on_step_callback({
                        "type": "progress",
                        "step": step,
                        "max_iters": max_iters,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "learning_rate": lr,
                        "tokens_per_sec": int(running_tokens_per_sec),
                        "sample": generated_text,
                        "elapsed_time": int(time.time() - t0)
                    })

            step += 1
            
        # Fim do treinamento
        self.is_training = False
        self.log("Fim do loop de treinamento.")
        
        # Salva o checkpoint final do modelo
        checkpoint_path = os.path.join(self.output_dir, "model_checkpoint.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "vocab_size": len(self.tokenizer.vocab),
            "step": step,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history
        }, checkpoint_path)
        self.log(f"Modelo final salvo com sucesso em: {checkpoint_path}")
        
        if self.on_step_callback:
            self.on_step_callback({
                "type": "completed",
                "checkpoint": checkpoint_path
            })

    def load_model(self, checkpoint_path):
        """Carrega um modelo pré-salvo de um arquivo de checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = checkpoint["model_config"]
        
        self.model = GPTModel(
            vocab_size=checkpoint["vocab_size"],
            n_embd=self.model_config.get("n_embd", 256),
            n_head=self.model_config.get("n_head", 8),
            n_layer=self.model_config.get("n_layer", 6),
            max_seq_len=self.model_config.get("max_seq_len", 128)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.train_loss_history = checkpoint.get("train_loss_history", [])
        self.val_loss_history = checkpoint.get("val_loss_history", [])
        self.log(f"Modelo carregado com sucesso de {checkpoint_path}!")
        return self.model
