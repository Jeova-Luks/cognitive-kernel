# -*- coding: utf-8 -*-
"""
Arquitetura completa do Transformer (Decoder-only) escrita 100% do zero usando PyTorch.
Inclui inovações modernas como RMSNorm, Rotary Position Embeddings (RoPE) e SwiGLU.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Substitui o LayerNorm tradicional eliminando a média e escalando apenas pela variância residual.
    Usado em LLMs modernos como LLaMA, Mistral e Gemma.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Aplica uma rotação vetorial nos espaços de Query e Key para injetar informação de posição relativa.
    Permite maior generalização para contextos longos.
    """
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Cria as frequências de rotação de acordo com a fórmula do RoPE
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pré-computa cos e sin para posições até max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Duplica as frequências para cobrir a dimensão completa dos vetores (dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        """Rotaciona a metade dos canais dos vetores."""
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, seq_len):
        # x tem dimensão: [batch_size, num_heads, seq_len, head_dim]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, head_dim]
        
        # x_rotated = x * cos(pos) + rotate_half(x) * sin(pos)
        return (x * cos.to(x.device)) + (self._rotate_half(x) * sin.to(x.device))


class CausalSelfAttention(nn.Module):
    """
    Mecanismo de Multi-Head Causal Self-Attention com suporte a RoPE.
    Garante que o modelo apenas preste atenção em tokens passados (máscara causal).
    """
    def __init__(self, n_embd, n_head, max_seq_len, n_kv_head=None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        assert n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.head_dim = n_embd // n_head
        self.max_seq_len = max_seq_len
        self.kv_dim = self.head_dim * self.n_kv_head  # smaller K/V projection when GQA

        # Q is full-size; K and V are reduced when n_kv_head < n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x):
        B, T, C = x.size()

        # Q is [B, n_head, T, head_dim]; K/V are [B, n_kv_head, T, head_dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Rotary position embedding on Q and K (same RoPE works for both)
        q = self.rope(q, T)
        k = self.rope(k, T)

        # Expand K/V so each query group sees the right K/V head
        if self.n_kv_head < self.n_head:
            n_repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_repeat, dim=1)
            v = v.repeat_interleave(n_repeat, dim=1)

        # Flash Attention 2 when available; falls back to math on CPU.
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class SwiGLUMLP(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Multi-Layer Perceptron.
    Uma evolução moderna do MLP convencional que usa ativação Swish com portas (gating).
    Provê ganho substancial de expressividade e qualidade em comparação a ReLU/GeLU.
    """
    def __init__(self, n_embd, intermediate_dim=None):
        super().__init__()
        if intermediate_dim is None:
            # Padrão de dimensionamento aproximado de 8/3 da dimensão de embedding
            intermediate_dim = int(2 * (4 * n_embd / 3))
            
        # w1 é a porta de gating, w2 é a projeção principal
        self.w1 = nn.Linear(n_embd, intermediate_dim, bias=False)
        self.w2 = nn.Linear(n_embd, intermediate_dim, bias=False)
        # w3 faz a projeção de volta para a dimensão original
        self.w3 = nn.Linear(intermediate_dim, n_embd, bias=False)

    def forward(self, x):
        # SwiGLU(x) = (Silu(xW1) * xW2) * W3
        # Silu é matematicamente equivalente à ativação Swish
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """
    Bloco clássico do Transformer contendo Self Attention e MLP (SwiGLU) com RMSNorm.
    Usa arquitetura Pre-LN (Normalização antes das operações) para maior estabilidade.
    """
    def __init__(self, n_embd, n_head, max_seq_len, n_kv_head=None):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, n_kv_head=n_kv_head)

        self.mlp_norm = RMSNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd)

    def forward(self, x):
        # Conexões residuais
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPTModel(nn.Module):
    """
    Modelo GPT Decoder-only completo parametrizado.
    Combina embedding de tokens, múltiplos blocos Transformer e projeção de saída para logits.
    """
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_kv_head=None,
                 n_layer=6, max_seq_len=128, grad_checkpoint=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.grad_checkpoint = grad_checkpoint

        # Camada de Embedding dos Tokens de entrada
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)

        # Pilha de blocos Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, max_seq_len, n_kv_head=n_kv_head)
            for _ in range(n_layer)
        ])
        
        # Normalização final
        self.norm_f = RMSNorm(n_embd)
        
        # Cabeça de saída que projeta os embeddings para probabilidades sobre o vocabulário
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Compartilha os pesos do embedding de tokens com a projeção de saída (Weight Tying)
        # Reduz pela metade a memória gasta com parâmetros de vocabulário e acelera o treino!
        self.token_embeddings.weight = self.lm_head.weight
        
        # Inicializa todos os parâmetros de forma cuidadosa
        self.apply(self._init_weights)
        print(f"Modelo LLM Pessoal criado com sucesso! Parâmetros totais: {self.get_num_params():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Inicialização normal de Xavier/Glorot com desvio padrão ajustado
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Retorna a contagem total de parâmetros do modelo."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.max_seq_len, \
            f"Input length ({T}) exceeds configured context window ({self.max_seq_len})."

        x = self.token_embeddings(idx)

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Gera sequencialmente novos tokens baseados em um prompt inicial de IDs (idx).
        Suporta temperatura de amostragem, Top-K e Top-P (Nucleus) Filtering.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Corta a sequência para caber no limite físico da nossa janela de contexto (max_seq_len)
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Obtém os logits da última posição
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Foco apenas no último token previsto: [B, vocab_size]
            
            if temperature == 0:
                # Amostragem gulosa (greedy decode): pega sempre o maior logit
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Aplica temperatura de amostragem
                logits = logits / temperature
                
                # Aplica filtro Top-K se configurado
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Aplica filtro Top-P (Nucleus Sampling) se configurado
                if top_p is not None and top_p > 0.0 and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens que excedem o limiar de probabilidade acumulada (mantém o primeiro excedente)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Desloca a máscara de remoção para a direita para não remover o primeiro que passa do limite
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Espalha a máscara para os logits originais
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Calcula probabilidades softmax
                probs = F.softmax(logits, dim=-1)
                
                # Sorteia o próximo token a partir da distribuição categórica de probabilidades
                next_token = torch.multinomial(probs, num_samples=1)
                
            # Concatena o novo token na sequência gerada até agora
            idx = torch.cat((idx, next_token), dim=1)
            
        return idx


# Teste rápido se executado diretamente
if __name__ == "__main__":
    # Testa integridade da rede
    B, T, V = 4, 32, 256 # Batch size = 4, contexto = 32 tokens, vocabulário = 256
    model = GPTModel(vocab_size=V, n_embd=64, n_head=4, n_layer=3, max_seq_len=128)
    
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    
    logits, loss = model(x, y)
    print("Logits shape:", logits.shape)
    print("Loss inicial (esperado ~ ln(V)):", loss.item(), "Esperado:", math.log(V))
    
    # Testa geração básica
    prompt = torch.randint(0, V, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print("Comprimento gerado total:", generated.shape[1], "Esperado: 15")
