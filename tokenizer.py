# -*- coding: utf-8 -*-
"""
Tokenizador Byte-Pair Encoding (BPE) escrito 100% do zero.
Permite treinar um vocabulário customizado a partir de um corpus de texto,
além de codificar (encode) e decodificar (decode) strings.
"""

import json
import os

class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        # Inicializa o vocabulário básico com os 256 bytes individuais
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # Mapeia (int, int) -> int (o novo token mesclado)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def _get_stats(self, ids):
        """Conta a frequência de pares consecutivos de inteiros em uma lista."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        """Substitui todas as ocorrências consecutivas de 'pair' em 'ids' pelo novo token 'idx'."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, verbose=False):
        """Treina o tokenizador BPE no texto fornecido até atingir o vocab_size desejado."""
        if not text:
            raise ValueError("O texto para treinamento não pode ser vazio.")
            
        # Converte a string de entrada para uma lista de inteiros correspondentes a bytes (0-255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        num_merges = self.vocab_size - 256
        if num_merges <= 0:
            print("vocab_size menor ou igual a 256. Nenhum treinamento necessário.")
            return

        print(f"Treinando BPE Tokenizer no texto ({len(text)} caracteres)...")
        print(f"Vocabulário inicial: 256 tokens básicos. Alvo: {self.vocab_size} tokens.")
        
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break  # Não há mais pares para mesclar
            
            # Encontra o par mais comum
            best_pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            
            # Executa a mesclagem na nossa sequência
            ids = self._merge(ids, best_pair, new_token_id)
            
            # Registra a mesclagem e atualiza o vocabulário
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_token_id} (ocorrências: {stats[best_pair]})")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Treinamento concluído! Tamanho final do vocabulário: {len(self.vocab)}")

    def encode(self, text):
        """Codifica o texto de entrada em uma lista de IDs de tokens."""
        if not text:
            return []
            
        # Começa convertendo a string para seus bytes individuais (0-255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # Se temos merges cadastrados, aplicamos na ordem em que foram criados no treino
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Encontra o par elegível que aparece primeiro nas nossas regras de mesclagem (menor ID resultante)
            pair_to_merge = None
            min_merge_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_merge_idx:
                        min_merge_idx = self.merges[pair]
                        pair_to_merge = pair
                        
            if pair_to_merge is None:
                break # Nenhum par elegível encontrado para mesclar
                
            ids = self._merge(ids, pair_to_merge, self.merges[pair_to_merge])
            
        return ids

    def decode(self, ids):
        """Decodifica uma lista de IDs de tokens de volta para uma string do usuário."""
        if not ids:
            return ""
            
        # Concatena todos os bytes mapeados pelos IDs correspondentes
        text_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        # Decodifica de volta para UTF-8 substituindo erros de forma segura
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        """Salva o vocabulário e os merges em um arquivo JSON."""
        # Não podemos usar tuplas como chaves em JSON, então convertemos as chaves de merges para strings
        str_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            "vocab_size": self.vocab_size,
            "merges": str_merges
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Tokenizador salvo com sucesso em: {filepath}")

    def load(self, filepath):
        """Carrega o vocabulário e os merges de um arquivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        str_merges = data["merges"]
        
        # Reconstrói os merges com as tuplas corretas
        self.merges = {}
        for k, v in str_merges.items():
            p1, p2 = map(int, k.split(","))
            self.merges[(p1, p2)] = v
            
        # Reconstrói o vocabulário a partir das regras de mesclagem
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, token_id in sorted(self.merges.items(), key=lambda item: item[1]):
            self.vocab[token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizador carregado! Vocabulário de {len(self.vocab)} tokens obtido de {filepath}")


# Teste rápido se rodado diretamente
if __name__ == "__main__":
    text = "O rato roeu a roupa do rei de Roma. E o rei de Roma ficou com raiva!"
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(text, verbose=True)
    
    encoded = tokenizer.encode("rato roeu")
    print("\nCodificado 'rato roeu':", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decodificado:", decoded)
    
    # Testa salvar e carregar
    tokenizer.save("test_tokenizer.json")
    
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("test_tokenizer.json")
    print("Decodificado pelo novo:", new_tokenizer.decode(new_tokenizer.encode("rato roeu")))
    
    os.remove("test_tokenizer.json")
