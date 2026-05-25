# -*- coding: utf-8 -*-
"""
Script de execução automática para o LLM Pessoal.
Verifica as dependências, instala qualquer pacote ausente e inicia o servidor FastAPI.
"""

import sys
import subprocess
import os

def check_and_install_dependencies():
    print("=" * 60)
    print("   VERIFICANDO DEPENDÊNCIAS DO SEU LLM PESSOAL DO ZERO   ")
    print("=" * 60)
    
    dependencies = {
        "torch": "PyTorch (essencial para rodar a rede neural)",
        "fastapi": "FastAPI (para criar as rotas do backend)",
        "uvicorn": "Uvicorn (para rodar o servidor ASGI)",
        "websockets": "WebSockets (para transmitir métricas de treino em tempo real)"
    }
    
    missing_packages = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"[✓] {package} já está instalado.")
        except ImportError:
            print(f"[✗] {package} está ausente ({description}).")
            missing_packages.append(package)
            
    if missing_packages:
        print("\nInstalando dependências ausentes via pip...")
        try:
            # Tenta rodar o pip install para os pacotes ausentes
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("[✓] Todas as dependências foram instaladas com sucesso!")
        except Exception as e:
            print(f"\n[!] Falha automática ao rodar o pip: {e}")
            print("Por favor, execute o seguinte comando manualmente no seu terminal:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
            
    print("\n[✓] Tudo pronto para iniciar o workbench do LLM!")
    print("=" * 60)

def start_server():
    print("\nIniciando servidor local na porta 8000...")
    print("Abra seu navegador em: http://127.0.0.1:8000")
    print("=" * 60)
    
    try:
        import uvicorn
        # Roda o servidor uvicorn apontando para o app FastAPI no arquivo server.py
        uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nServidor finalizado pelo usuário. Até logo!")
    except Exception as e:
        print(f"\n[!] Erro ao rodar o servidor: {e}")

if __name__ == "__main__":
    check_and_install_dependencies()
    start_server()
