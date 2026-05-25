/* ==========================================================================
   FRONTEND LOGIC - LLM PESSOAL WORKBENCH (VANILLA JS & WEBSOCKETS)
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ESTADO GLOBAL
    let ws = null;
    let lossChart = null;
    let isTraining = false;
    let isPaused = false;
    let lossHistoryTrain = [];
    let lossHistoryVal = [];
    let stepLabels = [];
    
    // ELEMENTOS DOM
    const tabs = document.querySelectorAll(".nav-tab");
    const panes = document.querySelectorAll(".tab-pane");
    const wsStatusDot = document.querySelector("#ws-status .status-dot");
    const wsStatusText = document.querySelector("#ws-status .status-text");
    const wsStatusBadge = document.querySelector("#ws-status");
    
    // ELEMENTOS DA CONFIGURAÇÃO
    const configForm = document.querySelector("#config-form");
    const saveConfigBtn = document.querySelector("#save-config-btn");
    const tokenizerInput = document.querySelector("#tokenizer-input");
    const tokenCount = document.querySelector("#token-count");
    const tokenizerVisualOutput = document.querySelector("#tokenizer-visual-output");

    // ELEMENTOS DO TREINO
    const startTrainBtn = document.querySelector("#start-train-btn");
    const pauseTrainBtn = document.querySelector("#pause-train-btn");
    const stopTrainBtn = document.querySelector("#stop-train-btn");
    const trainSpinner = document.querySelector("#train-spinner");
    const trainStateTitle = document.querySelector("#train-state-title");
    const trainStateDesc = document.querySelector("#train-state-desc");
    
    const valTrainLoss = document.querySelector("#val-train-loss");
    const valValLoss = document.querySelector("#val-val-loss");
    const valSpeed = document.querySelector("#val-speed");
    const valStep = document.querySelector("#val-step");
    
    const liveSampleOutput = document.querySelector("#live-sample-output");
    const terminalLogs = document.querySelector("#terminal-logs");

    // ELEMENTOS DO CHAT PLAYGROUND
    const chatInput = document.querySelector("#chat-input");
    const chatSendBtn = document.querySelector("#chat-send-btn");
    const chatMessagesBox = document.querySelector("#chat-messages-box");
    const clearChatBtn = document.querySelector("#clear-chat-btn");
    const chatModelStatus = document.querySelector("#chat-model-status");
    
    const genTemperature = document.querySelector("#gen-temperature");
    const genTopK = document.querySelector("#gen-topk");
    const genTopP = document.querySelector("#gen-topp");
    const genTokens = document.querySelector("#gen-tokens");
    
    const valTemp = document.querySelector("#val-temp");
    const valTopK = document.querySelector("#val-topk");
    const valTopP = document.querySelector("#val-topp");
    const valTokens = document.querySelector("#val-tokens");

    // ==========================================================================
    // SISTEMA DE ABAS (Navegação)
    // ==========================================================================
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(`pane-${tab.dataset.tab}`);
            if (targetPane) {
                targetPane.classList.add("active");
            }
            
            // Corrige redimensionamento do gráfico ao trocar para a aba de treino
            if (tab.dataset.tab === "train" && lossChart) {
                setTimeout(() => lossChart.resize(), 100);
            }
        });
    });

    // ==========================================================================
    // SLIDERS DINÂMICOS
    // ==========================================================================
    genTemperature.addEventListener("input", (e) => valTemp.textContent = parseFloat(e.target.value).toFixed(1));
    genTopK.addEventListener("input", (e) => valTopK.textContent = parseInt(e.target.value));
    genTopP.addEventListener("input", (e) => valTopP.textContent = parseFloat(e.target.value).toFixed(2));
    genTokens.addEventListener("input", (e) => valTokens.textContent = parseInt(e.target.value));

    // ==========================================================================
    // INICIALIZAÇÃO DO GRÁFICO (Chart.js)
    // ==========================================================================
    function initChart() {
        const ctx = document.getElementById("lossChart").getContext("2d");
        
        // Se já existe, destrói para resetar
        if (lossChart) {
            lossChart.destroy();
        }
        
        lossChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: stepLabels,
                datasets: [
                    {
                        label: "Perda Treino (Train Loss)",
                        data: lossHistoryTrain,
                        borderColor: "#9333ea",
                        backgroundColor: "rgba(147, 51, 234, 0.1)",
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: "Perda Validação (Val Loss)",
                        data: lossHistoryVal,
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.05)",
                        borderWidth: 2,
                        tension: 0.3,
                        fill: false,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: "#9ca3af",
                            font: { family: "Inter", weight: 600 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#9ca3af", font: { family: "JetBrains Mono", size: 10 } }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.03)" },
                        ticks: { color: "#9ca3af", font: { family: "JetBrains Mono", size: 10 } }
                    }
                }
            }
        });
    }
    initChart();

    // ==========================================================================
    // SISTEMA WEBSOCKET (Logs e Otimizações em Tempo Real)
    // ==========================================================================
    function connectWS() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            wsStatusBadge.className = "status-badge online";
            wsStatusText.textContent = "Conectado";
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Roteamento de mensagens do WebSocket do backend
            switch(data.type) {
                case "status":
                    updateTrainStatus(data.is_training, data.is_paused);
                    if (data.vocab_size > 256) {
                        chatModelStatus.textContent = `Modelo treinado (vocab: ${data.vocab_size} tokens)`;
                        chatModelStatus.style.color = "#10b981";
                    }
                    break;
                    
                case "log":
                    appendTerminalLog(data.message);
                    break;
                    
                case "progress":
                    updateProgressMetrics(data);
                    break;
                    
                case "completed":
                    handleTrainingCompletion(data.checkpoint);
                    break;
            }
        };
        
        ws.onclose = () => {
            wsStatusBadge.className = "status-badge";
            wsStatusText.textContent = "Desconectado";
            // Reconecta a cada 3 segundos
            setTimeout(connectWS, 3000);
        };
    }
    connectWS();

    // ==========================================================================
    // TOKENIZADOR BPE INTERATIVO
    // ==========================================================================
    let tokenizerTimeout = null;
    
    tokenizerInput.addEventListener("input", () => {
        clearTimeout(tokenizerTimeout);
        
        // Debounce de 300ms para evitar chamadas de API excessivas
        tokenizerTimeout = setTimeout(() => {
            const text = tokenizerInput.value;
            if (!text) {
                tokenCount.textContent = "0";
                tokenizerVisualOutput.textContent = "Aguardando digitação...";
                return;
            }
            
            fetch("/api/tokenize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            })
            .then(res => res.json())
            .then(data => {
                tokenCount.textContent = data.token_count;
                
                // Limpa visualização
                tokenizerVisualOutput.innerHTML = "";
                
                data.tokens.forEach(tok => {
                    const span = document.createElement("span");
                    span.className = "token-chunk";
                    span.style.backgroundColor = tok.color;
                    span.setAttribute("data-id", tok.id);
                    span.textContent = tok.text;
                    tokenizerVisualOutput.appendChild(span);
                });
            })
            .catch(err => {
                console.error("Falha ao tokenizar:", err);
            });
        }, 300);
    });

    // ==========================================================================
    // ORQUESTRADOR DE TREINAMENTO (APIs HTTP)
    // ==========================================================================
    
    // 1) Configura / Inicializa Vocab
    saveConfigBtn.addEventListener("click", () => {
        const formData = new FormData(configForm);
        const config = Object.fromEntries(formData.entries());
        
        // Força tipos numéricos apropriados
        const payload = {
            n_embd: parseInt(config.n_embd),
            n_head: parseInt(config.n_head),
            n_layer: parseInt(config.n_layer),
            vocab_size: parseInt(config.vocab_size),
            learning_rate: parseFloat(config.learning_rate),
            batch_size: parseInt(config.batch_size),
            block_size: parseInt(config.block_size),
            max_iters: parseInt(config.max_iters),
            eval_interval: parseInt(config.eval_interval),
            output_dir: config.output_dir ? config.output_dir.trim() : ""
        };
        
        saveConfigBtn.disabled = true;
        saveConfigBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Inicializando Tokenizador BPE...`;
        
        appendTerminalLog(`[System] Treinando tokenizador BPE para vocabulário de ${payload.vocab_size} tokens com base no dataset...`);

        fetch("/api/start-train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === "training_started") {
                // Limpa gráficos anteriores
                stepLabels = [];
                lossHistoryTrain = [];
                lossHistoryVal = [];
                initChart();
                
                // Troca aba para o painel de treino
                document.querySelector("[data-tab='train']").click();
                
                updateTrainStatus(true, false);
            }
        })
        .catch(err => {
            alert("Erro ao iniciar treinamento. Verifique se o servidor está ativo.");
            console.error(err);
        })
        .finally(() => {
            saveConfigBtn.disabled = false;
            saveConfigBtn.innerHTML = `<i class="fa-solid fa-circle-check"></i> Inicializar Modelo e Carregar Dataset`;
        });
    });

    // Controles Rápidos do Painel de Treino
    startTrainBtn.addEventListener("click", () => {
        saveConfigBtn.click(); // Dispara o fluxo completo
    });

    pauseTrainBtn.addEventListener("click", () => {
        fetch("/api/pause-train", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            isPaused = (data.status === "paused");
            updateTrainStatus(isTraining, isPaused);
        });
    });

    stopTrainBtn.addEventListener("click", () => {
        if (confirm("Tem certeza que deseja interromper o treinamento? O progresso atual será cortado, mas você poderá salvar o modelo parcial.")) {
            fetch("/api/stop-train", { method: "POST" })
            .then(res => res.json())
            .then(() => {
                appendTerminalLog("[System] Interrupção enviada. Finalizando iteração corrente...");
            });
        }
    });

    // AUXILIARES DE RENDERIZAÇÃO E LOGS
    function appendTerminalLog(message) {
        const p = document.createElement("p");
        p.textContent = message;
        terminalLogs.appendChild(p);
        
        // Auto scroll
        terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }

    function updateTrainStatus(training, paused) {
        isTraining = training;
        isPaused = paused;
        
        // Desbloqueia botões corretos
        startTrainBtn.disabled = isTraining;
        pauseTrainBtn.disabled = !isTraining;
        stopTrainBtn.disabled = !isTraining;
        
        if (isTraining) {
            wsStatusBadge.className = "status-badge training";
            trainSpinner.style.display = "block";
            
            if (isPaused) {
                trainStateTitle.textContent = "Treinamento Pausado";
                trainStateDesc.textContent = "Otimização congelada temporariamente pelo usuário.";
                pauseTrainBtn.innerHTML = `<i class="fa-solid fa-play"></i> Retomar`;
                trainSpinner.querySelector("i").className = "fa-solid fa-circle-pause";
            } else {
                trainStateTitle.textContent = "Treinando LLM...";
                trainStateDesc.textContent = "Computando forward e backward passes no seu hardware.";
                pauseTrainBtn.innerHTML = `<i class="fa-solid fa-pause"></i> Pausar`;
                trainSpinner.querySelector("i").className = "fa-solid fa-spinner fa-spin";
            }
        } else {
            wsStatusBadge.className = "status-badge online";
            trainSpinner.style.display = "none";
            trainStateTitle.textContent = "Treinamento Inativo";
            trainStateDesc.textContent = "Configure a arquitetura e clique em 'Iniciar Treino'.";
            pauseTrainBtn.innerHTML = `<i class="fa-solid fa-pause"></i> Pausar`;
        }
    }

    function updateProgressMetrics(data) {
        valTrainLoss.textContent = data.train_loss.toFixed(4);
        valValLoss.textContent = data.val_loss.toFixed(4);
        valSpeed.innerHTML = `${data.tokens_per_sec} <span class="unit">tok/s</span>`;
        valStep.textContent = `${data.step}/${data.max_iters}`;
        
        liveSampleOutput.textContent = `"${data.sample}"`;
        
        // Adiciona ao gráfico se for novo passo
        if (!stepLabels.includes(data.step)) {
            stepLabels.push(data.step);
            lossHistoryTrain.push(data.train_loss);
            lossHistoryVal.push(data.val_loss);
            if (lossChart) {
                lossChart.update();
            }
        }
    }

    function handleTrainingCompletion(checkpoint) {
        updateTrainStatus(false, false);
        valSpeed.textContent = "Pronto";
        appendTerminalLog(`[System] O treinamento terminou com sucesso! Pesos do modelo salvos em: ${checkpoint}`);
        
        // Atualiza status do chat
        chatModelStatus.textContent = "Modelo Treinado e Carregado!";
        chatModelStatus.style.color = "#10b981";
        
        alert("Treinamento Concluído! Vá para a aba 'Playground' para conversar com seu LLM.");
    }

    // ==========================================================================
    // INTERAÇÃO DO PLAYGROUND (CHAT ENDPOINT)
    // ==========================================================================
    
    // Pressionar enter no chat envia o prompt
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendPrompt();
        }
    });

    chatSendBtn.addEventListener("click", sendPrompt);
    
    clearChatBtn.addEventListener("click", () => {
        chatMessagesBox.innerHTML = `
            <div class="chat-message system">
                <div class="msg-avatar"><i class="fa-solid fa-circle-info"></i></div>
                <div class="msg-content">
                    <p>Chat reinicializado. Digite uma frase para seu LLM continuar a geração!</p>
                </div>
            </div>
        `;
    });

    function sendPrompt() {
        const text = chatInput.value.trim();
        if (!text) return;
        
        // Limpa campo
        chatInput.value = "";
        
        // 1) Adiciona mensagem do usuário à tela
        appendMessage("user", text);
        
        // 2) Adiciona marcador de "Pensando..."
        const thinkingId = appendMessage("assistant", `<i class="fa-solid fa-ellipsis fa-bounce"></i> Gerando tokens autoregressivos...`, true);
        
        // Parâmetros do prompt
        const payload = {
            prompt: text,
            max_tokens: parseInt(genTokens.value),
            temperature: parseFloat(genTemperature.value),
            top_k: parseInt(genTopK.value),
            top_p: parseFloat(genTopP.value)
        };
        
        const t0 = performance.now();
        
        fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) {
                throw new Error("Modelo não carregado ou sem checkpoint salvo. Treine o modelo primeiro!");
            }
            return res.json();
        })
        .then(data => {
            const dt = (performance.now() - t0) / 1000;
            const tokenSec = (payload.max_tokens / dt).toFixed(1);
            
            // Remove marcador de pensando e coloca o texto decodificado
            const thinkingEl = document.getElementById(thinkingId);
            if (thinkingEl) {
                const contentBox = thinkingEl.querySelector(".msg-content");
                contentBox.innerHTML = "";
                
                // Efeito simples de digitação rápida
                let currentText = "";
                const fullResponse = data.response;
                let i = 0;
                
                function typeChar() {
                    if (i < fullResponse.length) {
                        currentText += fullResponse.charAt(i);
                        contentBox.textContent = currentText;
                        i++;
                        chatMessagesBox.scrollTop = chatMessagesBox.scrollHeight;
                        setTimeout(typeChar, 15);
                    } else {
                        // Ao terminar, injeta o meta-log
                        const meta = document.createElement("span");
                        meta.className = "msg-meta";
                        meta.innerHTML = `<i class="fa-solid fa-bolt"></i> Tempo: ${dt.toFixed(2)}s | Velocidade: ${tokenSec} tokens/s | Novos Tokens: ${payload.max_tokens}`;
                        contentBox.appendChild(meta);
                    }
                }
                typeChar();
            }
        })
        .catch(err => {
            const thinkingEl = document.getElementById(thinkingId);
            if (thinkingEl) {
                thinkingEl.querySelector(".msg-content").innerHTML = `<p style="color: var(--color-danger);"><i class="fa-solid fa-triangle-exclamation"></i> ERRO: ${err.message}</p>`;
            }
        });
    }

    function appendMessage(role, text, isThinking = false) {
        const id = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const div = document.createElement("div");
        div.className = `chat-message ${role}`;
        div.id = id;
        
        let avatarIcon = '<i class="fa-solid fa-user"></i>';
        if (role === "assistant") {
            avatarIcon = '<i class="fa-solid fa-robot"></i>';
        }
        
        div.innerHTML = `
            <div class="msg-avatar">${avatarIcon}</div>
            <div class="msg-content">
                <p>${text}</p>
            </div>
        `;
        
        chatMessagesBox.appendChild(div);
        chatMessagesBox.scrollTop = chatMessagesBox.scrollHeight;
        return id;
    }
});
