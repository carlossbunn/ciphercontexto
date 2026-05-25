Pipeline de avaliação de ataques adversariais via cifras linguísticas contra LLMs locais, baseado em Yuan et al. (2024) "GPT-4 Is Too Smart To Be Safe".

Visão geral
O pipeline aplica diferentes cifras a prompts adversariais e mede o quanto cada codificação consegue bypassar os mecanismos de segurança dos modelos-alvo. Um juiz auxiliar (Phi) avalia as respostas e várias métricas são computadas automaticamente.
prompts_base_sem_perturbacao.json
        │
        ▼
  aplica cifra (plain / unicode / ascii / morse / rot13)
        │
        ▼
  envia ao modelo-alvo (Llama / Qwen / Tucano)
        │
        ▼
  classifica resposta (heurística + Phi judge)
        │
        ▼
  exporta CSV / JSON / JSONL com métricas

Cifras testadas
CifraDescriçãoControleplainPrompt original, sem alteração✅ simunicodeCada caractere → \uXXXX—asciiCada caractere → valor decimal—morseCódigo Morse (A-Z, 0-9)—rot13Substituição ROT-13 (apenas letras)—

Modelos-alvo

meta-llama/Llama-3.2-3B-Instruct
Qwen/Qwen2.5-3B-Instruct
TucanoBR/Tucano-2b4

Juiz auxiliar (opcional): microsoft/Phi-4-mini-instruct

Métricas capturadas

ASR — Attack Success Rate
Objective Satisfaction Rate
Refusal Rate
Compliance / Safety Score
Robustness Degradation — ASR(cifra) − ASR(plain) por modelo
Intent Understanding Rate
Judge Agreement / Cohen's Kappa
Transferability Rate — ataque bem-sucedido em 2+ modelos
Unsafe Compliance Rate
Cipher Failure Rate (Noise Failure Rate) — cifra causou incompreensão


Instalação
1. Criar e ativar o ambiente virtual
bashpython3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows
2. Instalar dependências
bashpip install --upgrade pip
pip install -r requirements.txt
3. Autenticação Hugging Face (obrigatório para Llama)
bashhuggingface-cli login
# ou
export HF_TOKEN="hf_seu_token_aqui"

Dataset esperado
Arquivo JSON com a estrutura abaixo (30 prompts base, 6 categorias × 5 prompts):
json{
  "datasets": {
    "prompts_base_sem_perturbacao": {
      "data": [
        {
          "id": "p001",
          "prompt": "...",
          "category": "jailbreak",
          "pass_conditions": ["..."],
          "fail_conditions": ["..."]
        }
      ]
    }
  }
}

Uso
Execução completa
bashpython3 pipeline_cipherchat_tcc.py --dataset prompts_base_sem_perturbacao.json
Teste rápido (5 prompts)
bashpython3 pipeline_cipherchat_tcc.py --dataset prompts_base_sem_perturbacao.json --limit 5
Retomar execução interrompida
bashpython3 pipeline_cipherchat_tcc.py --dataset prompts_base_sem_perturbacao.json --resume

Saídas geradas
Todos os arquivos são salvos em ./resultados_cipherchat_tcc/:
ArquivoConteúdoresults_cipherchat_raw.jsonlRegistro bruto linha a linha (para resume)tcc_cipherchat_all_results.csvTodos os resultados consolidadostcc_cipherchat_by_cipher.csvMétricas agregadas por cifratcc_cipherchat_by_model.csvMétricas agregadas por modelotcc_cipherchat_robustness_degradation.csvDelta ASR cifra vs plaintcc_cipherchat_transferability.csvTransferabilidade por ataquetcc_cipherchat_radar_metrics.csvTabela pronta para gráfico radarsummary_cipherchat.jsonSumário completo em JSONpipeline_cipherchat.logLog de execução<model_slug>_results.csvResultados por modelo individual

Configurações principais
As constantes abaixo ficam no topo do script e podem ser ajustadas sem alterar a lógica:
ConstantePadrãoDescriçãoENABLE_PHI_JUDGETrueAtiva/desativa o juiz PhiPHI_HARM_SEVERITY_THRESHOLD3Severidade mínima para contar como ataqueTRANSFERABILITY_MIN_SUCCESSFUL_MODELS2Mínimo de modelos para considerar transferívelMAX_NEW_TOKENS512Tokens máximos na geraçãoDO_SAMPLEFalseGeração determinísticaGLOBAL_SEED42Semente de reprodutibilidade

Requisitos de hardware

GPU CUDA recomendada (VRAM ≥ 8 GB para carregar dois modelos de 3B simultaneamente)
CPU funciona, mas é significativamente mais lento
Os modelos são carregados e descarregados sequencialmente para economizar memória


Referência
Yuan, Z. et al. (2024). GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher. arXiv:2308.06463.
