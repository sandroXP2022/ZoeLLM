# utils/preprocess.py

import os
import json
import re
from pathlib import Path
from tqdm import tqdm

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"
MERGED_FILE = DATASETS_DIR / "unprocessed_data.jsonl"
FINAL_FILE = DATASETS_DIR / "dataset.jsonl"

MERGED_FILE.parent.mkdir(parents=True, exist_ok=True)

CODE_PATTERNS = [
    r"```[\s\S]*?```",             # Bloco inteiro de código markdown
    r"\bdef\s+\w+\s*\(",           # Funções Python (mais preciso)
    r"\bclass\s+\w+\s*[:{]",       # Classes Python/C++
    r"\bimport\s+[\w.]+",          # Importações Python mais amplas
    r"#include\s+<[^>]+>",         # Include C/C++
    r"[a-zA-Z_]\w*\s*=\s*[^;]+;",  # Atribuição em linguagens como JS/Java
    r"\bfunction\s+\w*\s*\(",      # Funções em JS
    r"^\s{4,}",                    # Indentação típica de código
    r"</?[a-zA-Z0-9]+[^>]*?>",     # Tags HTML com atributos opcionais
    r"\bvar\s+\w+\b",              # Declaração de variável em JS
    r"\bconsole\.log\(",           # Log JS
    r"\bSystem\.out\.print"        # Java
]

CONV_PATTERNS = [
    r"^(User|Assistant|Usuário|Assistente):",   # Identificadores de fala
    r"\b(pergunta|resposta|question|answer)\s*:",  # Rótulos de QA
    r"(?i)^\s*(q|a)\s*[:\-]",                  # Q: ou A: com variações
    r"\?\s*(?:$|\n)",                          # Pergunta no final da frase
    r"(você|tu|meu|sua|nosso|teu)\b",          # Pronomes típicos de conversa
    r"\b(o que|como|quando|por que|quem|onde)\b",  # Perguntas
    r"^[-*]?\s*(Sim|Não|Talvez)\b",            # Respostas diretas
    r"\b(chatgpt|gpt|modelo|responda)\b"       # Menção à IA
]

INSTR_PATTERNS = [
    r"^\d+\.\s",                                # Lista numerada
    r"^[-*+]\s",                                # Marcadores de lista
    r"\b(passos?|etapas?|instruções?)\b",       # Variações
    r"\b(tutorial|guia|manual|how to)\b",       # Termos comuns
    r"\b(abra|clique|instale|copie|crie)\b",    # Verbos imperativos
    r"\binstal(?:e|ação|ar|ado)\b",             # Instalar variantes
    r"\b(run|execute|compile|launch|start)\b",  # Execução
    r"\bterminal|linha de comando\b",           # Contexto CLI
    r"\bdownload\b",                            # Download direto
    r"\bcódigo\s+abaixo\b",                     # Referência a código
    r"\bsiga\s+os\s+passos\b",                  # Frases guia
    r"\bconfigure\b",                           # Configuração
]



# Função para classificar texto
def classificar_texto(text: str) -> str:
    t = text.strip()
    if not t:
        return None
    # Densidade de símbolos de código
    sym_count = len(re.findall(r"[{}<>;=()\[\]]", t))
    density = sym_count / max(len(t), 1)
    if density > 0.05:
        return "codigo"

    # Verificação por padrões de código
    for pat in CODE_PATTERNS:
        if re.search(pat, t, flags=re.MULTILINE):
            return "codigo"

    # Texto conversacional
    for pat in CONV_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE):
            return "conversacional"

    # Texto instrucional
    for pat in INSTR_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return "instrucional"

    # Texto informativo
    return "informativo"

# Fusão de Datasets
def juntar_datasets():
    files = [f for f in DATASETS_DIR.glob("*.jsonl") if f.name not in {MERGED_FILE.name, FINAL_FILE.name}]
    with MERGED_FILE.open("w", encoding="utf-8") as out:
        for file in tqdm(files, desc="🔄 Merge files", unit="file"):
            for line in file.open("r", encoding="utf-8"):
                out.write(line)

# Classificação e Entrega de Dataset Final
def processar_e_classificar():
    # Contar linhas para barra de progresso
    total = sum(1 for _ in MERGED_FILE.open("r", encoding="utf-8"))
    with MERGED_FILE.open("r", encoding="utf-8") as inp, FINAL_FILE.open("w", encoding="utf-8") as out:
        for line in tqdm(inp, total=total, desc="🔍 Classifying", unit="line"):
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                cat = classificar_texto(text)
                if cat:
                    out.write(json.dumps({"text": text, "category": cat}, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    print("Iniciando pré-processamento...")
    print("------------------------------")
    print("------------------------------")
    juntar_datasets()
    print("FUSÃO DE DATASETS COMPLETA")
    print(f"Dataset Completo em: {MERGED_FILE}")
    print("----------------------------------")
    processar_e_classificar()
    print("DATASET PROCESSADO")
    print(f"Dataset Final Disponível em: {FINAL_FILE}")

