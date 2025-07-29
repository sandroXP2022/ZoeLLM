# utils/download_datasets.py

import os
import json
import re
import requests
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def salvar_jsonl(nome, textos):
    caminho = os.path.join(DATASETS_DIR, nome)
    with open(caminho, "w", encoding="utf-8") as f:
        for t in textos:
            t = t.strip().replace("\n", " ")
            t = re.sub(r"\s+", " ", t)
            if len(t) >= 20:
                f.write(json.dumps({"text": t}) + "\n")

def existe(nome):
    return os.path.exists(os.path.join(DATASETS_DIR, nome))

def baixar_wikipedia(lang="en", limite=5000):
    nome = f"wikipedia_{lang}.jsonl"
    if existe(nome):
        print(f"🔁 {nome} já existe — ignorado")
        return
    configs = get_dataset_config_names("wikimedia/wikipedia")
    lang_configs = [c for c in configs if c.endswith(f".{lang}")]
    latest = sorted(lang_configs)[-1]
    ds = load_dataset("wikimedia/wikipedia", latest, split="train")
    textos = [ex["text"] for ex in ds.select(range(min(limite, len(ds)))) if ex["text"].strip()]
    salvar_jsonl(nome, textos)


def baixar_code_search_net(limite=3000):
    nome = "code_search_net_python.jsonl"
    if existe(nome):
        print(f"🔁 {nome} já existe — ignorado")
        return
    ds = load_dataset("code_search_net", "python", split="train")
    textos = []
    for ex in tqdm(ds, desc="💻 CodeSearchNet Python"):
        code = ex.get("code")
        if code and ("def " in code or "class " in code):
            textos.append(code)
        if len(textos) >= limite:
            break
    salvar_jsonl(nome, textos)

def baixar_project_euler(limite=50):
    nome = "project_euler.jsonl"
    if existe(nome):
        print(f"🔁 {nome} já existe — ignorado")
        return
    url = "https://projecteuler.net/archives"
    resp = requests.get(url)
    from lxml import html
    tree = html.fromstring(resp.content)
    links = tree.xpath('//a[contains(@href, "problem=")]/@href')
    textos = []
    base = "https://projecteuler.net/"
    for link in tqdm(links[:limite], desc="🧮 Project Euler"):
        page = requests.get(base + link)
        doc = html.fromstring(page.content)
        txt = " ".join(doc.xpath('//div[@class=\"problem_content\"]//text()')).strip()
        textos.append(txt)
    salvar_jsonl(nome, textos)

def baixar_gsm8k(limite=3000):
    nome = "gsm8k.jsonl"
    if existe(nome):
        print(f"🔁 {nome} já existe — ignorado")
        return
    ds = load_dataset("gsm8k", "main", split="train")
    textos = []
    for ex in tqdm(ds, desc="📐 GSM8K"):
        textos.append(f"{ex['question']}\n{ex['answer']}")
        if len(textos) >= limite:
            break
    salvar_jsonl(nome, textos)

def barra_geral():
    tarefas = [
        ("Wikipedia EN", lambda: baixar_wikipedia("en")),
        ("Wikipedia PT", lambda: baixar_wikipedia("pt")),
        ("CodeSearchNet (Python)", baixar_code_search_net),
        ("Project Euler", baixar_project_euler),
        ("GSM8K", baixar_gsm8k),
    ]
    print(f"\n📦 Iniciando download de dataset {len(tarefas)}\n")
    for i, (nome, func) in enumerate(tarefas, 1):
        print(f"[{i}/{len(tarefas)}] ⏳ {nome}")
        try:
            func()
            print(f"[{i}/{len(tarefas)}] ✅ {nome} concluído\n")
        except Exception as e:
            print(f"[{i}/{len(tarefas)}] ❌ Erro em {nome}: {e}\n")
    print(f"🎉 Todos os datasets (novos ou já existentes) estão em `{DATASETS_DIR}`")

if __name__ == "__main__":
    barra_geral()

