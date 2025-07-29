import json
from tqdm import tqdm
from unidecode import unidecode
import sentencepiece as spm
from pathlib import Path

sp = spm.SentencePieceProcessor(model_file="model/unigram.model")

input_path = Path("datasets/dataset.jsonl")
output_path = Path("datasets/dataset_tokenized.jsonl")

max_length = 256
pad_token_id = sp.pad_id() if sp.pad_id() >= 0 else 0

num_valid = 0
num_skipped = 0

with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    lines = fin.readlines()
    for line in tqdm(lines, desc="Tokenizando..."):
        text = line.strip()
        if not text:
            num_skipped += 1
            continue

        try:
            text = unidecode(text)
            input_ids = sp.encode(text, out_type=int)
            input_ids = input_ids[:max_length]

            attention_mask = [1] * len(input_ids)

            if len(input_ids) < max_length:
                pad_len = max_length - len(input_ids)
                input_ids += [pad_token_id] * pad_len
                attention_mask += [0] * pad_len

            json.dump({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }, fout)
            fout.write("\n")
            num_valid += 1
        except Exception:
            num_skipped += 1


def encode(text):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)



print("\n-------------------------------------------")
print(f"Feito! \n{num_valid} linhas processadas com sucesso, {num_skipped} ignoradas.")
print("-------------------------------------------")

