import json
import sentencepiece as spm
from unidecode import unidecode
from tqdm import tqdm
from pathlib import Path

DATASET = Path("datasets/dataset.jsonl")
SP_MODEL_PREFIX = Path("model/unigram")
VOCAB_SIZE = 100000
CHARACTER_COVERAGE = 0.996
INPUT_SENTENCE_SIZE = 5_000_000
SHUFFLE = True
USER_DEFINED_SYMBOLS = ["<mask>", "<input>"]

def preprocess_and_overwrite_dataset(path):
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for raw in tqdm(f, desc="A preparar dataset..."):
            try:
                data = json.loads(raw)
                text = data.get("text", "")
            except:
                text = raw
            lines.append(unidecode(text))
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")

def train_unigram(corpus_path, model_prefix, vocab_size):
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=1, desc="A treinar modelo Unigram") as p:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=CHARACTER_COVERAGE,
            input_sentence_size=INPUT_SENTENCE_SIZE,
            shuffle_input_sentence=SHUFFLE,
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=USER_DEFINED_SYMBOLS,
        )
        p.update(1)

if __name__ == "__main__":
    preprocess_and_overwrite_dataset(DATASET)
    train_unigram(DATASET, SP_MODEL_PREFIX, VOCAB_SIZE)
    print(f"\n✅ Modelo treinado: {SP_MODEL_PREFIX}.model\n   Vocabulário:   {SP_MODEL_PREFIX}.vocab")
