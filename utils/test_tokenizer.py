import sentencepiece as spm
from unidecode import unidecode

sp = spm.SentencePieceProcessor(model_file="model/unigram.model")

def interactive_tokenizer():
    print("Enter your text to tokenize. Type 'exit' or 'quit' to end.")
    while True:
        text = input(">> ")
        if text.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        clean = unidecode(text)
        token_strs = sp.encode(clean, out_type=str)
        token_ids  = sp.encode(clean, out_type=int)
        print("Tokens:", token_strs)
        print("Token IDs:", token_ids)
        decoded = sp.decode(token_ids)
        print("Decoded:", decoded)

if __name__ == "__main__":
    interactive_tokenizer()

