import json

def validate_dataset(path):
    total_lines = 0
    valid_lines = 0
    invalid_lines = 0
    max_token = None
    min_token = None
    invalid_tokens_found = False

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, dict) and 'input_ids' in obj:
                    tokens = obj['input_ids']
                    if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                        valid_lines += 1
                        # Atualiza máximo e mínimo token
                        line_max = max(tokens) if tokens else None
                        line_min = min(tokens) if tokens else None
                        if line_max is not None:
                            if max_token is None or line_max > max_token:
                                max_token = line_max
                        if line_min is not None:
                            if min_token is None or line_min < min_token:
                                min_token = line_min
                    else:
                        print(f"Tokens não são todos inteiros na linha {line_num}")
                        invalid_lines += 1
                        invalid_tokens_found = True
                else:
                    print(f"Formato inesperado ou 'input_ids' ausente na linha {line_num}")
                    invalid_lines += 1
            except json.JSONDecodeError:
                print(f"Erro a ler linha {line_num}: JSON inválido")
                invalid_lines += 1

    print(f"Total linhas: {total_lines}")
    print(f"Linhas válidas: {valid_lines}")
    print(f"Linhas inválidas: {invalid_lines}")
    print(f"Token máximo encontrado: {max_token}")
    print(f"Token mínimo encontrado: {min_token}")
    print(f"Tokens inválidos encontrados? {'Sim' if invalid_tokens_found else 'Não'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python validate_dataset.py caminho_para_dataset.jsonl")
        sys.exit(1)
    validate_dataset(sys.argv[1])

