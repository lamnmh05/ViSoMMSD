import json

def load_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_json(input_path, output_path, indent=4):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(input_path, f, ensure_ascii=False, indent=indent)
