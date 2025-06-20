import json
import re
from yacs.config import CfgNode as CN
import yaml


def load_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_to_json(input_path, output_path, indent=4):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(input_path, f, ensure_ascii=False, indent=indent)


def remove_emojis(text):
    # Unicode ranges cho emoji (cơ bản)
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)


def remove_links(text):
    # Loại bỏ http, https, www link
    return re.sub(r'http\S+|www\.\S+', '', text)


def preprocess_text(text):
    text = remove_emojis(text)
    text = remove_hashtags(text)
    text = remove_links(text)
    # Xóa khoảng trắng thừa
    text = ' '.join(text.split())
    return text


def get_config(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as f:
        return CN(init_dict=yaml.load(f, Loader=yaml.FullLoader))


def remove_json_markdown(response: str) -> dict:
    """
    Extracts JSON object from a markdown-style code block in LLM output.
    """
    match = re.search(r"```json\s*([\s\S]+?)\s*```", response)
    match2 = re.search(r"```\s*([\s\S]+?)\s*```", response)
    if match:
        json_str = match.group(1)
    elif match2:
        json_str = match2.group(1)
    else:
        json_str = response

    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None