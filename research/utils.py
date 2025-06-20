import re
from yacs.config import CfgNode
import yaml

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
    return CfgNode(init_dict=yaml.load(open(yaml_file, "r"), Loader=yaml.FullLoader))
