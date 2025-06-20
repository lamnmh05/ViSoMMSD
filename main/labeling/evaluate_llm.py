import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)

def evaluate_llm_vs_human(json_file, pos_label="sarcasm"):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    gold = []
    pred = []

    # Đảm bảo label thống nhất chữ hoa/thường
    for item in data:
        gold.append(item["human_label"].strip().lower())
        pred.append(item["label"].strip().lower())

    # Đánh giá
    accuracy = accuracy_score(gold, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average='binary', pos_label='sarcasm')


    print("== Evaluation metrics between LLM and human label ==")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    print("\nClassification report:\n", classification_report(
        gold, pred, digits=4, target_names=[pos_label, f"not_{pos_label}"]))

if __name__ == "__main__":
    file_path = r'D:\Git_repo\ViSoMMSD\research\gemma.json'
    evaluate_llm_vs_human(file_path, pos_label="sarcasm")
