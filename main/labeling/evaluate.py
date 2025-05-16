import json
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(pred_path, truth_path):
    predictions = load_json(pred_path)
    ground_truth = load_json(truth_path)

    # Đảm bảo caption trùng khớp giữa 2 file
    caption_to_label_truth = {item["caption"]: item["label"] for item in ground_truth}
    y_true = []
    y_pred = []

    mismatched = 0
    for pred in predictions:
        caption = pred["caption"]
        if caption not in caption_to_label_truth:
            print(f"[!] Caption không khớp: {caption}")
            mismatched += 1
            continue

        y_true.append(caption_to_label_truth[caption])
        y_pred.append(pred["label"])

    print(f"[INFO] Tổng mẫu khớp: {len(y_true)}")
    print(f"[INFO] Mẫu không khớp: {mismatched}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred, labels=["sarcasm", "non-sarcasm"]))

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

# Đường dẫn file
pred_file = "output/text_labeled.json"
truth_file = "ground_truth.json"

#evaluate(pred_file, truth_file)