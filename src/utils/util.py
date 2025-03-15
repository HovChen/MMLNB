import argparse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, classification_report, roc_auc_score
from typing import Optional, Dict, Any, Union, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import csv
import json
import re

def check_pth_extension(value):
    if not value.endswith('.pth'):
        raise argparse.ArgumentTypeError("'--save_checkpoint' has to end with .pth")
    return value

def valid_path(value):
    if re.match(r'^.*[/\\]$', value):
        raise argparse.ArgumentTypeError("Paths cannot end with a slash or backslash")
    if not os.path.isabs(value):
        raise argparse.ArgumentTypeError("Absolute path required")
    return value

def load_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# 计算评估指标
def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
        f"{prefix}precision": cls_rep["weighted avg"]["precision"],
        f"{prefix}recall": cls_rep["weighted avg"]["recall"],
    }

    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    if probs_all is not None:
        try:
            roc_auc = roc_auc_score(targets_all, probs_all, multi_class="ovr", **roc_kwargs)
            eval_metrics[f"{prefix}auroc"] = roc_auc
        except ValueError:
            eval_metrics[f"{prefix}auroc"] = None

    return eval_metrics

# 打印评估指标
def print_metrics(eval_metrics: Dict[str, Any]) -> None:
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")

# 显示并保存混淆矩阵
def display_confusion_matrix(cm, class_names, save_path):
    os.makedirs(save_path, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")

    # 保存文件到指定路径
    save_file = os.path.join(save_path, f"confusion_matrix.png")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_file}")

def save_metrics(best_metrics,best_metrics_epoch,metrics_path):
    # 在训练结束后保存最佳指标
    best_metrics_save_path = metrics_path
    os.makedirs(os.path.dirname(best_metrics_save_path), exist_ok=True)  # 确保目录存在
    
    # 转换numpy类型到Python原生类型
    best_metrics_serializable = {}
    for key, value in best_metrics.items():
        if isinstance(value, np.generic):
            best_metrics_serializable[key] = value.item()
        else:
            best_metrics_serializable[key] = value

    best_metrics_info = {
        'best_epoch': best_metrics_epoch,
        'best_metrics': best_metrics_serializable
    }

    with open(best_metrics_save_path, 'w') as f:
        json.dump(best_metrics_info, f, indent=4)
    
    print(f"\nBest metrics saved to {best_metrics_save_path}")

def convert_csv_to_json(csv_path, output_path,root,prompt_path):
    data = []
    identity_counter = 1
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        prompt = load_prompt(prompt_path)
        for row in reader:
            entry = {
                "id": f"identity_{identity_counter}",
                "conversations": [
                    {
                        "from": "user",
                        "value": prompt+" <|vision_start|>"+root+f"/{row['file_name']}<|vision_end|>"
                    },
                    {
                        "from": "assistant",
                        "value": row['description']
                    }
                ]
            }
            data.append(entry)
            identity_counter += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)