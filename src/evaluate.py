import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
import os
import sys

from utils.util import print_metrics,display_confusion_matrix,check_pth_extension
from models import MMLNB
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import dataLoad, dataPreprocess
from utils.evaluation import evaluate_on_training

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=2024,help='set seed to make it reproducible')
    parser.add_argument('--csv_path',type=str,required=True,help='dataset: csv file path')
    parser.add_argument('--image_dir',type=str,required=True,help='dataset: image directory')
    parser.add_argument('--checkpoint_path',type=check_pth_extension,required=True,help='e.g, model.pth')
    parser.add_argument('--save_path',type=str,required=True,help='path to save confusion matrix')
    return parser.parse_args()

# 设置种子以确保实验可重现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 评估函数
def evaluate_model(model, val_loader, criterion, device, checkpoint_path, save_path):
    # 加载最佳模型权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    _, eval_metrics, all_targets, all_preds = evaluate_on_training(model,val_loader,criterion,device,desc='Evaluating')
    print("Evaluation Metrics:")
    print_metrics(eval_metrics)

    # 生成混淆矩阵
    class_names = ["PD", "D", "UD"]  # 类别名称列表
    cm = confusion_matrix(all_targets, all_preds)
    display_confusion_matrix(cm, class_names,save_path)

# 评估模型
def main():
    p = parse_arguments()
    set_seed(p.seed)
    data = dataPreprocess.csvLoad(p.csv_path)
    # 划分训练集和验证集
    dataset = dataLoad.TextAndImageDataset(data, p.image_dir, dataPreprocess.tokenizer, transform=dataPreprocess.preprocess_image())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = MMLNB.MMLNB(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(model, val_loader, criterion, device, p.checkpoint_path,p.save_path)

if __name__ == "__main__":
    main()
