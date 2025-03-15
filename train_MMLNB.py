from src.utils import evaluation
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
import argparse

from src.models import MMLNB
from src.utils import dataLoad, dataPreprocess
from src.utils import train_engin,evaluation
from src.utils.util import print_metrics,save_metrics,check_pth_extension

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=2024,help='set seed to make it reproducible')
    parser.add_argument('--csv_path',type=str,required=True,help='dataset: csv file path')
    parser.add_argument('--image_dir',type=str,required=True,help='dataset: image directory')
    parser.add_argument('--metrics_path',type=str,default='./metrcs.json',help='path to save best metrics')
    parser.add_argument('--save_checkpoint',type=check_pth_extension,required=True,help='path to save pth file')
    return parser.parse_args()

# 设置种子以确保实验可重现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CurriculumScheduler:
    def __init__(self, total_epoch=150):
        self.epoch = 0
        self.total_epoch = total_epoch
        
    def get_text_weight(self):
        # 前期图像特征主导，后期文本特征逐渐引入
        if self.epoch < 50:
            return 0.3
        elif self.epoch < 100:
            return 0.3 + ((self.epoch - 50) // 10) * (1 - 0.3) / 5
        else:
            return 1.0

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, save_checkpoint, num_epochs=1):
    best_f1 = 0.0
    best_metrics = {}

    scheduler = CurriculumScheduler()

    # 分阶段训练
    for epoch in range(num_epochs):
        running_loss, train_metrics = train_engin.train_one_epoch(model,train_loader,scheduler,optimizer,device,epoch,num_epochs)
        running_loss_val, val_metrics, _, _ = evaluation.evaluate_on_training(model,val_loader,criterion,device)

        print(f'\nEpoch {epoch+1} - Train Loss: {running_loss:.4f}, Val Loss: {running_loss_val:.4f}')
        print("Train Metrics:")
        print_metrics(train_metrics)
        print("Validation Metrics:")
        print_metrics(val_metrics)

        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            best_metrics = val_metrics
            best_metrics_epoch = epoch + 1
            torch.save(model.state_dict(), save_checkpoint)
            print("New best model saved!")

        print(f"\nBest Model Validation Metrics:")
        print_metrics(best_metrics)

    return best_metrics,best_metrics_epoch

def main():
    p = parse_arguments()
    set_seed(p.seed)
    data = dataPreprocess.csvLoad(p.csv_path)

    # 划分训练集和验证集
    dataset = dataLoad.TextAndImageDataset(data, p.image_dir, dataPreprocess.tokenizer, transform=dataPreprocess.preprocess_image())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"训练集数量：{train_size}\n验证集数量：{val_size}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 模型训练和验证
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MMLNB.MMLNB(num_classes=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 开始训练和验证
    best_metrics,best_metrics_epoch = train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, p.save_checkpoint)
    save_metrics(best_metrics,best_metrics_epoch,p.metrics_path)


if __name__ == "__main__":
    main()
