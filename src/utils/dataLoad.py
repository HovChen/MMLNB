import os
from PIL import Image
from torch.utils.data import Dataset

# 自定义数据集
class TextAndImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, tokenizer, transform=None):
        self.data = dataframe
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 获取文本数据
        input_ids = row['input_ids'].squeeze()
        attention_mask = row['attention_mask'].squeeze()
        label = row['label']
        
        # 获取图像
        image_path = os.path.join(self.image_dir, row['file_name'])
        image = Image.open(image_path).convert("RGB")  # 确保图像是RGB格式
        if self.transform:
            image = self.transform(image)

        return input_ids, attention_mask, image, label