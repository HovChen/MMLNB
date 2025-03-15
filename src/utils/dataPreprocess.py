from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_model = AutoModel.from_pretrained("bert-base-uncased")

# 文本预处理
def preprocess_text(text):
    try:
        encoding = tokenizer(text, padding='max_length', max_length=128,
                            truncation=True, return_tensors="pt")
        return encoding['input_ids'], encoding['attention_mask']
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return None, None

# 图像预处理
def preprocess_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 标签映射
def get_label_from_class(class_value):
    if class_value == 'PD':
        return 0
    elif class_value == 'D':
        return 1
    elif class_value == 'U':
        return 2
    else:
        return None

# 数据加载
def csvLoad(csv_path):
    data = pd.read_csv(csv_path)
    if 'class' not in data.columns:
        raise ValueError("The `class` column is missing from the CSV file, check the file structure.")
    data['label'] = data['class'].apply(get_label_from_class)

    # 过滤无效值
    if 'description' not in data.columns:
        raise ValueError("The `description` column is missing from the CSV file, check the file structure.")
    data = data.dropna(subset=['description'])
    encoded_data = data['description'].map(preprocess_text)
    valid_encoded_data = [item for item in encoded_data if item is not None and item[0] is not None and item[1] is not None]
    if not valid_encoded_data:
        raise ValueError("Text preprocessing failed for all descriptions, please check the input data.")

    data = data.iloc[:len(valid_encoded_data)]
    data['input_ids'], data['attention_mask'] = zip(*valid_encoded_data)
    return data
