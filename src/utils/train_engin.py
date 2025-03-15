import torch
from tqdm import tqdm
import torch.nn.functional as F
from .util import get_eval_metrics

def curriculum_loss(outputs, img_outputs, labels, alpha):
    # 主损失
    main_loss = F.cross_entropy(outputs, labels)
    
    img_outputs = img_outputs.view(img_outputs.size(0), -1)
    img_loss = F.cross_entropy(img_outputs, labels)
    
    return alpha * main_loss + (1 - alpha) * img_loss

def train_one_epoch(model, train_loader, scheduler, optimizer, device, epoch, num_epochs):
    """执行单个epoch的训练"""
    model.train()
    running_loss = 0.0
    all_preds_train = []
    all_targets_train = []

    print(f'Epoch {epoch+1}/{num_epochs}')
    train_bar = tqdm(train_loader, desc='Training', leave=False)

    # 第一阶段
    if epoch < 50:
        for param in model.text_model.parameters():
            param.requires_grad = False
    # 第二阶段
    elif epoch < 100:
        for param in model.text_model.parameters():
            param.requires_grad = True
    # 第三阶段
    else:
        for param in model.text_model.parameters():
            param.requires_grad = False
        for param in model.image_model.parameters():
            param.requires_grad = False

    alpha = scheduler.get_text_weight()

    for input_ids, attention_mask, images, labels in train_bar:
        # 数据传输到设备
        input_ids, attention_mask, images, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            images.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)

        # 使用课程损失
        img_outputs = model.image_model(images)
        loss = curriculum_loss(outputs, img_outputs, labels, alpha)
        loss.backward()
        optimizer.step()

        # 统计误差和预测值
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds_train.extend(predicted.cpu().numpy())
        all_targets_train.extend(labels.cpu().numpy())

    train_metrics = get_eval_metrics(all_targets_train, all_preds_train)
    return running_loss, train_metrics