from tqdm import tqdm
import torch
from .util import get_eval_metrics
import numpy as np

def evaluate_on_training(model, val_loader, criterion, device, desc='Validating'):
    model.eval()

    all_preds_val = []
    all_probs_val = []
    all_targets_val = []
    running_loss_val = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, images, labels in tqdm(val_loader, desc):
            input_ids, attention_mask, images, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                images.to(device),
                labels.to(device),
            )

            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_probs_val.extend(probs)
            _, predicted = torch.max(outputs, 1)
            all_preds_val.extend(predicted.cpu().numpy())
            all_targets_val.extend(labels.cpu().numpy())
    
    val_metrics = get_eval_metrics(all_targets_val, all_preds_val, probs_all=np.array(all_probs_val))
    return running_loss_val, val_metrics, all_preds_val, all_targets_val