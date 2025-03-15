from transformers import AutoModel
import torch
import torch.nn as nn
from torchvision import models

# 模型的配置和初始化
class NoiseRobustFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 可信度评估网络
        self.confidence_net = nn.Sequential(
            nn.Linear(768 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出文本可信度[0,1]
        )
        
        self.res_fusion = nn.Linear(512, 768)

    def forward(self, text_feat, img_feat):
        # 计算文本可信度
        combined = torch.cat([text_feat, img_feat], dim=1)
        text_conf = self.confidence_net(combined)  # [B,1]

        img_proj = self.res_fusion(img_feat)  # [B,768]
        
        # 动态融合
        fused = text_conf * text_feat + (1 - text_conf) * img_proj
        return fused

# 定义融合模型
class MMLNB(nn.Module):
    def __init__(self, num_classes):
        super(MMLNB, self).__init__()
        
        self.num_classes = num_classes

        # 文本部分：BERT
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.text_feature_dim = self.text_model.config.hidden_size  # BERT 输出特征维度 (768)

        # 图像部分：VGG16
        vgg16 = models.vgg16(pretrained=False)
        self.image_model = torch.nn.Sequential(
            *list(vgg16.features),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.image_feature_dim = 512

        # 融合模块
        self.fusion_module = NoiseRobustFusion()

        # 分类器
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes),
        )

    def forward(self, input_ids, attention_mask, image_tensor):
        # 文本特征
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        # 图像特征
        image_features = self.image_model(image_tensor)
        image_features = image_features.view(image_features.size(0), -1)

        # 特征融合
        fused_features = self.fusion_module(text_features, image_features)

        # 分类器
        output = self.classifier(fused_features)
        return output