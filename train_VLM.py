from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
from peft import get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import swanlab
import json
import os
import argparse

from src.utils import vlm,util

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, help='finetune dataset: json file path')
    parser.add_argument('--output_dir', type=str, required=True, help='finetuned model output')
    parser.add_argument('--project_name', type=str, required=True, help='e.g, Qwen2.5-VL-finetune')
    parser.add_argument('--experiment_name', type=str, required=True, help='e.g, qwen2.5-vl-pathology')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name(e.g, Neuroblastoma Pathology Gallery 1500)')
    parser.add_argument('--prompt_path', type=str, default='src/configs/prompt.txt', help='prompt text file path')
    parser.add_argument('--model', type=str, default='https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct', help='model url')
    return parser.parse_args()

def train(p,model,processor,tokenizer,config,prompt):
    # ========== 主流程 ==========

    model.config.use_cache = False  # 与 gradient_checkpointing 冲突，需禁用 use_cache
    model.enable_input_require_grads()

    train_json_path = p.json_path
    with open(train_json_path, 'r') as f:
        data = json.load(f)
        train_data = data[:-4]
        test_data = data[-4:]

    with open("data_vl_train.json", "w") as f:
        json.dump(train_data, f)
    with open("data_vl_test.json", "w") as f:
        json.dump(test_data, f)

    train_ds = Dataset.from_json("data_vl_train.json")
    train_dataset = train_ds.map(vlm.process_func,fn_kwargs={"processor": processor, "tokenizer": tokenizer})

    peft_model = get_peft_model(model, config)

    # 配置训练参数
    args = TrainingArguments(
        output_dir=p.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=93,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        report_to="none",
    )

    # SwanLab 回调（可选）
    swanlab_callback = SwanLabCallback(
        project=p.project_name,
        experiment_name=p.experiment_name,
        config={
            "model": p.model,
            "dataset": p.dataset,
            "prompt": prompt,
            "train_data_number": len(train_data),
            "lora_rank": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
        },
    )

    # 构建 Trainer 并训练
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )
    trainer.train()


def test(p,model,processor,config,prompt):
    # ========== 测试阶段 ==========
    # 加载微调后的权重
    output_dir = p.output_dir
    peft_model_path = f"{output_dir}/checkpoint-{max([int(d.split('-')[-1]) for d in os.listdir(output_dir) if d.startswith('checkpoint-')])}"
    print(f"peft_model_path: {peft_model_path}")
    val_peft_model = PeftModel.from_pretrained(model, peft_model_path, config=config)

    # 加载测试集
    with open("data_vl_test.json", "r") as f:
        test_dataset = json.load(f)

    test_image_list = []
    for item in test_dataset:
        input_image_prompt = item["conversations"][0]["value"]
        origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": origin_image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        response = vlm.predict(messages, val_peft_model,processor)
        messages.append({"role": "assistant", "content": f"{response}"})
        print(messages[-1])

        test_image_list.append(
            swanlab.Image(origin_image_path, caption=response)
        )

    swanlab.log({"Prediction": test_image_list})
    swanlab.finish()

def main():
    p = parse_arguments()
    model, processor, tokenizer = vlm.load_model_processor_tokenizer()
    config = vlm.config
    prompt = util.load_prompt(p.prompt_path)
    train(p,model,processor,tokenizer,config,prompt)
    test(p,model,processor,config,prompt)

if __name__ == "__main__":
    main()
