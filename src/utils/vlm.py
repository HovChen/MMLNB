import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import pandas as pd
from qwen_vl_utils import process_vision_info

config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

def load_model_processor_tokenizer(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
    )
    return model, processor, tokenizer

def build_messages(image_path, prompt_text):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text}
        ]
    }]

def save_results(results, output_path):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))

def handle_error(image_path, error):
    print(f"Error processing {image_path}: {str(error)}")

def process_func(example, processor, tokenizer):
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径

    # 构造 messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 256, "resized_width": 256},
            ],
        }
    ]
    
    # 1. 生成 prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 处理图像信息
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.tolist() for key, value in inputs.items()}

    # 3. 生成 labels
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 处理超长截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs['image_grid_thw']).squeeze(0)
    }

def predict(messages, model, processor):
    """ 用于推理验证的函数 """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # 取生成的后半部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]