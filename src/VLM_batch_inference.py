import os
from qwen_vl_utils import process_vision_info
import torch
import argparse
from utils import vlm, util
from peft import PeftModel

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, help='image directory path')
    parser.add_argument('--output_path', type=str, required=True, help='output csv file path')
    parser.add_argument('--prompt_path', type=str, default='src/configs/prompt.txt', help='prompt text file path')
    parser.add_argument('--finetuned_model', type=str, required=True, help='finetuned model path. e.g, Qwen2.5-VL-Pathology/checkpoint-465')
    return parser.parse_args()

def generate_description(model, processor, messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

def process_images(args, model, processor, prompt_text):
    results = []
    for root, _, files in os.walk(args.img_dir):
        for file in files:
            if not vlm.is_image_file(file):
                continue
            image_path = os.path.join(root, file)
            try:
                messages = vlm.build_messages(image_path, prompt_text)
                description = generate_description(model, processor, messages)
                results.append({"file_name": file, "description": description})
                vlm.save_results(results, args.output_path)
                print(f"Processed {file}")
            except Exception as e:
                vlm.handle_error(image_path, e)
    return results

def main():
    p = parse_arguments()
    model, processor, _ = vlm.load_model_processor_tokenizer()
    model = PeftModel.from_pretrained(model, model_id=p.finetuned_model, config=vlm.config)
    prompt_text = util.load_prompt(p.prompt_path)
    results = process_images(p, model, processor, prompt_text)
    vlm.save_results(results, p.output_path)
    print(f"Final results saved to {p.output_path}")

if __name__ == "__main__":
    main()
