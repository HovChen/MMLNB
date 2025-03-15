import argparse
from utils.util import convert_csv_to_json,valid_path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv',type=str,required=True,help='input csv file path')
    parser.add_argument('--output_json', type=str, required=True, help='output json file path')
    parser.add_argument('--root',type=valid_path,required=True,help='root path of images in csv file')
    parser.add_argument('--prompt_path', type=str, default='configs/prompt.txt', help='prompt text file path')

    return parser.parse_args()

if __name__ == "__main__":
    p = parse_arguments()
    csv_path = p.input_csv
    output_path = p.output_json
    convert_csv_to_json(csv_path, output_path,p.root, p.prompt_path)
    print(f"Converted! Data saved to {output_path}") 