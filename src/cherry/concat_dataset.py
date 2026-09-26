import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Concatenate cherry-picked dataset files")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing cherry-picked files")
    parser.add_argument("--save-path", type=str, required=True, help="Output path for concatenated JSON file")
    parser.add_argument("--file-suffix", type=str, default="_cherry.json", help="File suffix to match (default: _cherry.json)")
    args = parser.parse_args()

    pre_experience_data = []
    for file in os.listdir(args.root_dir):
        if file.endswith(args.file_suffix):
            with open(os.path.join(args.root_dir, file), "r", encoding="utf-8") as f:
                data = json.load(f)

            pre_experience_data.extend(data)

    print(f"Total samples: {len(pre_experience_data)}")

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(pre_experience_data, f, indent=4, ensure_ascii=False)

    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()