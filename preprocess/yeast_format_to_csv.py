import pandas as pd
import os
from tqdm import tqdm

DICT_PATH = "./data/seq_ppi-master/yeast/preprocessed/protein.dictionary.tsv"

FILES_TO_PROCESS = [
    "./yeast/dataset/train.tsv",
    "./yeast/dataset/test.tsv",

]

OUTPUT_DIR = "./process_yeast/"

def load_sequence_dictionary(dict_path):

    print(f"🔑 正在加载序列字典: {dict_path} ...")
    seq_dict = {}
    with open(dict_path, 'r') as f:
        for line in tqdm(f, desc="读取字典"):
            parts = line.strip().split('\t')
            if len(parts) >= 2:

                seq_dict[parts[0]] = parts[1]
    print(f"✅ 成功加载了 {len(seq_dict)} 个蛋白质序列！\n")
    return seq_dict

def process_and_format(file_list, seq_dict, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件，已跳过: {file_path}")
            continue

        print(f"🔄 正在处理并转换: {file_path}")

        df = pd.read_csv(file_path, sep='\t', header=None, names=['protein_a_id', 'protein_b_id', 'label'])
        original_len = len(df)

        df['seq_a'] = df['protein_a_id'].map(seq_dict)
        df['seq_b'] = df['protein_b_id'].map(seq_dict)

        missing_count = df['seq_a'].isna().sum() + df['seq_b'].isna().sum()
        df = df.dropna(subset=['seq_a', 'seq_b'])
        final_len = len(df)

        df = df[['protein_a_id', 'protein_b_id', 'seq_a', 'seq_b', 'label']]

        base_name = os.path.basename(file_path)
        csv_name = base_name.replace('.tsv', '.csv')
        out_name = os.path.join(out_dir, csv_name)

        df.to_csv(out_name, sep=',', index=False)

        print(f"   -> 原始数据: {original_len} 条")
        if missing_count > 0:
            print(f"   -> ⚠️ 字典缺失，丢弃了 {original_len - final_len} 条不完整数据。")
        print(f"   -> ✅ 转换成功！已保存为标准格式: {out_name}\n")

if __name__ == "__main__":
    if not os.path.exists(DICT_PATH):
        print(f"❌ 致命错误：找不到字典文件 {DICT_PATH} ！！")
    else:

        global_seq_dict = load_sequence_dictionary(DICT_PATH)

        process_and_format(FILES_TO_PROCESS, global_seq_dict, OUTPUT_DIR)

        print("🎉 全部处理完成！去 ./formatted_csv_datasets/ 目录下拿你的完美 CSV 喂给大模型吧！")
