import pandas as pd
import requests
import time
from tqdm import tqdm
import os

train_file = './multi_class/dataset/train.tsv'
test_file = './multi_class/dataset/test.tsv'

if not os.path.exists(train_file) or not os.path.exists(test_file):
    print("❌ 请确保 train.tsv 和 test.tsv 在当前目录下！")
    exit()

print("📂 正在读取原始 TSV 数据...")

df_train = pd.read_csv(train_file, sep='\t', header=None, names=['protein_a_id', 'protein_b_id', 'label'])
df_test = pd.read_csv(test_file, sep='\t', header=None, names=['protein_a_id', 'protein_b_id', 'label'])

all_ids = set(df_train['protein_a_id']).union(set(df_train['protein_b_id'])) \
          .union(set(df_test['protein_a_id'])).union(set(df_test['protein_b_id']))

print(f"🎯 提取到 {len(all_ids)} 个唯一的人类蛋白质 ID，准备获取序列...")

server = "https://rest.ensembl.org"
seq_dict = {}
dict_file = 'temp_protein_dict.tsv'

if os.path.exists(dict_file):
    with open(dict_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                seq_dict[parts[0]] = parts[1]
    print(f"♻️ 从本地缓存读取了 {len(seq_dict)} 条已下载的序列。")

ids_to_download = [pid for pid in all_ids if pid not in seq_dict]

if ids_to_download:
    print(f"🚀 开始连接 Ensembl 数据库下载剩余的 {len(ids_to_download)} 条序列...")
    with open(dict_file, 'a') as f:
        for full_id in tqdm(ids_to_download, desc="下载序列"):
            ensp_id = full_id.split('.')[-1]
            ext = f"/sequence/id/{ensp_id}?type=protein"

            try:
                r = requests.get(server + ext, headers={"Content-Type": "text/plain"})
                if r.ok:
                    seq = r.text.strip()
                    seq_dict[full_id] = seq
                    f.write(f"{full_id}\t{seq}\n")
                else:
                    pass
            except Exception as e:
                print(f"\n❌ 网络错误 {full_id}: {e}")

            time.sleep(0.1)
else:
    print("✅ 所有序列已在本地准备就绪！\n")

print("⚙️ 正在拼装数据并生成完美的 CSV 文件...")

def process_dataframe(df, seq_dict):

    df['seq_a'] = df['protein_a_id'].map(seq_dict)
    df['seq_b'] = df['protein_b_id'].map(seq_dict)

    df_clean = df.dropna(subset=['seq_a', 'seq_b']).copy()

    df_final = df_clean[['protein_a_id', 'protein_b_id', 'seq_a', 'seq_b', 'label']]
    return df_final

df_train_final = process_dataframe(df_train, seq_dict)
df_train_final.to_csv("train.csv", sep=',', index=False)
print(f"📝 1. train.csv 生成完毕！(有效数据: {len(df_train_final)} 条)")

df_test_final = process_dataframe(df_test, seq_dict)
df_test_final.to_csv("test.csv", sep=',', index=False)
print(f"📝 2. test.csv 生成完毕！ (有效数据: {len(df_test_final)} 条)")

df_merged = pd.concat([df_train_final, df_test_final], ignore_index=True)
df_merged.to_csv("dataset_merged_all.csv", sep=',', index=False)
print(f"📝 3. dataset_merged_all.csv (合并总文件) 生成完毕！(总数据: {len(df_merged)} 条)")

print("\n🎉 全部大功告成！你现在拥有了独立的 train.csv、test.csv，以及包含它们所有的 dataset_merged_all.csv！")
