import os
import json
from tqdm import tqdm
import sys
import pandas as pd

DATA = r"C:\E\works\CC\4\BlockA\RAG\data\plot_data\theorems_Qwen3-Embedding-0.6B_CO_50k_only6_projection.jsonl"


def read_json(jsonl, json):

    with open(json, "w", encoding="utf-8") as cf:
        with open(jsonl, "r", encoding="utf-8") as jf:
            try:
                cf.write('[')
                line_num = 1
                line = jf.readline()
                cf.write(line)
                line = jf.readline()
                while line:
                    cf.write(',')
                    cf.write(line)
                    line_num += 1
                    line = jf.readline()
                cf.write(']')
            except Exception as e:
                print(f"[Warning] Fail to parse JSON in line {line_num}: {e}")


read_json(DATA, "data_CO.json")


df = pd.read_json('data_CO.json')


# convert the coordinate list into separate columns
df['x'] = df['projection_coords'].apply(lambda x: x[0])
df['y'] = df['projection_coords'].apply(lambda x: x[1])
df.drop(columns='projection_coords',inplace=True)


# clean categories
df['categories'] = df['categories'].apply(lambda x: str(x).split(' ')[0])
print(f"before: {len(df['categories'])}")
df = df[df['categories'].apply(lambda x: x.startswith('math.'))]
print(f"after: {len(df['categories'])}")

df.to_csv('data_CO_clean_v1.csv', encoding='utf-8')

