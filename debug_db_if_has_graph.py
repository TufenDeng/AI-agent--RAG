from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 1. 获取数据库中所有的元数据 (metadatas)
print("正在读取所有元数据进行统计...")
all_data = db.get() #获取db的所有数据
metadatas = all_data['metadatas']#获取所有数据下的元数据，查找来源

# 2. 统计一共有多少个独立的文件
unique_sources = set([meta.get('source') for meta in metadatas])#把独立文件打包成一个列表，并且是去重的（）

print(f"\n--- 数据库统计报告 ---")
print(f"总知识块数量: {len(metadatas)}")
print(f"独立文件数量: {len(unique_sources)}")

# 3. 检查你关心的文件夹是否存在
print("\n--- 文件夹覆盖检查 ---")
categories = ["graph", "dp", "string", "math", "ds"] # 常见的 OI-Wiki 文件夹；[]定义了一个列表
for cat in categories:
    # 看看有没有任何一个 source 包含这个路径
    found = any(cat in source.lower() for source in unique_sources)#只要有一个包含就是true
    status = "✅ 已包含" if found else "❌ 未找到"
    print(f"{cat:10}: {status}")#：10是指给这个变量预留10个字符的位置，变成了一个对齐的表格

# 4. 随机打印 5 个不同的来源看看
#这一步是检查来源的正确性
print("\n随机抽取 5 个不同文件的路径：")
import random
sample_sources = random.sample(list(unique_sources), min(5, len(unique_sources)))#random.sample不支持对无序的set操作
for s in sample_sources:
    print(f"- {s}")