from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 1. 获取数据库中所有的文本内容
print("正在从数据库提取全量文本内容...")
all_data = db.get()
all_docs = all_data['documents']
all_metadatas = all_data['metadatas']

# 2. 暴力搜索关键词
target = "Dijkstra" # 试试 Dijkstra 或者 “最短路”
found_count = 0

print(f"\n--- 暴力搜索报告: 关键词 '{target}' ---")
for i in range(len(all_docs)):
    if target.lower() in all_docs[i].lower():
        found_count += 1
        if found_count <= 3: # 只打印前 3 条结果
            print(f"\n[匹配项 {found_count}]")
            print(f"来源文件: {all_metadatas[i].get('source')}")
            print(f"内容片段: {all_docs[i][:150]}...")

if found_count == 0:
    print(f"\n❌ 警告：数据库中完全没有包含 '{target}' 的知识块！")
    print("这说明 ingest.py 的数据采集阶段出了问题。")
else:
    print(f"\n✅ 成功：在数据库中找到了 {found_count} 个包含 '{target}' 的块。")
    print("这说明：数据在库里，但向量搜索 (similarity_search) 找不到它们。")