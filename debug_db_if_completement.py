from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 加载数据库
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# 1. 查看数据库里一共有多少条记录
count = db._collection.count()
print(f"数据库中总共有 {count} 条知识块。")

# 2. 看看这些记录都来自哪些文件（取前10个看一眼）
print("\n前10条数据的来源预览：")
all_data = db.get(limit=10)
for metadata in all_data['metadatas']:
    print(metadata.get('source'))