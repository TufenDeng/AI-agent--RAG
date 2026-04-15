import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"#还是需要保持环境的一致性，不然需要获取向量化模型的时候又慢
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

#加载数据库
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={'device': 'cpu'} )
db=Chroma(persist_directory="chroma_db",embedding_function=embeddings)#问题本身也需要被转化为向量，这个转化过程用到的模型一致

#接入deepseek
llm=ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base='https://api.deepseek.com/v1'
)

#设计教练风格，使用这样的格式可以一眼看清楚给ai下的指令逻辑
template="""
你是一个严谨的ccpc算法教练，
请根据以下参考资料回答问题。如果资料里没写，就说你不知道，不要瞎编
参考资料：
{context}

用户问题：
{question}
"""

prompt_template = ChatPromptTemplate.from_template(template)

query = "dijkstra的时间复杂度是多少"

print(f"\n> 正在本地数据库中搜寻关于 '{query}' 的资料...")
# 检索最相关的 3 个片段
docs = db.similarity_search(query, k=10)#这涉及到一个检索精度，如果精度太小，可能查不到要问的问题的参考文件
context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])#这是给ai看的，ai看不懂python给的列表，必须是纯字符；join是把这三个文本使用\n---\n连接起来

print("> 资料已找到，正在请 DeepSeek 教练总结并回答...\n")
full_prompt = prompt_template.format(context=context_text, question=query)#这时候才传参
response = llm.invoke(full_prompt)

print("=== 教练的回答 ===")
print(response.content)
