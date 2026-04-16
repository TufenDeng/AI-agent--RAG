#使用streamlit写一个网页，用来与算法助理交互
#目前全球最火的、专为 机器学习和 AI 开发者 准备的开源 Web 应用框架

import streamlit as st#缩写
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1.页面配置
st.set_page_config(page_title="CCPC算法教练",page_icon="🤖")

#加载环境变量
load_dotenv()

# 2.缓存加载模型，避免每次刷新页面都重新加载
@st.cache_resource#装饰器模型只在第一次启动时加载到内存
def init_resource():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com/v1'
    )
    return db, llm

db,llm=init_resource()

# 3.网页界面布局
st.title("🤖 CCPC 算法智能教练")
st.caption("基于 OI-Wiki 知识库的专业算法指导")

# 如果没有历史消息，初始化
if "messages" not in st.session_state:#短期记忆保险箱，防止某些数据被删除
    st.session_state.messages=[]

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):#在streamlit里面，with代表了一个视觉上的容器,所有的东西都放进来一个聊天框
        st.markdown(message["content"])#专门处理了markdown文件，使得在网页上显示得漂亮

# 4.聊天逻辑
if prompt:=st.chat_input("请问关于算法的问题（如：什么是并查集）"):
    # 将用户输入放到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG流程
    with st.spinner("教练真正翻阅资料..."):
        #A 检索
        docs = db.similarity_search(prompt, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # B. 构造提示词（针对重复问题进行了优化）
        template = """
        你是一个严谨的 CCPC 算法竞赛教练。
        请根据以下参考资料回答问题。

        ### 要求：
        1. 必须优先使用资料中的内容。
        2. **禁止重复**：如果资料中有重合的内容，请合并总结，不要机械重复。
        3. **格式化**：使用清晰的标题和点句符。数学公式请使用 LaTeX 格式。
        4. 如果资料没写，就说不知道。

        ### 参考资料：
        {context}

        ### 用户问题：
        {question}
        """
        prompt_tpl = ChatPromptTemplate.from_template(template)
        full_prompt = prompt_tpl.format(context=context_text, question=prompt)

        # C 调用LLM并流式显示答案
        with st.chat_message("assistant"):
            response=llm.invoke(full_prompt)
            answer=response.content
            st.markdown(answer)
             # 这里的 st.expander 可以展示“信源”，点击可看
            with st.expander("查看参考资料来源"):
                for i, doc in enumerate(docs):
                    st.write(f"来源 {i+1}: {doc.metadata.get('source')}")

    # 将助手回答加入历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
    