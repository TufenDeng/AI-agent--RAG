#使用streamlit写一个网页，用来与算法助理交互
#目前全球最火的、专为 机器学习和 AI 开发者 准备的开源 Web 应用框架

#为了使app能在云端运行
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st#缩写,并且是使用streamlit run app.py去运行
import os
# 强制禁用 ChromaDB 的遥测系统，防止它调用出问题的库
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1.页面配置
st.set_page_config(page_title="CCPC算法教练(代码陪练款)",page_icon="🤖",layout="wide")#使用宽屏模式

#加载环境变量
load_dotenv()

# 2.缓存加载模型，避免每次刷新页面都重新加载
@st.cache_resource#装饰器模型只在第一次启动时加载到内存
def init_resource():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5",model_kwargs={'device': 'cpu'})
    # 修改 db 的加载方式，加入配置关掉遥测
    import chromadb
    from chromadb.config import Settings
    
    client_settings = Settings(is_telemetry_enabled=False) # 关掉遥测
    
    db = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embeddings,
        client_settings=client_settings # 传入配置
    )
    llm = ChatOpenAI(
        model='deepseek-chat', 
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"), 
        openai_api_base='https://api.deepseek.com/v1'
    )
    return db, llm

db,llm=init_resource()




# 3.增加侧边栏:代码展示栏
with st.sidebar:
    st.title("代码工作台")
    st.markdown("将你的代码贴在这里，教练会结合资料为你审计。")
    user_code=st.text_area("C++/Python 代码:", height=400, placeholder="在此粘贴你的代码...")

    st.divider()#生成了一条分割线

    k_value=st.slider("检索深度（k）",1,10,5)#范围为1~10，初始值设为5
    if st.button("清空所有对话"):
        st.session_state.messages = []
        st.rerun()#增加一个清空功能

# 4.主界面
st.title("🤖 CCPC 算法智能教练")
st.caption("基于 OI-Wiki 知识库的专业算法指导与代码审计")

# 如果没有历史消息，初始化
if "messages" not in st.session_state:#短期记忆保险箱，防止某些数据被删除
    st.session_state.messages=[]

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):#在streamlit里面，with代表了一个视觉上的容器,所有的东西都放进来一个聊天框
        st.markdown(message["content"])#专门处理了markdown文件，使得在网页上显示得漂亮

# 5.重点修改逻辑，用户既可以提问，也可以提交代码进行审计
# 增加”意图识别“的功能
if prompt:=st.chat_input("提问或是发送“审计”来检查代码"):

    #-----意图判断逻辑-----
    is_audit_mode = False#用来清楚地告诉ai，啥时候听指令，啥时候自动审计
    if user_code and (len(prompt) < 5 or any(word in prompt for word in ["审计", "看看", "检查", "代码", "bug", "错"])):#以提示词简短为自动审计的依据
        is_audit_mode = True
        display_prompt = f"🔍 正在为您审计左侧代码...\n(指令: {prompt})" if len(prompt) > 1 else "🔍 正在为您自动审计左侧代码..."
    else:
        display_prompt = prompt

    # 展示用户意图，将用户输入放到历史
    st.session_state.messages.append({"role": "user", "content": display_prompt})
    with st.chat_message("user"):
        st.markdown(display_prompt)

    # RAG流程
    with st.spinner("教练正在翻阅资料..."):
        #A 检索
        # 增加有代码的情况
        search_query=f"{prompt}{user_code[:100]}"
        docs = db.similarity_search(search_query, k=k_value)
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # B. 增加工作模式判定，构造提示词（针对重复问题进行了优化）
        template = """
        你是一个顶级的 CCPC 竞赛教练。
        你手头有一份来自 OI-Wiki 的专业参考资料。

         ### 你的工作模式判定：
        1. **自动审计模式**：如果用户没有提出具体问题，请根据资料库全方位检查代码的：
           - 逻辑正确性（算法选型是否正确）
           - 潜在 Bug（溢出、初始化、边界条件）
           - 复杂度性能（是否能通过常规竞赛时限）
        2. **定向解答模式**：如果用户提出了具体问题，请以解决该问题为主，代码审计为辅。

        ###任务要求
        1. **对比分析**：如果用户提供了代码，请对比资料库中的标准算法，检查其逻辑是否正确。
        2. **潜在风险提示**：
           - 检查是否有 `int` 溢出（如：$10^{{18}}$ 是否用了 `long long`）？
           - 检查数组大小（如：双向边是否开了 2 倍空间）？
           - 检查时间复杂度（如：Dijkstra 必须用优先队列优化吗）？
        3. **启发式回复**：不要直接重写全部代码，通过提问或指出错误行来引导用户
           - 如果是审计模式，请分条列出 [逻辑]、[风险]、[建议]。
           - 如果是定向解答，指出问题所在。
        
        ### 参考资料：
        {context}

        ### 用户提问：
        {question}

        ### 用户当前代码：
        ```cpp
        {code}
        ```
        """
        prompt_tpl = ChatPromptTemplate.from_template(template)

        #code_black处理格式
        formatted_code=f"```cpp\n{user_code}\n```" if user_code else "（未提供代码）"

        full_prompt = prompt_tpl.format(context=context_text,question=search_query,code=formatted_code)

        # C 调用LLM并流式显示答案
        with st.chat_message("assistant"):
            response=llm.invoke(full_prompt)
            answer=response.content
            st.markdown(answer)
             # 这里的 st.expander 可以展示“信源”，点击可看
            with st.expander("查看参考资料来源"):
                if docs:
                    for i, doc in enumerate(docs):
                    # 注意：这里是 doc (单数)，对应 enumerate 出来的变量
                        source_name = doc.metadata.get('source', '未知来源')
                        st.write(f"排名 {i+1} 的相关文档: {source_name}")
                else:
                    st.write("未找到匹配的参考资料")
    # 将助手回答加入历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
    