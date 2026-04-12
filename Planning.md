## 5月：数据工程（Data Ingestion）

核心任务： 让 AI 拥有 OI-Wiki 的全量知识。

技术栈： Python 3.12, LangChain, Git.

实操动作：

- 克隆数据： Git clone OI-Wiki 仓库。
- 编写加载器： 使用 DirectoryLoader 和 UnstructuredMarkdownLoader 加载文件。
- 逻辑切分： 学习 MarkdownHeaderTextSplitter，按照 #、## 标题层级切分文档，确保知识块的逻辑完整。

产出： 一个能将数千个 Markdown 文件转化为干净、结构化“知识块”的 Python 脚本。

## 6月：存储与检索（Retrieval）

核心任务： 实现“毫秒级”的语义搜索。

技术栈： ChromaDB, Embedding Models (如 OpenAI 或 HuggingFace 开源模型)。

实操动作：

- 向量化： 调用 Embedding 接口将文本块转为数字向量。
- 持久化： 将向量存入 ChromaDB 数据库，确保下次启动不用重新训练。
- 检索调优： 尝试 Similarity Search 和 MMR (最大边际相关性)，防止搜出来的内容太重复。

产出： 一个可以根据用户问题，自动在数据库中找到最匹配的“线段树”或“并查集”原文片段的系统。

## 7月：对话链与 UI（Agent & UI）

核心任务： 变成一个真正的“智能教练”。

技术栈： Chain-of-Thought, Streamlit 或 Gradio.

实操动作：

- Prompt 工程： 编写 System Prompt，规定 AI 的语气：“你是一个严厉的算法教练，请不要直接给代码，先提示思路”。
- 对话历史： 引入 BufferMemory，让 AI 记得你刚才问过的上一个问题。
- 前端展示： 用 Streamlit（只需几行 Python）写一个网页界面，左边刷算法题，右边问助手。

产出： 一个完整的、可运行的、可以写在简历上的“基于 RAG 架构的算法竞赛智能助手”。
