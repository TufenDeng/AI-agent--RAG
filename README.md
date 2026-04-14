项目简介：
     这是一个基于RAG技术开发的算法ai指导师，参考文献为OI_WIKI
     技术栈为python langchain embedding 

4.12 1.配置搭建RAG的虚拟环境，并从镜像站下载longChain，测试其能正常使用

     2.理解什么叫做RAG（检索增强生成），解决ai幻觉问题

     3.为什么使用ChatPromtTemplate这个类:它会把字符转化成消息队列（角色化的结构），这样AI就能更好地理解其设定，
     让template带有标签，区分哪些是“背景资料”，哪些是“用户的追问”，特别在做RAG时，提示词会非常长，不结构化就不太利于AI处理消息

     4.invoke():目前主流常用的传参方法，打包成字典
     
     5.做RAG，定义template（变量）时，变量是需要后面才传进来的（延时填充），不能带f（即时求值）
     LangChain Template：用于编排 AI 逻辑。它允许变量暂不存在，通过 {} 留出空位，等 invoke() 时再统一填入

4.13 1.编写ingest,用以清理数据
     2.理解DirectoryLoader和TextLoader的功能，并明白如何使用这两个类实例化对象
     3.理解什么是元数据和块的起始位置

4.14 0.对于昨天的文件切割，不同的文件的处理逻辑不同，应该使用不同的切割工具，分割逻辑应该根据文件的“语义结构”来定
     今天就是跑通RAG的大脑
     1.安全开发：使用单独的.env文件存储api
     2.理解embedding模型和llm模型的调用
     3.环境攻坚：解决了国内访问 Hugging Face 模型的网络瓶颈（HF_ENDPOINT 镜像设置）
     4.RAG 逻辑：理清了检索（Similarity Search）、数据清洗（join 拼接）到最终生成（DeepSeek 调用）的全流程
     *remaining problem:数据库并没有正确地转化近万个知识块，问题待排查