#Document对象和chunking区分
import os
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#TextLoader:读取单个文件，提取文字返回document
#Document:巡视整个文件夹；运行逻辑为，你告诉它要读取.md文件，它就在文件夹里找到后缀为md的，叫textLoader来读取

#指定数据路径
DATA_PATH="data"

def load_documents():
    #从文件中递归加载所有特定类型的文件，这里给的是markdown
    print("正在加载文档...")
    loader=DirectoryLoader(#Loader是一个对象，括号里的内容是将这个对象实例化
        DATA_PATH,#指定数据路径
        glob="**/*.md",#读取路径下的所有md文件
        loader_cls=TextLoader,#使用的是这个工具读取文件
        show_progress=True#显示加载进程
    )
    docs=loader.load()#执行真的的“读文件”操作，把所有读到的文件保存到docs变量里
    print(f"读取了{len(docs)}个md文件")
    return docs

def split_documents(docs):#切分文件,长文件切成一个个小块（chunks），避免长文本导致的幻觉
    text_splitter=RecursiveCharacterTextSplitter(#这是一个文本切割器工具
        chunk_size=500,#每块500个字符
        chunk_overlap=50,#块与块之间的重叠部分为50个字符，保证了语义的连贯性
        add_start_index=True#记住每块在原文中的起始位置,比如说第1块的起始位置是0，第2块的起始位置就是450
    )
    chunks=text_splitter.split_documents(docs)
    print(f"文档切分完成，分成了{len(chunks)}个知识块")
    return chunks

if __name__=="__main__":
    #执行加载
    raw_docs=load_documents()

    #执行切分
    final_chunks=split_documents(raw_docs)

    #验证切分结果
    if len(final_chunks)>0:
        print("\n——————第一个知识块样例——————")
        print(f"内容摘要：\n{final_chunks[0].page_content[:100]}")#读取第1个板块到第100个字符
        print(f"元数据为：\n{final_chunks[0].metadata}")#这个块的背后信息，如，chunk是一瓶可乐，那么chunk.metadata就是生产日期之类的
        """
        metadata是一个字典，里面存着：
        source: "data/graph/dijkstra.md" （来源路径）
        start_index: 450 （坐标）
        language: "zh" （语言，如果你设置了的话）
        """


#解决疑问：什么是add_starter_index，什么是metadata
#还要安装文档处理辅助库：unstructured



