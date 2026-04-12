from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# 1. 定义一个简单的提示词模板
template = "你是一个{difficulty}难度的CCPC 算法教练。请解释什么是 {topic}。"##{}是占位符，表示这是一个代填的坑；f代表了马上就要填入变量
prompt_template = ChatPromptTemplate.from_template(template) ##from_template 方法把这一行普通的字符串，转化成了一个智能模板对象。这个对象知道自己需要一个叫 topic 的参数
##而使用ChatPT这个类，会让template带有标签，区分哪些是“背景资料”，哪些是“用户的追问”

# 2. 模拟填充数据
formatted_prompt = prompt_template.invoke({"difficulty":"中等","topic":"并查集"})##正式需要喂给AI的提示词
##.format() 是一个“通用快捷方法”。它的设计初衷是：无论模板多复杂，最后都给我吐出一个最简单的字符串（str）
##所以还是要用.invoke()，自动识别是哪个类，type才会变得不同（注意语法问题。。。，因为要打包传参）

print("ChatPT生成的对象类型是：",type(formatted_prompt))
print("ChatPT生成的提示词是：")
print(formatted_prompt)

# 如果能正常打印出：你是一个 CCPC 算法教练。请解释什么是 并查集。
# 说明你的 LangChain 环境已经完全就绪！



template2 ="你是一个专业的ccpc算法教练，请解释什么是{topic2}。"
prompt_template2 = PromptTemplate.from_template(template2)

formatted_prompt2 = prompt_template2.invoke({"topic2":"快速幂"})

print("PT生成的对象类型是：",type(formatted_prompt2))
print("PT生成的提示词是：")
print(formatted_prompt2)