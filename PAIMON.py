import re
import os
import json
import time

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from adjustment.openai import ChatOpenAI
from embedding import Embedding

class PAIMON:
    name: str
    info: str
    verbose: bool
    memory_window: int
    chain = None
    model = None
    memory = None
    prompt = None
    embedding = None

    def __init__(self, memory_window=5, bot_name="PAIMON", verbose=False):
        self.name = bot_name
        self.verbose = verbose
        self.model = ChatOpenAI(
            model_name="Qwen",
            openai_api_base="http://localhost:8001/v1",
            openai_api_key="EMPTY",
            streaming=False
        )
        self.embedding = Embedding(512, 128, self.name)
        self.memory_window = memory_window
        self.build_chain()

    def load(self, file_list:list=[]):
        start_time = time.time()
        if self.embedding.load(file_list):
            print("向量库加载完成，耗时：", time.time() - start_time)
        else:
            print(f"向量库加载失败，未创建过向量库“{self.name}”，且未加入任何文件，将无法进行搜索。")
    
    def clear_history(self):
        if self.memory is not None:
            self.memory.clear()

    def save_info(self, info:dict):
        self.info = "资料：{}\n问题：{}".format(info["reference"], info["question"])
        return info
    def save_history(self, output):
        if self.memory is not None:
            self.memory.save_context({"input": self.info}, {"output": output.content})
        print(self.memory.load_memory_variables({}))
        self.full_prompt = ""
        return output

    def build_chain(self):
        messages = [
            SystemMessagePromptTemplate.from_template("你将扮演一个教务处的工作人员，根据给定的资料，回答学生的问题。下面你将会收到一些资料，以“下面开始回答问题”结束。请分析这些资料，然后详细回答给定的问题。注意，如果提供的资料并不能告诉你问题的答案，请输出“给定的资料无法回答这个问题”。"),
            HumanMessagePromptTemplate.from_template("对话历史：{history}\n（如果对话历史不能帮你解决现有问题，请忽略其中的信息）\n资料：{reference}\n（如果资料不能帮你解决现有问题，请忽略其中的信息）\n下面开始回答问题：{question}")
        ]
        self.prompt = ChatPromptTemplate.from_messages(messages)
        self.memory = ConversationTokenBufferMemory(
            llm = self.model,
            memory_key = "history",
            max_token_limit = 25*1024,
            return_messages = True
        )
        self.chain = (
            RunnableLambda(self.save_info)
            | RunnablePassthrough.assign(
                history = RunnableLambda(self.memory.load_memory_variables) | itemgetter("history")
            )
            | self.prompt
            | self.model
            | RunnableLambda(self.save_history)
        )
    
    def search(self, question:str):
        if self.embedding.vec_db is not None:
            return self.embedding.search(question, 0.5)
        else:
            return []

    def chat(self, question:str, reference:list[Document]=[]) -> dict:
        result = self.chain.invoke({
            "question": question,
            "reference": "\n".join([i.page_content for i in reference])
        })
        return {
            "question": question,
            "answer": result.content
        }

    def rag(self, question:str) -> dict:
        reference = self.search(question)
        result = self.chat(question, reference)
        result.update({
            "reference": [{
                "content": i.page_content,
                "metadata": i.metadata
            } for i in reference]
        })
        return result

chatbot = PAIMON(verbose=True, bot_name="TEST")

from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route("/RAG/chat", methods=["POST"])
def RAG_chat():
    ret = chatbot.rag(
        question = request.json["question"]
    )
    return jsonify(ret)

@app.route("/model/chat", methods=["POST"])
def model_chat():
    ret = chatbot.chat(
        question = request.json["question"]
    )
    return jsonify(ret)

@app.route("/clear", methods=["GET"])
def clear():
    chatbot.clear_history()
    return "OK"

if __name__ == "__main__":
    # dir_list, name_list = ["./data/", "./data/website/"], []
    # for i in dir_list:
    #     for j in os.listdir(i):
    #         if os.path.isfile(i + j):
    #             name_list.append(i + j)
    name_list = ["./data/" + i for i in [
        "【2023级本科生】美育核心课选课通知.md",
        "【2023级本科生】新生入学教务活动安排.md",
        # "第二章：需求、供给与价格机制.pdf",
        "苏州校区.docx",
        # "选课.csv"
    ]]
    chatbot.load(name_list)
    # chatbot.load()
    # chatbot.load(['南哪QA.qa'])
    app.run(host="127.0.0.1", port=8002)