# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from Agent.Meeshoo import Meeshoo
from Models.Factory import ChatModelFactory
from Tools import *
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory


def launch_agent(agent: Meeshoo):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    smile_icon = "\U0001F60A"
    chat_history = ChatMessageHistory()

    while True:
        task = input(f"{ai_icon}：请吩咐{smile_icon}\n{human_icon}：")
        if task.strip().lower() == "quit":
            break#
        reply = agent.run(task, chat_history, verbose=True)
        print(f"{ai_icon}：{reply}\n")


def main():

    # 语言模型
    llm = ChatModelFactory.get_model("gpt-4o")

    # 自定义工具集
    tools = [
        document_qa_tool,
        directory_inspection_tool,
        finish_placeholder,
        search_from_internet,
    ]

    # 定义智能体
    agent = Meeshoo(
        llm=llm,
        tools=tools,
        work_dir="./data",
        main_prompt_file="./prompts/main/main.txt",
        max_thought_steps=20,
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__":
    main()
