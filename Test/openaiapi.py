from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
llm = OpenAI()
completion = llm.chat.completions.create(
    model="gpt-4o-mini",
    # model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "halo."
        }
    ]
)

print(completion.choices[0].message)
