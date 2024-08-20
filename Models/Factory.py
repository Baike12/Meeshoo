import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "model_kwargs": {"seed": 42},
    }

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if "gpt" in model_name:
            if not use_azure:
                return ChatOpenAI(model=model_name, **cls.model_params)
            else:
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                    **cls.model_params
                )
        elif model_name == "deepseek":
            return ChatOpenAI(
                model="deepseek-chat",
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
                **cls.model_params,
            )

    @classmethod
    def get_default_model(cls):
        return cls.get_model("gpt-3.5-turbo")


class EmbeddingModelFactory:

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("text-embedding"):
            if not use_azure:
                return OpenAIEmbeddings(model="text-embedding-ada-002")
            else:
                return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("text-embedding-ada-002")
