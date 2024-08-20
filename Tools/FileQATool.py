from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from Models.Factory import ChatModelFactory, EmbeddingModelFactory
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        print(ext)
        if ext == "pdf":
            return PyMuPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")


def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]


def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages


def ask_docment(
        filename: str,
        query: str,
) -> str:
    """根据一个PDF文档的内容，回答一个问题"""

    raw_docs = load_docs(filename)
    if len(raw_docs) == 0:
        return "抱歉，文档内容为空"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    documents = text_splitter.split_documents(raw_docs)
    if documents is None or len(documents) == 0:
        return "无法读取文档内容"

    db = Chroma.from_documents(documents, EmbeddingModelFactory.get_default_model(),persist_directory=r"D:\ChromaData\LangchainChromaData")
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatModelFactory.get_default_model(),  # 语言模型
        chain_type="stuff",  # prompt的组织方式，简单拼接
        retriever=db.as_retriever()  # 检索器
    )
    response = qa_chain.run(query + "(请用中文回答)")
    return response


if __name__ == "__main__":
    filename = r"D:\Code\Python\autogpt\auto-gpt-work\data\Buffer of Thoughts.pdf"
    query = "buffer of thought是什么"
    response = ask_docment(filename, query)
    print(response)
