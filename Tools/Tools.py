import warnings
warnings.filterwarnings("ignore")

from langchain.tools import StructuredTool
from .FileQATool import ask_docment
from .FileTool import list_files_in_directory
from .FinishTool import finish
from .SearchTool import search

document_qa_tool = StructuredTool.from_function(
    func=ask_docment,
    name="AskDocument",
    description="根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。",
)

directory_inspection_tool = StructuredTool.from_function(
    func=list_files_in_directory,
    name="ListDirectory",
    description="探查文件夹的内容和结构，展示它的文件名和文件夹名",
)

search_from_internet = StructuredTool.from_function(
    func=search.run,
    name="Search",
    description="调用浏览器搜索，获取即时信息",
)

finish_placeholder = StructuredTool.from_function(
    func=finish,
    name="FINISH",
    description="结束任务，将最终答案返回"
)
