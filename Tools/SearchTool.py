from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
search = SerpAPIWrapper()
