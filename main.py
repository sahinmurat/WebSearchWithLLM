import os
from dotenv import load_dotenv
import bs4
from langchain_classic import text_splitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma, vectorstores
from langchain_core.tools import retriever
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ.setdefault("USER_AGENT", "4webSearch/1.0")

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

loader = WebBaseLoader(

    web_path="https://lilianweng.github.io/posts/2026-07-04-harness/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-body", "post-body-inner")
        )),
)

docs = loader.load()
text_splitter = text_splitter.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstores = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstores.as_retriever(search_kwargs={"k": 1})


# ra prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from langsmith import Client

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

if __name__ == "__main__":
    for chunk in rag_chain.stream("Harness Layer vs Core Intelligence?"):
        print(chunk, end="", flush=True)
