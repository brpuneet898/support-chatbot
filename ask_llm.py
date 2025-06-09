import os
import PyPDF2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
import torch
from pprint import pprint
from IPython.display import Markdown as md
from groq import Groq
import yaml

# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)
model = None

groq_api_key = os.environ["GROQ_API_KEY"]

hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

system_prompt = """
You are an AI assistant trained to answer questions based on the context of a document. The context can be either from Student Handbook, or the Grading Document, or both.
When asked a question, you should refer to the provided context only and respond with the most accurate and relevant answer.
Answer only based on the provided context and do not make up any information.
"""

client = Groq(api_key=groq_api_key)

def query_model_with_rag(query, vector_store):
    # print(f" Recieved query is {query}")
    relevant_chunks = vector_store.similarity_search(query, k=16)
    relevant_content = [f"{i}: {doc.page_content}" for i, doc in enumerate(relevant_chunks, start=1)]
    context = "\n".join(relevant_content)
    # print(f" Recieved CONTEXT is {context}")
    prompt = f'{system_prompt}\n\nContext:\n{context}\n\nQuestion:  {query}\nAnswer:'
    # print("LENGTH OF PROMPT IN WORDS IS: ", len(prompt.split()))
    # with open("prompt.txt","w", encoding="utf-8") as f:
    #     f.write(prompt)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        stream=False
    )
    return chat_completion.choices[0].message.content.strip()