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
You are an AI assistant designed to answer questions strictly based on the provided context, which consists of excerpts from Student Handbook and Grading document. The context includes a mix of paragraphs and tables which can contain grading policies of various subjects, and features of the program, including any rules or terms and conditions and deadlines that must be followed by the students.

Your task is to extract the most accurate and relevant answer solely from the given context.
- Do not rely on external knowledge or assumptions.
- If the answer requires data from a table, read and interpret the table carefully. Remember that the tables are given in HTML format.
- DO NOT mention the context from which the answer is derived. Because users don't have access to the context.
- Your response should be Clear, complete, human-readable, using bullet points or structured format when helpful.
- If the answer is not found in the context, strictly respond with the following message without any modification: "The answer is not available in the provided context. It would be helpful if you can include the heading of the section that you are looking for. Try to include relevant keywords from the heading and section to get the best answer. If you didn't get the answer in two attempts, better contact the support team directly."
"""

client = Groq(api_key=groq_api_key)

def query_model_with_rag(query, vector_store, grading_doc_chunks, k=4):
    relevant_chunks = vector_store.similarity_search(query, k=k) # "k" specifies the number of most similar chunks to retrieve.
    context = []

    # print(f"Relevant chunks found are : {relevant_chunks}\n\n***********************************\n\n")
    # if not relevant_chunks:
    #     return [], "The answer is not available in the provided context. It would be helpful if you can include the heading of the section that you are looking for. Try to include relevant keywords from the heading and section to get the best answer. If you didn't get the answer in two attempts, better contact the support team directly."
    for i, doc in enumerate(relevant_chunks):
      if doc.page_content in grading_doc_chunks:
        context.append(f"Extract from Grading Document:\nHeading:{doc.page_content}\nContent:{grading_doc_chunks[doc.page_content]}")
      else:
        context.append(f"{doc.page_content}")


    final_context = "\n\n".join(f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context))
    prompt = system_prompt + "\n\n" + "Context Starts {\n"+ final_context + "\n} Context Ends" + "\n\nUser Question: " + query + "\nAnswer:"
    with open("prompt.txt","w", encoding="utf-8") as f:
      f.write(prompt)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        stream=False
    )
    return chat_completion.choices[0].message.content.strip()