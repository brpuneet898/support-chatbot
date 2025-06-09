import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import re, requests
from bs4 import BeautifulSoup, NavigableString, Tag

def create_vector_store_from_json(json_file_path, vector_store_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    all_documents = []
    total_characters = 0

    for entry in data:
        subject = entry.get('subject', '')
        description = entry.get('description', '')

        doc_content = f"Subject: {subject}\nDescription: {description}"

        document = Document(
            page_content=doc_content,
            metadata={"subject": subject, "description": description}
        )

        all_documents.append(document)
        total_characters += len(doc_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128000,  
        chunk_overlap=200,  
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_documents(split_documents, embeddings)

    document_ids = vector_store.add_documents(documents=split_documents)
    vector_store.save_local(vector_store_path)

    return vector_store, total_characters, len(data), split_documents, len(document_ids)

def create_vector_store_from_json_using_subject(json_file_path, vector_store_path):
    """
    Create a FAISS vector store using only 'subject' field for embeddings.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_documents = []
    total_characters = 0

    for entry in data:
        subject = entry.get('subject', '').strip()  # Use only subject for similarity search
        description = entry.get('description', '').strip()  # Store description as metadata only

        document = Document(
            page_content=subject,  # Store only subject for embeddings
            metadata={"subject": subject, "description": description}  # Keep description in metadata
        )

        all_documents.append(document)
        total_characters += len(subject)  # Count only subject characters

    # Text splitting is optional for short subjects, but keeping it in case of long text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128000,  
        chunk_overlap=200,  
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load existing vector store if available, otherwise create a new one
    if os.path.exists(vector_store_path):
        vector_store = FAISS.load_local(
            vector_store_path, embeddings, allow_dangerous_deserialization=True
        )
        document_ids = vector_store.add_documents(documents=split_documents)
    else:
        vector_store = FAISS.from_documents(split_documents, embeddings)
        document_ids = list(range(len(split_documents)))  # Assign dummy IDs

    vector_store.save_local(vector_store_path)

    return vector_store, total_characters, len(data), split_documents, len(document_ids)

def load_vector_store(vector_store_path):
    """
      This function is used to load a vector store that has been saved to disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # This is the model that was used to create the vector store
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

# HELPER FUNCTIONS
import re
import requests
from bs4 import BeautifulSoup
from functools import lru_cache
def remove_unicode_escapes(text):
    # Regex to match \u followed by exactly 4 hexadecimal digits
    cleaned_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text).replace("\xa0","")
    return cleaned_text

def strip_html_attributes(html_str):
    """
    Remove all attributes from an HTML string.
    For example, <table class="x" id="y"> becomes <table>.
    """
    soup = BeautifulSoup(html_str, "html.parser")

    for tag in soup.find_all(True):  # True matches all tags
        tag.attrs = {}  # Remove all attributes

    return str(soup).replace("<span>","").replace("</span>","").replace("</p></td>","</td>").replace("<p><td>","<td>")

def normalize(text):
    # Replace any kind of whitespace (space, \xa0, tabs, etc.) with a single space
    return re.sub(r'\s+', ' ', text.replace('\xa0', ' ').strip())

def html_table_to_json(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    # Extract headers
    headers = []
    header_row = table.find("tr")
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

    # Extract rows
    data = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        row_data = {headers[i] if i < len(headers) else f"col_{i}": cell.get_text(strip=True)
                    for i, cell in enumerate(cells)}
        data.append(row_data)

    return data

def remove_unwanted_keys_from_grading_doc(grading_doc_dict):
  keys_to_remove = ["Foundation level courses", "ASSIGNMENT DEADLINES:","Bonus Marks","Suggested pathway to register and study Foundation level courses:","Diploma Level courses","Suggested pathway to register and study Diploma level courses:","Diploma level courses","Degree Level courses","Annexure I"]
  for key in grading_doc_dict.keys():
    if (
        key.strip()=="" # Key is empty
        or
      len(grading_doc_dict[key].strip().split())<=10 # Value has less than 10 words
      ) and (key not in keys_to_remove): # if value is an empty string/list and key is still not in rejected list
      keys_to_remove.append(key)

  # print(len(keys_to_remove))

  original_keys = grading_doc_dict.keys()
  for key in keys_to_remove:
    try:
      if key in original_keys:
        grading_doc_dict.pop(key)
    except KeyError:
      print(f"Key '{key}' not found in the dictionary.")

  print("Keys removed from the grading documents dictionary")

def remove_unwanted_keys_from_student_hand_doc(student_doc_dict):
  keys_to_remove = []
  for key in student_doc_dict.keys():
    if (key.strip()=="" # Key is empty
        or
      len(student_doc_dict[key].strip().split())<=10 # Value has less than 10 words
      ) and (key not in keys_to_remove): # if value is an empty string/list and key is still not in rejected list
      keys_to_remove.append(key)

  # print(len(keys_to_remove))

  original_keys = student_doc_dict.keys()
  for key in keys_to_remove:
    try:
      if key in original_keys:
        student_doc_dict.pop(key)
    except KeyError:
      print(f"Key '{key}' not found in the dictionary.")

  print("Keys removed from the student handbook dictionary")


from bs4 import BeautifulSoup, NavigableString, Tag

@lru_cache(maxsize=None)
def extract_content_by_headers_with_tables(url, headers=("h1", "h2", "h3", "h4", "h5", "h6")):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    content = {}
    all_headers = soup.find_all(headers)

    for i, header in enumerate(all_headers):
        section_title = header.get_text(strip=False)
        section_title = normalize(re.sub(r'^[^a-zA-Z]+', '', section_title)) # Remove leading non-alphabetic characters from the key
        section_title = remove_unicode_escapes(section_title)
        section_content = []

        next_node = header.find_next_sibling()
        while next_node and next_node not in all_headers:
            if isinstance(next_node, Tag):
                if next_node.name == "table":
                    html_table_string = strip_html_attributes(str(next_node))
                    # json_table = html_table_to_json(html_table_string)
                    section_content.append(f"\n\n{str(html_table_string)}\n\n")  # preserve table HTML
                else:
                    # Gather only direct children text without duplication
                    for child in next_node.children:
                        if isinstance(child, NavigableString):
                            text = child.strip()
                            if text:
                                section_content.append(text)
                        elif isinstance(child, Tag) and child.name != "table":
                            text = child.get_text(separator=' ', strip=True)
                            if text:
                                section_content.append(text)
            next_node = next_node.find_next_sibling()

        # Join with space to avoid newlines, but keep HTML tables as-is

        content[section_title] = ' '.join(section_content)
        content[section_title] = normalize(re.sub(r'^[^a-zA-Z]+', '', content[section_title]))
        content[section_title] = remove_unicode_escapes(content[section_title])

    return content

# grading_doc_chunks = None  # Define at module level

def create_vector_store_from_docs(student_handbook_url, grading_doc_url):
    global grading_doc_chunks  # Declare as global
    grading_doc_chunks = extract_content_by_headers_with_tables(grading_doc_url)
    remove_unwanted_keys_from_grading_doc(grading_doc_chunks)
    print(f"Extracted {len(grading_doc_chunks)} sections from the grading document.")
    import json
    with open("grading_doc_chunks.json", "w", encoding="utf-8") as f:
        json.dump(grading_doc_chunks, f, ensure_ascii=False, indent=4)
    student_handbook_chunks = extract_content_by_headers_with_tables(student_handbook_url)
    remove_unwanted_keys_from_student_hand_doc(student_handbook_chunks)

    grading_doc_final_chunks = list(grading_doc_chunks.keys())
    student_handbook_final_chunks = [f"Extract from Student Handbook:\nHeading:{key}\nContent:{value}" for key, value in student_handbook_chunks.items()]

    combinined_chunks = grading_doc_final_chunks + student_handbook_final_chunks

    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(combinined_chunks, hf_embeddings)
    return vector_store, grading_doc_chunks


if __name__ == "__main__":
    student_handbook_url = "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub#h.104s7slbfmp7"
    grading_doc_url = "https://docs.google.com/document/d/e/2PACX-1vRKOWaLjxsts3qAM4h00EDvlB-GYRSPqqVXTfq3nGWFQBx91roxcU1qGv2ksS7jT4EQPNo8Rmr2zaE9/pub#h.cbcq4ial1xkk"
    latest_vector_store = create_vector_store_from_docs(student_handbook_url, grading_doc_url)
    vector_store_path = "LATEST_VECTOR_STORE"
    latest_vector_store.save_local(vector_store_path)

    