
from API import api
from Dns_serverTool import dns_lookup
from cli_execution import execute_cmd_from_prompt
from weather_Tool import extract_city_from_prompt, get_weather_summary
from rag_with_bert import RAGSystem
from web_scraping import web_scraping, extract_url_from_prompt
import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import os
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings
#----------------------------------------------------
api_key = api()
#-------------------------------------------------
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Initialize ChromaDB client
client = chromadb.Client(Settings(persist_directory="./chroma_store"))
collection = client.get_or_create_collection(name="bert_embeddings")
#---------------------------------------------------------------------------------------------
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        masked_hidden = last_hidden * mask
        summed = torch.sum(masked_hidden, dim=1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
    return mean_pooled.squeeze().numpy()
#---------------------------------------------------------------------------------------------------------------------
def Tool_Selection(user_prompt):
    """
    Select the best-matching tool for a user query from: Weather, DNS Lookup,Web Scraping, CLI,
    or RAG (general fallback).
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""You are a tool selector for user queries.
    Your task is to choose the most appropriate tool from the list below based on the user's prompt.
    Rules:
    - If the prompt contains a city name or appears to ask for temperature or forecast, choose "Weather".
    - If it asks about a domain (like google.com or DNS/IP), choose "DNS Lookup".
    - If it talks about extracting data from a webpage or HTML, choose "Web Scraping".
    - If it looks like a command or system action (like create folder, run program), choose "CLI".
    - Otherwise, choose "RAG".
    User Prompt:
    "{user_prompt}"
    """

    response = model.generate_content(prompt)
    return response.text.strip()
#---------------------------------------------------------------------------------------------------------------------
DATA_DIR = "C:/Users/Public/Internship/admin_data"
os.makedirs(DATA_DIR, exist_ok=True)
FILE_PATH = os.path.join(DATA_DIR, "admin_text.txt")
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model
tokenizer, model = load_bert()
role = st.sidebar.selectbox("Select Role", ["User", "Admin"])
#----------------------------------------ADMIN --------------------------------------------------------
if role == "Admin":
    st.set_page_config(page_title="Admin Panel", page_icon="👤", layout="wide")
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>
            Welcome to Admin Panel
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
            <style>
                .stTextInput > div > div > input {
                    background-color: #f0f2f6;
                    padding: 10px;
                    border-radius: 8px;
                }
            .stButton > button {
                background-color: #1f77b4;  /* Blue */
                foreground-color: white;
                color: white;
                padding: 8px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
            }
            .stButton > button:hover {
                background-color: #155a8a;  /* Darker Blue */
                }
                .custom-box {
                    background-color: #ffffff;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
                }
            </style>
        """, unsafe_allow_html=True)

    admin_text = st.text_area("Enter text here...")
    if st.button("Save Text"):
        with open(FILE_PATH, "a", encoding="utf-8") as f:
            # f.write("[User Entry]\n")
            f.write(admin_text + "\n")
        st.success("Text saved to admin_text.txt")
    st.markdown("---")
    # 2. File Upload
    st.subheader("Upload File (.txt or .pdf)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded.")

        content = ""
        # Handle .txt file
        if uploaded_file.name.endswith(".txt"):
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Preview", content, height=200)
        # Handle .pdf file
        elif uploaded_file.name.endswith(".pdf"):
            with open(os.path.join(DATA_DIR, "temp.pdf"), "wb") as temp_pdf:
                temp_pdf.write(uploaded_file.read())

            reader = PdfReader(os.path.join(DATA_DIR, "temp.pdf"))
            for page in reader.pages:
                content += page.extract_text()
            st.text_area("PDF Preview", content, height=200)

        # Save extracted content to the same text file
        # Save content to file
        with open(FILE_PATH, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        st.success("Content saved to admin_text.txt")

        # Embed and store in ChromaDB
        embedding = get_embedding(content)
        collection.add(
            documents=[content],
            embeddings=[embedding.tolist()],
            ids=[f"id_{len(collection.get()['ids']) + 1}"]
        )
        st.success("Content embedded and stored in ChromaDB.")

        st.success("Content saved to admin_text.txt")

#----------------------------------------USER ------------------------------------------
if role == "User":
    st.set_page_config(page_title="User Dashboard", page_icon="👤", layout="wide")
    st.title("How may I assist you today?")
    st.markdown("""
        <style>
            .stTextInput > div > div > input {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 8px;
            }
            .custom-box {
                background-color: #ffffff;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    text = st.text_input("")
    file_path = "../uploaded_text.txt"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # shape: [seq_len, 768]
        cls_embedding = token_embeddings[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])  # tokenized text

    embedding_list = cls_embedding.tolist()

    # Store each token and its embedding in ChromaDB
    for i, token in enumerate(tokens):
        token_emb = token_embeddings[i].tolist()
        collection.add(
            documents=[token],
            embeddings=[token_emb],
            ids=[f"id_{len(collection.get()['ids']) + 1}"]
        )
    results = collection.get(include=["documents", "embeddings"])

#--------------------------------------------------------------------------------------------------------------------
    tool = Tool_Selection(text)
    st.write(f"Chosen Tool: {tool}")
    if tool == "DNS Lookup":
        dns_info=dns_lookup(text)
        st.write(dns_info)
    elif tool == "CLI":
        cmd=execute_cmd_from_prompt(text)
        st.write(cmd)
    elif tool == "Weather":
        city=extract_city_from_prompt(text)
        Weather=get_weather_summary(city)
        st.write(Weather)
    elif tool == "Web Scraping":
        url=extract_url_from_prompt(text)
        if url:
            rag = RAGSystem(api_key)  # Use same key here
            content=web_scraping(url,rag)
            st.write(content)
    else:
        rag = RAGSystem(api_key)
        rag.process_document("C:/Users/Public/Internship/admin_data/admin_text.txt")
        answer, context = rag.query(text)
        st.write(f"Answer: {answer}")

#--------------------------------------------------------------------------------------
# Example usage
