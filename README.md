This project is an intelligent, Streamlit-based assistant that dynamically selects and runs the appropriate tool based on user input. It supports a variety of tasks such as:

- 🌦 Weather Reporting  
- 🌐 DNS Lookup  
- 🕸 Web Scraping  
- 💻 CLI Command Execution  
- 📚 Retrieval-Augmented Generation (RAG) for general questions

## 💡 Features

- **Role-Based Access:** Admins can upload documents (.txt/.pdf) to be used as a knowledge base. Users can ask questions which are answered using the appropriate tool.
- **BERT Embeddings:** All user/admin content is embedded using `bert-base-uncased` and stored in ChromaDB for efficient retrieval.
- **Dynamic Tool Selection:** A Gemini Pro model is used to intelligently choose which tool to trigger based on the user's query.
- **PDF & Text Upload Support** for Admin.
- **Real-time Utility Execution** including weather lookup, DNS analysis, and command execution.


## 🛠 Tools & Technologies

- `Streamlit` — Web Interface  
- `Google Gemini API` — Prompt Understanding  
- `BERT` — Text Embeddings  
- `ChromaDB` — Embedding Storage  
- `PyPDF2` — PDF Parsing  
- Custom tools: `DNS Lookup`, `CLI Executor`, `Weather Tool`, `Web Scraper`, `RAG System`

