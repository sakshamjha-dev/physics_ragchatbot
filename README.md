
# Physics RAG Tutor

This is a physics rag chatbot made using langchain framework, HuggingFace all-MiniLM-L6-v2 embedding model and Groq llama-3.1-8b-instant chatmodel. It is trained on the university physics book.Ask this chatbot anything from University physisc book volume 1 , 2 & 3 and it will provide the answers from the textbook.



## Tech Stack

| Component | Tool |
| :---: | :---: |
| Framework | Langchain |   
| Doc loader | PyMuPDF |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector DB  | FAISS    |
| LLM       | Groq llama-3.1-8b-instant   |
| Deployment | Streamlit Cloud |
---

## Demo

Try the chatbot using this link  https://physicsragchatbot.streamlit.app/

You need to have a free Groq API key to use the app.
You can get one from https://console.groq.com/keys



## How it works
1. PDFs are loaded and the split into 2000 character chunks
2. Each chunk is embedded using Hf all-MiniLM-L6-v2
3. Vectors are saved to FAISS index on disk
4. If the user asks a query then the query is also embedded and similarity search operation is done to find relevant chunks
5. Groq llm generates answer using only the relevant chunks.
---





## Run locally

### Prerequisites
- python 3.11
- Git

### Clone the repo
```bash
  git clone https://github.com/sakshamjha-dev/physics_ragchatbot.git

  cd physics_ragchatbot
```
### Create virtual environment

```bash
  python -m venv venv
  #Windows
  venv\Scripts\activate
```
### Install dependencies

```bash
  pip install -r requirements.txt
```
### Create .env file
Rename .env.example to .env and add your api keys
### Add PDFs
Download the University physics volume 1 , 2 & 3 and drop it into the 'pdfs/' folder
### Run
```bash
streamlit run app.py
```
---


