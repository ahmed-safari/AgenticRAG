# 📌 UDST Policy Chatbot

The **UDST Policy Chatbot** is an AI-powered system that enables users to query **official university policies** efficiently. It utilizes **retrieval-augmented generation (RAG)** to fetch relevant policy information and generate accurate responses using **Mistral AI**. The chatbot processes, stores, and searches policy data using **FAISS (Facebook AI Similarity Search)** for efficient retrieval.

---

## **📂 Project Structure**
```
📦 UDST-Policy-Chatbot
├── assets
│   ├── combined_documents.txt   # Merged policy documents
│   ├── rag_data.pkl             # Pickled policy embeddings
│   ├── rag_index.faiss          # FAISS vector database
├── app.py                       # Streamlit-based chatbot application
├── notebook.ipynb               # Jupyter notebook for preprocessing & testing
└── README.md                    # Project documentation
```

---

## **🛠 How It Works**
1. **Data Collection**  
   - Policy text is scraped from **official UDST webpages**.
   - All policy documents are **merged into** `combined_documents.txt`.

2. **Text Embedding & Indexing**  
   - The **Mistral AI** model generates **dense vector embeddings** for each text chunk.
   - These embeddings are stored in **`rag_data.pkl`** and indexed in **FAISS (`rag_index.faiss`)** for fast similarity search.

3. **Query Processing**  
   - A user submits a query via **Streamlit (`app.py`)**.
   - The system converts the query into an **embedding** and searches FAISS for the most relevant policy chunks.
   - The retrieved context is passed to **Mistral AI**, which generates a policy-based response.

---

## **🚀 Setup & Installation**
### **🔧 Prerequisites**
Ensure you have **Python 3.8+** and install the required dependencies:

```bash
pip install -r requirements.txt
```

### **📌 Running the Chatbot**
Start the Streamlit application:

```bash
streamlit run app.py
```

### **🔑 Getting your API key**
Head to [**Mistral AI's API Keys Page**](https://console.mistral.ai/api-keys) to genearate a new API key. 

### **📊 Running the Notebook**
To explore data processing or update embeddings, run:

```bash
jupyter notebook notebook.ipynb
```

---

## **📂 Data Files**
- **`combined_documents.txt`** → Full combined policy dataset.
- **`rag_data.pkl`** → Stored **embeddings** for policies.
- **`rag_index.faiss`** → **FAISS index** for efficient retrieval.

---

## **🔍 Key Features**
✅ **Fast Policy Retrieval** – Uses **FAISS vector search** for efficient lookup.  
✅ **Accurate Answers** – **Mistral AI** generates **context-aware** responses.  
✅ **Scalable & Efficient** – Precomputed embeddings **reduce API calls** and **speed up search**.  
✅ **Easy Deployment** – **Streamlit UI** makes querying policies simple and user-friendly.

---

## **🔗 References**
- **Mistral AI**: [https://mistral.ai](https://mistral.ai)  
- **FAISS**: [https://faiss.ai](https://faiss.ai)  
- **Streamlit**: [https://streamlit.io](https://streamlit.io)  

---


🎯 **Now you're ready to explore UDST policies effortlessly!** 🚀
