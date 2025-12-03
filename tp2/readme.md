# RAG (Retrieval-Augmented Generation) for Document Q&A

Click here to watch the demo video on YouTube:

[![Watch the video](https://img.youtube.com/vi/513GaoDAdbI/maxresdefault.jpg)](https://www.youtube.com/watch?v=513GaoDAdbI)

# Preparation

### Install Dependencies

```sh
pip install -r requirements.txt
```
>NOTE: The system was developed and tested with Python 3.11 and Ubuntu 22.04.5 LTS.

### Setup Pinecone Index with Document Embeddings

```sh
export PINECONE_API_KEY="your_pinecone_api_key"
```

Run the database builder to create embeddings and populate the Pinecone index:

```sh
python db_builder.py -d ./dataset/ -m jinaai/jina-embeddings-v2-small-en -n cv-embeddings -v
```
>NOTE: The only supported model for embeddings is `jinaai/jina-embeddings-v2-small-en`. Its being used in the chatbot for document retrieval and must be also used here to create the embeddings. 

## Run the Q&A Application

Export the necessary API keys:

```sh
export PINECONE_API_KEY="your_pinecone_api_key"
export GROQ_API_KEY="your_groq_api_key"
```

Run the app:

```sh
streamlit run streamlit_chatbot.py
```

Go to `http://localhost:8501` in your web browser to interact with the chatbot. You will see a interface like this:

![Chatbot Interface](assets/thumb.png)

