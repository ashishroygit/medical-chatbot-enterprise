from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from pathlib import Path
import os
import traceback
from flask import Response

app = Flask(__name__)

# --- load env reliably (from .env next to app.py, regardless of CWD) ---
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # fallback to default search

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val or not val.strip():
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your OS or in a .env file next to app.py."
        )
    return val.strip()

# --- validate required keys (fail fast with clear message) ---
try:
    PINECONE_API_KEY = require_env("PINECONE_API_KEY")
    OPENAI_API_KEY   = require_env("OPENAI_API_KEY")
except RuntimeError as e:
    # Show a clear message in the console and a simple page in the browser
    @app.route("/")
    def missing_keys():
        return f"<pre>{e}</pre>", 500
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8080, debug=True)
    raise

# If libraries read from os.environ, it's safe to set now (non-None)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- embeddings + vector store ---
embedding = download_embeddings()  # must return a valid LangChain Embeddings instance

index_name = "medicalai-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# Use a valid OpenAI model name unless you’ve changed base_url/provider
chatModel = ChatOpenAI(model="gpt-4o-mini")  # e.g., cheap/fast; adjust as needed

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])  # POST only (your JS posts)
def chat():
    user_msg = request.form.get("msg", "").strip()
    if not user_msg:
        return "Please type a message.", 400
    try:
        result = rag_chain.invoke({"input": user_msg})
        # Inspect what keys come back:
        # print("RAG keys:", list(result.keys()))
        return str(result.get("answer", "")).strip()
    except Exception as e:
        traceback.print_exc()  # prints full stack to your console
        # Return the error text to your browser for now (DEV ONLY)
        return Response(f"Error: {e}", status=500, mimetype="text/plain")
    
@app.route("/health", methods=["GET"])
def health():
    # keep it super fast and dependency-free
    return jsonify(status="ok"), 200

if __name__ == "__main__":
    # Note: Flask debug reloader can import the module twice.
    # All init is safe above; if you see double inits, set debug=False or FLASK_ENV=production.
    app.run(host="0.0.0.0", port=8080, debug=True)


