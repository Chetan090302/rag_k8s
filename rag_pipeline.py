from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = TextLoader("sample.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs = splitter.split_documents(documents)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vector_db = FAISS.from_documents(docs,embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)


class State(TypedDict):
    question: str
    answer: str
    documents: List[str]

def check_status(state: State):
    question = state["question"].lower()

    rag_keywords = [
        "rag",
        "retrieval",
        "vector db",
        "embedding",
        "faiss",
        "document",
        "knowledge base"
    ]

    for keyword in rag_keywords:
        if keyword in question:
            return "rag"

    return "llm"

def call_llm(state: State):
    prompt = f"You are a Generative AI Architect.Answer the following question in a short and crisp manner.Question:{state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}

def call_rag(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    documents = [doc.page_content for doc in retrieved_docs]
    return {"documents": documents}

def generate_content(state: State):

    context = "\n\n".join(state["documents"])
    prompt = f"""
You are a Generative AI Architect.

Use the provided context to answer the question.

Context:
{context}

Question:
{state['question']}

Answer briefly and accurately.
"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph = StateGraph(State)
graph.add_node("llm_node",call_llm)
graph.add_node("rag_node",call_rag)
graph.add_node("generate_node",generate_content)
graph.add_conditional_edges(
    START,
    check_status,
    {
        "rag": "rag_node",
        "llm": "llm_node"
    }
)
graph.add_edge("rag_node","generate_node")
graph.add_edge("generate_node",END)
graph.add_edge("llm_node",END)
app = graph.compile()

def trigger_graph(data:dict):
    result=app.invoke(data)
    return result["answer"]