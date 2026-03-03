import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
# from langchain.retrievers.multi_query import MultiQueryRetriever (Removed due to import issues)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv

load_dotenv()

# Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIRECTORY = r"chroma_db"
COLLECTION_NAME = "math_reasoning"

def get_retriever():
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Using standard retriever for now due to environment issues with MultiQueryRetriever
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_chain():
    retriever = get_retriever()
    if not retriever:
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    template = """You are a Mathematical Reasoning Agent. Use the following context (similar examples/formulas) to solve the problem.
    Context: {context}
    
    Problem: {question}
    
    Instructions:
    1. Break down the problem step-by-step.
    2. Write a Python script to calculate the final answer. Assign the final answer to a variable named `result`.
    3. The script must be self-contained and import any necessary modules (like math).
    4. Output the reasoning, then the code within ```python ... ``` block.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def get_fallback_chain():
    """Returns a chain that skips retrieval (for when quota is exceeded)."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    template = """You are a Mathematical Reasoning Agent. 
    Problem: {question}
    
    Instructions:
    1. Break down the problem step-by-step.
    2. Write a Python script to calculate the final answer. Assign the final answer to a variable named `result`.
    3. The script must be self-contained and import any necessary modules (like math).
    4. Output the reasoning, then the code within ```python ... ``` block.
    """
    
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def execute_python_code(code_str):
    repl = PythonREPL()
    try:
        # extract code block
        if "```python" in code_str:
            code = code_str.split("```python")[1].split("```")[0]
        elif "```" in code_str:
             code = code_str.split("```")[1].split("```")[0]
        else:
            code = code_str
        
        # We need to print the result to capture it in stdout if it's not printed
        if "print(result)" not in code:
             code += "\nprint(result)"
             
        output = repl.run(code)
        return output.strip()
    except Exception as e:
        return f"Error executing code: {e}"
