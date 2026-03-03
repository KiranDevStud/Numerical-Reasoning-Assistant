import streamlit as st
from rag_chain import get_chain, get_fallback_chain, execute_python_code
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Math Reasoning RAG", layout="wide")

st.title("Numerical Reasoning Assistant")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Please set GOOGLE_API_KEY in .env file")
    st.stop()

question = st.text_area("Enter your math problem:")

if st.button("Solve"):
    with st.spinner("Thinking..."):
        chain = get_chain()
        if not chain:
            st.error("Knowledge Base not found. Please run data_ingestion.py first.")
        else:
            try:
                response = chain.invoke(question)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    st.warning("⚠️ Embedding Quota Exceeded. Switching to Direct Solve (No Retrieval)...")
                    fallback_chain = get_fallback_chain()
                    response = fallback_chain.invoke(question)
                else:
                    st.error(f"An error occurred: {e}")
                    st.stop()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reasoning & Code")
                st.markdown(response)
            
            with col2:
                st.subheader("Computed Result")
                # Extract and run code
                result = execute_python_code(response)
                st.success(f"Answer: {result}")
