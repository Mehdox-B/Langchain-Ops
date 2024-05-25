# Import all the required Library 
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import openai
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import time
from langchain_core.output_parsers import StrOutputParser

# Load env variables 
load_dotenv()

os.environ['OPEANAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

# Initialization of stream session state , so we can maintaining app configuration across reruns
if "vector" not in st.session_state:
    st.session_state.embeddings=OpenAIEmbeddings()
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    st.session_state.final_splitted_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_splitted_documents,st.session_state.embeddings)

st.title("ChatGroq By LPU Inference Engine 🔥")
llm=ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context between <context></context> tags only.
    Please think step by step before you answer the question.
    Try to answer with the most accurate answer.
    <context>
    {context}
    </context>
    Question:{input}
   """
)
stuffed_document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt,output_parser=StrOutputParser())
retriever=st.session_state.vectors.as_retriever()
retriever_chain=create_retrieval_chain(retriever,stuffed_document_chain)

# Define The input
input_question=st.text_input(label="You can put you question is here ...")

if input_question:
    start=time.process_time()
    generated_response=retriever_chain.invoke({"input":input_question})
    st.write(generated_response['answer'])
    reponse_time=time.process_time()-start
    st.write("The Reponse Time is : {:.4f} Second".format(reponse_time))
    
    # Find the most relevant Chunks
    with st.expander("Document Similarity Search"):
        for index,doc in enumerate(generated_response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------------------")
