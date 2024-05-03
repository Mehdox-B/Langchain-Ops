from fastapi import FastAPI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import os
from dotenv import load_dotenv
import uvicorn
from langchain_community.llms import openai
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Api's Documentation for AI Assistant Based on LLM Model 👌 ..."
)

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
# OpenAi LLM Model
openai_model = ChatOpenAI()
# Design the Prompt Template
openai_prompt = ChatPromptTemplate.from_template(
"Write me an essay about {topic} with 100 words"
)

# Google Gemini Pro Model 
google_geminimodel=ChatGoogleGenerativeAI(model="gemini-pro")

# Design the prompt template 
google_geminimodel_prompt=ChatPromptTemplate.from_template('Generate me sitemap for a wordpress website specialized in this business : {business}')

# Building openai api route
add_routes(
    app,
    openai_prompt | openai_model,
    path="/openai"
)
# Building google_gemini route
add_routes(
    app,
    google_geminimodel_prompt | google_geminimodel,
    path="/google_gemini"
)
# Starting Point of The APP                                   
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=9090)


