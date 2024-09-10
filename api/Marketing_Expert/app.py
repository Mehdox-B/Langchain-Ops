from fastapi import FastAPI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import uvicorn
from langchain_community.llms import openai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
load_dotenv()

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="Api's Documentation for AI Assistant Based on LLM Model 🧾 ..."
)

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
# os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
# OpenAi LLM Model

memory = ConversationBufferWindowMemory(memory_key="chat_history")
llm_model = ChatOpenAI(model='gpt-4o',
                          api_key=os.environ['OPENAI_API_KEY'],
                          )

template = """Your are an expert ai assistant chatbot having conversation human in marketing strategy,analyse and think step by step before answering my questions about possible marketing strategies: 
             Take into consideration the following parameters:
             Previous conversation:{chat_history}
             New human question: {user_question}
             """
 # Design the Prompt Templates
prompt = ChatPromptTemplate.from_template(template)

def get_response_from_chain(user_query:str, chat_history:list):

    llm = llm_model
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

prompt_template = prompt
conversation = LLMChain(
    llm=llm_model,
    prompt=prompt_template,
    verbose=True,
    memory=memory
)

# Google Gemini Pro Model 
#google_geminimodel=ChatGoogleGenerativeAI(model="gpt-4o")

# Design the prompt template 
#google_geminimodel_prompt=ChatPromptTemplate.from_template('Generate me sitemap for a wordpress website specialized in this business : {business}')

# Building openai api route
add_routes(
    app,
    conversation ,
    path="/openai"
)
# Starting Point of The APP                                   
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=9090)


