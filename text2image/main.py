## Import the necessary Libraries and Packages
import os 
from dotenv import load_dotenv
import streamlit as st 
from langchain_openai.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import initialize_agent, create_openai_tools_agent,load_tools
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.agents import create_openapi_agent
# from langserve import add_routes
# from fastapi import FastAPI
# import uvicorn

## Load all Env variables
load_dotenv()

## LLM Used to think Intelligently
llm_model = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.3)

## Ai Model Used as toolkit to generate images
##text_2_image_tool = DallEAPIWrapper(model='dall-e-3',size="1024x1024",quality='Standard')
text_2_image_tool = load_tools(["dalle-image-generator"])
## Design Specific Prompt That Make Model More Efficient
text_2_image_prompt = ChatPromptTemplate.from_template(
    """ 
    Your are a powerfull ai assistant in prompt generation.
    Based on the description and the filters provided give me a specific prompt to generate good quality images.
    description:{description}
    filter:{filters}
    """
)
text_2_image_chain = LLMChain(name="image_generation_llm_chain",
                              llm=llm_model,
                              prompt=text_2_image_prompt,
                              verbose=True)
## Streamlit UI part
st.title("Text to image generation plugin")
input_image_description = st.text_input("Let me know a short description of your desired image ...")
## Insert a filter part to let the prompt more specific
filter_option = st.multiselect(
    "Add one or more filters you want to apply ...",
    ["High Resolution","vintage","Realistic","4k","Vibrant colors and gradients","anime"]
)
if input_image_description and filter_option:
    url_image_generated = DallEAPIWrapper(model="dall-e-3",quality="standard").run(text_2_image_chain.run(description=input_image_description,filters=filter_option))
    st.image(url_image_generated)
##---------------------------------------------------------------##
## Instantiante the large language model 
# text_to_image_llm_model = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.2)

# ## Design appropriate template 
# text_to_image_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system","Your are a powerfull ai assistant in prompt generation.Generate me a good prompt based on the following description."),
#         ("user","{description}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad")
#     ]
# )
# ## Create list of tools to be used 
# text_to_image_tools = load_tools(["dalle-image-generator"])

# ## Create a custom agent to be used 
# text_to_image_agent = create_openai_tools_agent(llm=text_to_image_llm_model,tools=text_to_image_tools,prompt=text_to_image_prompt)

# ## text to image Agent Execution
# text_to_image_agent_runnable = AgentExecutor(name="AI Assistant In Image Generation",agent=text_to_image_agent,tools=text_to_image_tools,verbose=True)
# ## Streamlit UI Part 
# st.title("Text to image generation plugin")
# st.write(text_to_image_agent_runnable.invoke({"description":"realistic white horse"}))

    

