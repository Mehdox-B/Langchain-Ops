import requests
import streamlit as st

def get_openai_response(input_text):
    response=requests.post("http://localhost:9090/openai/invoke",
    json={'input':{'topic':input_text}})

    
    return response.json()['output']['content']

st.title('Client Side API Consumer UI 🤷‍♂️ ...')
input_text=st.text_input('Ask me question about a specific topic ...',placeholder='What topic you want to know about ...')

if input_text:
    st.write(get_openai_response(input_text))


