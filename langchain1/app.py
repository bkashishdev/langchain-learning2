# from dotenv import load_dotenv
# load_dotenv()
# import os
# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import ollama
# from langchain_core.output_parsers import StrOutputParser
# prompts=ChatPromptTemplate.from_messages(
#     [
#         ("system","you are a helpful assistant. Please respond to question asked in funny manner")
#         ("user","Question:{question}")


#     ]
# )
# import streamlit as st 
# st.title("Langchain Demo with Gemmma Model")
# input=st.input("ENter your question")
# llm=OllamaLLM(model="gemma:2b")
# stroutputparser=StrOutputParser()
# chain=llm|StrOutputParser
# if input:
#     print(chain.invoke({"question":input}))
from dotenv import load_dotenv
load_dotenv()

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Set API key (optional for Ollama, usually local)
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# 🧠 Create prompt template
prompts = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to questions in a funny manner."),
    ("user", "Question: {question}")
])

# 🎨 Streamlit UI
st.title("Langchain Demo with Gemma Model")
user_input = st.text_input("Enter your question:")

# 🔧 Set up LLM and parser
llm = Ollama(model="gemma:2b")
stroutputparser = StrOutputParser()

# 🔗 Combine into a chain
chain = prompts | llm | stroutputparser

# 🚀 Run when input is given
if user_input:
    result = chain.invoke({"question": user_input})
    st.write(result)

    