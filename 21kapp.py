import streamlit as st
import os
from groq import Groq
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.title("21K  Learning Chatbot")

with st.expander("ℹ️ Disclaimer"):
    st.caption(
        """This chatbot is designed to guide students through math-related problems, providing step-by-step approach without giving away the answers"""
    )

groq_api_key = os.environ['GROQ_API_KEY']

def main():
    template = """You are an assistant to guide the kids from class 1 to 12th where you can provide guidance to solve
      the problem rather than giving answers. Don't give answers, instead, you give each single step not all steps at a time and display formulas.
    {chat_history}
    Human: {human_input}
    Chatbot:
    """
    
    prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']}, {'output': message['AI']})
    
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
    )
    
    llm_chain = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What's your math question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = llm_chain.predict(human_input=prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        message = {'human': prompt, 'AI': response}
        st.session_state.chat_history.append(message)

if __name__ == "__main__":
    main()
