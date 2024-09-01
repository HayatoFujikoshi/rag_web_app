from langchain.chains import RetrievalQA
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import PromptTemplate

def load_db(embeddings):
    return FAISS.load_local('faiss_store', embeddings, allow_dangerous_deserialization=True)


def init_page():
    st.set_page_config(
        page_title='Geminiを活用したRAGアプリケーション',
        page_icon="🧑‍💻"
    )
    st.header('ラララたかひらについて聞いてみよう')


def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    db = load_db(embeddings)
    init_page()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_retries=2,
    )

    # オリジナルのSystem Instructionを定義する
    prompt_template = """
    あなたは、ラララたかひらに詳しいチャットボットです。

    ラララたかひらに関する質問に答えてください。ラララたかひらに関係のない質問には、「ラララたかひらに関係することについて聞いてください」と答えてください。

    以下の背景情報を参照してください。背景情報にない情報を勝手に作成して噓をつかないでください
    # 背景情報
    {context}

    #質問
    {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}# システムプロンプトを追加
    )

    if "messages" not in st.session_state:
      st.session_state.messages = []
    if user_input := st.chat_input('質問しよう！'):
        # 以前のチャットログを表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        print(user_input)
        with st.chat_message('user'):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message('assistant'):
            with st.spinner('Gemini is typing ...'):
                response = qa.invoke(user_input)
            st.markdown(response['result'])
            st.markdown('---')
            st.markdown('**ソース**')
            st.markdown(response['source_documents'])
            st.markdown('---')
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})


if __name__ == "__main__":
  main()