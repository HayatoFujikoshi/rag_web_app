#ラララたかひらについての質問に答えるチャットボットを作成しています。
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
        page_title='ラララたかひらチャットボット',
        page_icon="🧑‍💻"
    )
    st.header('ラララたかひらについて聞いてみよう')
    st.markdown(
    '<p style="font-size:12px;">こちらのチャットボットはラララたかひらの団体について答えますが回答は必ずしも正しいとは限りません。詳しくは<a href="https://lalala-takahira.github.io/homepage/" target="_blank">公式ホームページ</a>、<a href="https://www.instagram.com/lalala_takahira/" target="_blank">インスタグラム</a>をご覧ください。</p>',
    unsafe_allow_html=True
    )


def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    db = load_db(embeddings)
    init_page()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        max_retries=5,
    )

    # オリジナルのSystem Instructionを定義する
    prompt_template = """
    あなたは、「ラララたかひら」という団体のリーダーです。
    以下の「背景情報」を参考に、質問に対して団体の人間になりきって、質問に回答してくだい。

    ラララたかひらに全く関係のない質問と思われる質問に関しては、「すみませんが、ラララたかひらに関係することについて聞いてください」と答えてください。

    以下の背景情報を参照してください。情報がなければ、「分からないです。詳しくはホームページをご覧くださいと答えてください」
    # 背景情報
    {context}

    # 質問
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

    # 質問テンプレートのリスト
    question_templates = [
        "ラララたかひらはどんな活動をしていますか？",
        "過去の活動について教えてください。",
        "ラララたかひらに参加するにはどうすればいいですか？",
        "ラララたかひらの活動場所はどこですか？",
        "ラララたかひらはどんなメンバーがいますか？",
        "ラララたかひらの活動目的は？",
    ]

    # ユーザーが選択できる質問テンプレート
    selected_question = st.selectbox("質問テンプレート", ["（テンプレートを使用しない）"] + question_templates)

    # 入力文字数の制限を設定
    max_length = 100

    # ユーザーの入力
    user_input = st.chat_input("質問しよう！")

    # テンプレートを選択した場合、その質問を user_input に設定
    if selected_question != "（テンプレートを使用しない）" and not user_input:
        user_input = selected_question

    # 入力がある場合の処理
    if user_input:
        char_count = len(user_input)

        # 100文字を超えた場合の警告
        if char_count > max_length:
            st.warning(f"入力は{max_length}文字以内にしてください。現在の文字数: {char_count}")
        else:
            # 以前のチャットログを表示
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("考え中..."):
                    response = qa.invoke(user_input)
                st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

if __name__ == "__main__":
    main()