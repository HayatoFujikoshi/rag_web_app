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
        model="gemini-1.5-flash",
        temperature=0.0,
        max_retries=2,
    )

    # オリジナルのSystem Instructionを定義する
    prompt_template = """
    あなたは、ラララたかひらに詳しいチャットボットです。

    ラララたかひらに関する質問に、背景情報を参考に答えてください。
    ラララたかひらに全然関係のない質問には、「ラララたかひらに関係することについて聞いてください」とのみ答えてください。
    
    また、最後に質問の内容をみて、以下のルールに従って参考urlを出してください。
    ラララたかひら全般に関することであれば、
    「ホームページはこちらをご覧ください。　https://lalala-takahira.github.io/homepage/　」と最後に答えてください。
    イベントの情報の情報についての質問なら
    「イベントはこちらをご覧ください。　https://lalala-takahira.github.io/homepage/events」と最後に答えてください。
    過去活動について聞かれたら、
    「過去の活動はこちらをご覧ください。　https://lalala-takahira.github.io/homepage/reports」と最後に答えてください。
    ラララたかひらについて詳しくし知りたそうな質問には、
    「ラララたかひらについては詳しく知りたい方はこちらをご覧ください。　https://lalala-takahira.github.io/homepage/about　」と最後に答えてください。
    メディア掲載に聞かれたら
    「こちらをご覧ください。　https://lalala-takahira.github.io/homepage/media」と最後に答えてください。
    さんだまち博について聞かれたら
    「さんだまち博については、こちらをご覧ください。　https://sanda-machihaku.jp/p-2024-28/　」と最後に答えてください。

    
    質問の回答には以下の背景情報を参照してください。背景情報にない情報を勝手に作成して噓をつかないでください
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
    # 入力文字数の制限を設定
    max_length = 100
    
    # ユーザーの入力
    user_input = st.chat_input('質問しよう！')

    # 入力がある場合の処理
    if user_input:
        char_count = len(user_input)

        # 100文字を超えた場合の警告
        if char_count > max_length:
            st.warning(f'入力は{max_length}文字以内にしてください。現在の文字数: {char_count}')
        else:
            # 以前のチャットログを表示
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            with st.chat_message('user'):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message('assistant'):
                with st.spinner('回答を取得中...'):
                    response = qa.invoke(user_input)
                st.markdown(response['result'])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})

if __name__ == "__main__":
    main()