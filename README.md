
# RAGを活用したオリジナルチャットボット

このプロジェクトでは、**Retrieval-Augmented Generation**を活用し、特定のドメイン(ラララたかひらという団体）に基づいてユーザーの質問に応答するオリジナルのチャットボットを構築しました。さらに、**Streamlit**を用いてWebアプリケーションとしてデプロイも行いました。
[ここ](https://lalala-takahira-chatbot.streamlit.app/)から、チャットボットを試せます。
この[Qiitaの記事](https://qiita.com/HayatoF/items/cc6477e1f7ab717c4cd6)に詳しい解説を書きました。

## 概要

- **RAG**技術により、質問に対して関連するデータを検索し、その情報を基にAIが自然な応答を生成します。
- **FAISS**を使った高速ベクトル検索で、膨大なデータの中から適切な情報を効率的に取り出します。
- **Google Generative AI**を使って、ユーザーの質問に対してAIが適切な返答を生成します。

## 機能

- ユーザーの質問に応じて関連するデータを検索し、適切な応答を生成。
- 応答に利用されたドキュメントの出典元（URL）を表示。
- FAISSによる高速なベクトル検索が可能。
- AIによる誤った応答（ハルシネーション）のリスクを出典元表示で軽減。

## 工夫点、今後の改善点

- **ハルシネーション**問題に対応するため、AIが生成する応答の出典元を表示し、信頼性を向上させています。
- 今後、ベクトル検索と単語検索を組み合わせたハイブリッド検索を導入やプロンプトの調整によって、さらに精度の高い応答を生成するように改善したいと考えています。



# Original Chatbot Utilizing RAG

In this project, I built an original chatbot based on **Retrieval-Augmented Generation (RAG)**, which responds to user queries related to a specific domain (the organization called "Lalala Takahira"). Additionally, I deployed the chatbot as a web application using **Streamlit**. You can try the chatbot [here](https://lalala-takahira-chatbot.streamlit.app/). A detailed explanation of the project can be found in this [Qiita article](https://qiita.com/HayatoF/items/cc6477e1f7ab717c4cd6).

## Overview

- The **RAG** technology searches for relevant data and generates natural responses from AI based on the retrieved information.
- **FAISS** enables fast vector search, efficiently extracting relevant information from large datasets.
- **Google Generative AI** is used to generate appropriate responses to user queries.

## Features

- Searches for relevant data based on user questions and generates appropriate responses.
- Displays the source URL of documents used to generate the response.
- Supports fast vector searches using FAISS.
- Reduces the risk of **hallucination** (incorrect AI responses) by showing the source of the information.

## Improvements and Future Plans

- To address the **hallucination** problem, the source of the information used by the AI is displayed to improve trustworthiness.
- We plan to introduce a hybrid search approach that combines vector search with keyword search in the future, and by fine-tuning the prompts, we aim to improve the accuracy of the responses.
