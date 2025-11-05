from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

class QnA:
    def __init__(self, file_path, model_name, api_key):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-70b-versatile",
            temperature=0
        )

        self.retriever = self.vectorstore.as_retriever()

        self.doc_chain = create_stuff_documents_chain(self.llm)

        self.chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=self.doc_chain
        )

        self.chat_history = []

    def get_answer(self, user_input):
        system_prompt = (
            "Você é um especialista em Brasil. "
            "Responda apenas com base nos documentos. "
            "Se não souber, diga que não sabe."
        )

        query = f"{system_prompt}\n\nPergunta: {user_input}"

        response = self.chain.invoke({"input": query})
        answer = response["answer"]

        self.chat_history.append((user_input, answer))
        return answer