from docx import Document
from langchain_community.document_loaders import Docx2txtLoader

from chains.chatbot_query_chain import initialize_rag_chain


def update_training_data(text: str):
    document = Document("resources/chatbot-resource.docx")
    document.add_paragraph(text)
    document.save('resources/chatbot-resource.docx')
    initialize_rag_chain()

    data = load_training_data()
    
    return data


def load_training_data():
    loader = Docx2txtLoader("resources/chatbot-resource.docx")
    data = loader.load_and_split()
    return {"data": data[0].page_content}
