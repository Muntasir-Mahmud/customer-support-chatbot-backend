from docx import Document
from langchain_community.document_loaders import Docx2txtLoader


def update_training_data(text: str):
    document = Document("chatbot-resource.docx")
    document.add_paragraph(text)
    document.save('chatbot-resource.docx')

    data = load_training_data()
    return data


def load_training_data():
    loader = Docx2txtLoader("chatbot-resource.docx")
    data = loader.load_and_split()
    return data[0].page_content
