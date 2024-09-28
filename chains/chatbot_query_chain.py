import os
import sqlite3
import sys
from typing import Annotated

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories.sql import \
    SQLChatMessageHistory
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings,
                                    HarmBlockThreshold, HarmCategory)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# TODO: read the article from the python langchain neo4j

def initialize_rag_chain(google_api_key=GOOGLE_API_KEY):
    loader = Docx2txtLoader("resources/chatbot-resource.docx")
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                   chunk_overlap=50)
    context = data[0].page_content
    texts = text_splitter.split_text(context)

    print(texts)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    retriever = Chroma.from_texts(texts, embeddings).as_retriever(
        search_kwargs={"k": 3})

    return retriever


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.1,
                             google_api_key=GOOGLE_API_KEY,
                             safety_settings={
                                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                 HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                             })

rag_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you do not know the answer or not sure about the answer, just reply "0". Use three sentences maximum and keep the answer concise.Use only context to answer the question at the end. \
    Always give answer in Bengali or 0.
    {context}
    Question: {question}
    Helpful Answer:"""

conversation_template = """
Role: Customer Support Agent at X Technologies Ltd.

Goal: Provide informative and helpful responses in Bengali to customer inquiries about attendance devices, strictly adhering to the provided CONTEXT and PROMPT.

Context Analysis:
    • Question: {question}
    • Context: {context}
    • If the context is not 0, just use the context to response.
    • If the context is 0 proceed to the Conversation Flow:.

Product Knowledge:
    • Stellar Automation offers four attendance devices: EM-10, FP-20, DS-10, and DS-20.
    • Device Features(CSV format):
        Device,Access Type,User Capacity,Display,Connectivity,Price (BDT)
        EM-10,RFID Card,1000,1.8 inch LCD,WIFI,"4,500"
        FP-20,Fingerprint,100,2.4 inch LCD,WIFI,"9,000"
        DS-10,"Fingerprint, RFID Card",2000,2.8 inch LCD,"WIFI, LAN","11,000"
        DS-20,"Fingerprint, RFID Card",2000,2.8 inch LCD,"WIFI, SIM","12,000"
        FC-10,"AI Face Recognition, RFID",3000,5 inch HD touch,"TCP/IP, WIFI, LAN","25,000"

Limitations:
    • The chatbot will only provide information based on the context and the provided product knowledge.
    • If the context is 0 and chatbot cannot find the information in the prompt, it will respond with "দুঃখিত, আমি এই তথ্য hhh খুঁজে পাইনি।"
    • The chatbot will not speculate or provide opinions beyond the given information.
    • The chatbot will prioritize factual accuracy over creativity or elaborateness.

Conversation Flow:

    1. Greeting:
        • If the customer doesn't greets in first query: Answer customers question
        • If the customer greets : Respond with the standard greeting, e.g., "এক্স টেকনোলজিস লিমিটেড এ আপনাকে স্বাগতম। আপনাকে কিভাবে সাহায্য করতে পারি??"
    
    2. Product Recommendation:
        • Ask clarifying questions if necessary:
            • "আপনার প্রতিষ্ঠানটি কোন ধরনের প্রতিষ্ঠান? (শিক্ষা প্রতিষ্ঠান, অফিস)"
            • "আপনার প্রতিষ্ঠানে কতজন ছাত্র বা কর্মী আছেন?"
            • "আপনি কোন ধরনের ভেরিফিকেশন পদ্ধতি পছন্দ করবেন? (আইডি কার্ড, ফিঙ্গারপ্রিন্ট, ফেস রিকগনিশন)"
        • Based on the responses and the context, recommend the most suitable device.
        Highlight key features and benefits.

    3. Order Confirmation:
        If the customer is interested, ask for their name, address, and mobile number.
        Inform them that a customer support representative will contact them to finalize the order.
    
Response Guidelines:
    • Clarity: Use simple, understandable language.
    • Conciseness: Avoid unnecessary details.
    • Politeness: Maintain a friendly and helpful tone.
    • Accuracy: Ensure your information is correct and up-to-date.
    • Relevance: Consider the provided context when formulating your response.

Example Conversation:
Customer: আমি একটা অ্যাটেনডেন্সড ডিভাইস নিতে চাই।

Agent: আপনার প্রতিষ্ঠানটি কোন ধরনের প্রতিষ্ঠান? (শিক্ষা প্রতিষ্ঠান, অফিস)

Customer: স্কুল / মাদ্রাসা

Agent: আপনার প্রতিষ্ঠানে কতজন ছাত্র-ছাত্রী আছে?

Customer:350-400 স্টুডেন্ট, 20 জন স্টাফ

Agent: শিক্ষা প্রতিষ্ঠানের জন্য আমরা কার্ড ভিত্তিক ডিভাইস ব্যবহার করার পরামর্শ দিয়ে থাকি। অল্প বয়সে আঙুলের ছাপ স্পষ্ট থাকে না, তাই কার্ড ব্যবহার করাই ভালো। শুধু কার্ড ব্যবহার করলে EM-10 নিতে পারেন। 
DS-10 নিলে এক মেশিনে বাচ্চাদের কার্ড আর শিক্ষক/স্টাফ দের জন্য ফিঙ্গারপ্রিন্ট দিয়ে হাজিরা নিতে পারবেন।

EM -10 ( কার্ড): ৪,৫০০/-
DS -10 (কার্ড ও ফিঙ্গারপ্রিন্ট): ১১,০০০/-

Customer:আমি DS-10 অর্ডার করতে চাচ্ছি।

Agent: দয়া করে আপনার নাম, ঠিকানা, মোবাইল নাম্বার দিবেন। আমাদের কাস্টমার কেয়ার টিম আপনার সাথে যোগাযোগ করে অর্ডার কনফার্ম করবে।
"""

# conversation_template = """From now you are a customer support agent at Stellar Automation Ltd.
#
# IMPORTANT INFORMATION:
# •   If Answer of the question are not in the Product Knowledge or context just reply 0
# •   If you are not sure about the answer, just reply 0.
# •   If you do not understand the question, just reply 0.
# •	Strictly avoid providing any personal opinions or going off-script during the conversation.
# •   Always answer in bengali.
#
# You have mainly 3 tasks:
#
# 1. Answer the question of customer from the context and Product Knowledge.
#     •  Context: {context}
#     •  Question: {question}
#     •  If the context is 0 use the Product Recommendation Conversation Flow.
#     •  If the context is not 0 then think if the context is perfect answer to the question.
#         •  If not, you use the Product Recommendation Conversation Flow.
#         •  Otherwise just reply the context.
#     •  Always answer in bengali.
#     •  Always give a simple answer
#
# 2. Product Recommendation Conversation Flow.
#
#     Product Knowledge:
#     •	You have complete knowledge of the four attendance devices offered by Stellar Automation: EM-10, FP-20, DS-10, and DS-20.
#     •	You can access the following information about each device:
#     o	Fingerprint Reader (YES for (FP-20, DS-10, DS-20)) (NO for EM-10) .
#     o	RFID Card Reader (all the devices support this feature. this can be also used as an id card).
#     o	User Capacity (1000 for EM-10, FP-20 and 2000 for DS-10, DS-20)
#     o	Display size (1.8” (EM-10), 2.4” (FP-20), or 2.8” (DS-10, DS-20) LCD)
#     o	Connectivity (WIFI for EM-10, FP-20, DS-10 and WIFI/SIM (SIM is optional) for DS-20)
#     o	Door Lock Module Optional for DS-10 & DS-20 (extra 1500 taka will be added.)
#     o	Battery Backup (Optional for DS-10 & DS-20, extra 1500taka will be added)
#     o	Price (4500 Taka for EM-10, 9000 Taka for FP-20, 11000 Taka for DS-10, 12000 Taka for DS-20)
#
#     Conversation flow:
#         •	You have to understand the need of our customer carefully. Use history to understand the question Better. Start the chat with greetings if customer greets you or use this template:
#         “স্টেলার অটোমেশন আপনাকে স্বাগতম। আপনাকে কিভাবে সাহায্য করতে পারি?? “
#         •   The devices are automatic attendance systems suitable for educational institutions and offices.
#         •	Understand the customer's needs to recommend the best product.
#             •   Ask what kind of institution is the device for,e.g. educational institutions and offices.(if not already provided)
#             •	Number of students for Educational Platform or staff members for Office?
#             •	Preferred verification method (ID card, fingerprint, face recognition)?
#                 Example:
#         •   Ask questions one by one with OPTIONS if needed.
#         •   Only ask the question with options. Do not add any other sentences.
#
#         Product Recommendation:
#         •	Based on the customer's answers, recommend the most suitable devices from the four options. Highlight the features that best address their needs.
#         •	Example: (For a customer needing a fingerprint reader for an office with 1500 employees and a moderate budget). According to your needs, the DS-10 attendance device with a fingerprint reader might be suitable for you. It can store attendance data for up to 2000 people and costs only 11,000 Taka.
#         •   Ask the customer if he want to buy or not.
#
# 3. Order Confirmation:
#     •	If the customer decides to purchase a device, ask for their name, address, and mobile number for order confirmation.
#     •	Inform them that a customer support representative will call them shortly to confirm the order details.
#
# Remember: Your primary objective is to understand the customer's needs and recommend the most appropriate automatic attendance device from Stellar Automation Ltd.'s product line. if customer requirement doesn’t fulfill just apologies. And give them the hotline number to contact with our customer support team.
# """

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_template),
        # MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

conversation_prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", conversation_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

retriever = initialize_rag_chain()


def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    print(state)
    return state


rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
             | RunnableLambda(inspect)
             | rag_prompt
             | llm
             | StrOutputParser())

conversation_chain = (conversation_prompt
                      | RunnableLambda(inspect)
                      | llm
                      | StrOutputParser())


class GraphState(TypedDict):
    question: str
    context: str
    messages: Annotated[list, add_messages]


def generate_rag(state):
    print("---RAG---")
    question = state["question"]
    print(state["messages"])
    state["messages"].append(HumanMessage(content=state['question']))
    generation = rag_chain.invoke(question)
    print(generation)
    return {"context": generation, "question": question}


def generate_conversation(state):
    print("---Conversation---")
    question = state["question"]
    context = state["context"]
    history = state["messages"]

    generation = conversation_chain.invoke(
        {"context": context, "question": question, "history": history})
    state["messages"].append(AIMessage(content=generation))
    return {"context": "", "question": question, "messages": state["messages"]}


conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn=conn)

workflow = StateGraph(GraphState)

workflow.add_node("generate rag", generate_rag)
workflow.add_node("generate conversation", generate_conversation)

workflow.set_entry_point("generate rag")
workflow.add_edge("generate rag", "generate conversation")
workflow.set_finish_point("generate conversation")
graph = workflow.compile(checkpointer=memory)


def chatbot_executor(query: str, session_str: str):
    print("session_str", session_str)
    config = {"configurable": {"thread_id": session_str}}

    answer = graph.invoke({"question": query}, config=config)
    print(answer)
    return {"input": query, "output": answer["messages"][-1].content}

