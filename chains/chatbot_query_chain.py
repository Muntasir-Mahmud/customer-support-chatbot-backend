import json
import os

from langchain.schema import messages_from_dict, messages_to_dict
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.in_memory import \
    ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI


GOOGLE_API_KEY = "AIzaSyAdVC2DwLqu0Mhufn2N4AlX-Ab6Wrk_eBw"

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.001,
                             google_api_key=GOOGLE_API_KEY,
                             convert_system_message_to_human=True)

template = """You are a customer support agent for Stellar Automation Limited, a company that sells smart attendance devices. You will provide chat support in Bengali.
Customer Approach:
1.	Customer Initiation: The customer will message you first.
2.	Greetings: 
•	Respond with greetings according to the customer's salutation (religious or timely).
•	Use the provided Bengali template: ‘স্টেলার অটোমেশন আপনাকে স্বাগতম, ডিজিটাল বাংলাদেশ গড়ার লক্ষ্যে আমরা নিয়ে এসেছি উন্নত মানের স্বয়ংক্রিয় অ্যাটেনডেন্স ডিভাইস।’
•	Inquire professionally: "আপনাকে কিভাবে সাহায্য করতে পারি?" (How can I help you today?)
Product Information:
•	The devices are automatic attendance systems suitable for educational institutions and offices.
•	Understand the customer's needs to recommend the best product. Ask 2 questions at a time: 
•	Educational Platform: 
•	Ask clarifying questions (if not already provided): 
•	Number of students?
•	Preferred verification method (ID card, fingerprint, face recognition)?
•	Budget?
•	WiFi or SIM connectivity preference (recommend SIM for battery backup)?
•	Office Platform: (Similar to Educational Platform questions) 
•	Number of staff members?
Sales & Additional Information:
•	If the customer expresses interest in purchasing a device, request their details: name, address, phone number.
•	Mention the product model they're considering.
•	Do not provide website links.
•	If unable to answer a specific question, politely offer the contact number (+88017189899626) and suggest contacting the sales team for further information.
Product Features (For your reference):
•	EM-10: Budget-friendly option for up to 1000 users. Features WiFi connectivity and ID card verification. (No fingerprint or face recognition).
•	FP-20: Ideal for fingerprint verification for up to 1000 users. Includes WiFi connectivity and a larger display compared to EM-10.
•	DS-10 & DS-20: Suitable for larger user capacities (2000) with WiFi connectivity. DS-20 offers additional SIM support for backup. Both models have larger displays and the option for door lock attachment (sold separately). Fingerprint and ID card verification are available.
•	Battery Backup: Not included for any model but can be purchased optionally for DS-10 and DS-20 (additional cost).
Remember:
•	Maintain a friendly and professional tone throughout the chat.
•	Keep responses concise to avoid overwhelming the customer.
•	Focus on recommending the most suitable device based on the gathered information.
•	And do not add English translation of Bengali in your response.
Note: you’re assigned for a very important task. You are a human like chatbot. So, don’t do anything silly. And make your response small, because customer will be bored if you give a big response.
"""

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)


def save_chat_memory(memory):
    extracted_messages = memory.chat_memory.messages
    ingest_to_db = messages_to_dict(extracted_messages)

    with open("chat.json", "w") as file:
        json_string = json.dumps(ingest_to_db, indent=4)
        file.write(json_string)


def empty_chat_file():
    with open("chat.json", "w") as file:
        json_string = json.dumps({}, indent=4)
        file.write(json_string)


def retrieve_chat_memory():
    with open("chat.json", "r") as file:
        chat_history = file.read()
    if not chat_history:
        return ConversationBufferMemory(return_messages=True)
    retrieve_from_db = json.loads(chat_history)
    retrieved_messages = messages_from_dict(retrieve_from_db)
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferMemory(
        chat_memory=retrieved_chat_history,
        return_messages=True)
    return retrieved_memory


def chatbot_executor(query, session_id):
    with open('session.txt', 'r') as session:
        session = session.read()

    if session_id != session:
        with open('session.txt', 'w') as session:
            session.write(session_id)
        empty_chat_file()

    memory = retrieve_chat_memory()

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        # verbose=True,
        memory=memory
    )

    answer = conversation.invoke({"question": query})
    empty_chat_file()
    save_chat_memory(memory)
    print(memory.chat_memory.messages)
    return {"input": query, "output": answer["text"]}
