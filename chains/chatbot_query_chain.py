from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories.sql import \
    SQLChatMessageHistory
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings,
                                    HarmBlockThreshold, HarmCategory)

GOOGLE_API_KEY = "AIzaSyAdVC2DwLqu0Mhufn2N4AlX-Ab6Wrk_eBw"

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# TODO: add dotenv for Goolge api key
# TODO: read the article from the python langchain neo4j

def initialize_rag_chain(google_api_key=GOOGLE_API_KEY):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest",
                                temperature=0.1,
                                google_api_key=google_api_key,
                                convert_system_message_to_human=True)

    contextualize_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""


    qa_system_prompt = """You are a customer support agent for Stellar Automation Limited, a company that sells smart attendance devices. You will provide chat support in Bengali.
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
    Note: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.
    {context}
    """

    loader = Docx2txtLoader("resources/chatbot-resource.docx")
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    context = data[0].page_content
    texts = text_splitter.split_text(context)

    print(texts)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":1})




    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, vector_index, contextualize_q_prompt
    )



    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


chain = initialize_rag_chain()

def chatbot_executor(query: str, session_str: str):
    """
    Executes the chatbot with the given query and session ID.

    Args:
        query (str): The query to be processed by the chatbot.
        session_str (str): The session ID for the chatbot.

    Returns:
        dict: A dictionary containing the input query and the chatbot's response.
    """
    # Initialize the chatbot with message history
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: SQLChatMessageHistory(
            session_id=session_id, connection_string="sqlite:///sqlite.db"
        ),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Set the configuration for the chatbot
    config = {"configurable": {"session_id": session_str}}

    # Invoke the chatbot with the query and configuration
    answer = chain_with_history.invoke({"input": query}, config=config)

    # Return the input query and the chatbot's response
    return {"input": query, "output": answer["answer"]}
