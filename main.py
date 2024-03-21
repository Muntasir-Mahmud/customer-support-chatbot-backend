from fastapi import FastAPI
from models.chatbot_query import ChatbotQueryInput, ChatbotQueryOutput
from utils.async_utils import async_retry

from chains.chatbot_query_chain import chatbot_executor

app = FastAPI(
    title="Customer Support Chatbot",
    description="Endpoints for a customer support chatbot",
)


def invoke_llm_with_retry(query: str, session_id: str):
    return chatbot_executor(query, session_id)


@app.get("/")
def get_status():
    return {"status": "running"}


@app.post("/chatbot")
def query_chatbot(
        query: ChatbotQueryInput,
) -> ChatbotQueryOutput:
    query_response = invoke_llm_with_retry(query.text, query.session_id)

    return query_response
