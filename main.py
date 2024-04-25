import json
import os

import requests as rq

from fastapi import FastAPI, Request, Response, status
from dotenv import load_dotenv


from models.chatbot_query import ChatbotQueryInput, ChatbotQueryOutput

from chains.chatbot_query_chain import chatbot_executor
from utils.sender_utils import send_message

load_dotenv()

VERIFY_TOKEN = os.getenv('FB_VERIFY_TOKEN')
BASE_URL = os.getenv('BASE_URL')
PAGE_ID = os.getenv('PAGE_ID')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')


app = FastAPI(
    title="Customer Support Chatbot",
    description="Endpoints for a customer support chatbot",
)


def invoke_llm_with_retry(query: str, session_id: str):
    return chatbot_executor(query, session_id)


@app.get("/")
def get_status() -> dict:
    return {"status": "running"}


@app.post("/chatbot")
def query_chatbot(
        query: ChatbotQueryInput,
) -> dict:
    query_response = invoke_llm_with_retry(query.text, query.session_id)
    return query_response


@app.get("/webhook", status_code=status.HTTP_200_OK)
async def verify_messenger(request: Request):
    fb_token = request.query_params.get("hub.verify_token")
    print(request.query_params)

    if fb_token == VERIFY_TOKEN:
        return Response(content=request.query_params["hub.challenge"])
    return 'Failed to verify token'


@app.post("/webhook", status_code=status.HTTP_200_OK)
async def query_chatbot_messenger(request: Request):
    data = await request.body()
    data_dict = json.loads(data.decode())
    if data_dict["entry"][0]["messaging"][0].get("message"):
        message = data_dict["entry"][0]["messaging"][0]["message"]["text"]
        sender_id = data_dict["entry"][0]["messaging"][0]["sender"]["id"]
        query_response = invoke_llm_with_retry(message, sender_id)
        message = query_response["output"]

        send_message_url = f"{BASE_URL}{PAGE_ID}/messages?recipient={{id:{sender_id}}}&message={{text:'{message}'}}&messaging_type=RESPONSE&access_token={ACCESS_TOKEN}"
        rq.post(send_message_url)

        return True

    return True
