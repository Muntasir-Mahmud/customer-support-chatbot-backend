import json
import os

import requests as rq
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status

from chains.chatbot_query_chain import chatbot_executor
from models.chatbot_query import ChatbotQueryInput, TrainingDataInput
from utils.training_data_utils import load_training_data, update_training_data

load_dotenv()

VERIFY_TOKEN = os.getenv('FB_VERIFY_TOKEN')
BASE_URL = os.getenv('BASE_URL')
PAGE_ID = os.getenv('PAGE_ID')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

app = FastAPI(
    title="Customer Support Chatbot",
    description="Endpoints for a customer support chatbot",
)


def invoke_llm_with_retry(query: str, session_id: str):
    return chatbot_executor(query, session_id)


@app.get("/")
def get_status() -> dict:
    return {"status": "running"}


@app.post("/training")
def upload_training_data(
        query: TrainingDataInput,
) -> dict:
    updated_training_data = update_training_data(query.text)
    return updated_training_data


@app.get("/training")
def get_training_data() -> dict:
    data = load_training_data()
    return data


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
    print("new message")
    data = await request.body()
    data_dict = json.loads(data.decode())
    if data_dict["entry"][0]["messaging"][0].get("message"):
        message = data_dict["entry"][0]["messaging"][0]["message"]["text"]
        print(message)
        sender_id = data_dict["entry"][0]["messaging"][0]["sender"]["id"]
        print(sender_id)
        query_response = invoke_llm_with_retry(message, sender_id)
        message = query_response["output"]
        send_message_url = f"{BASE_URL}{PAGE_ID}/messages?recipient={{id:{sender_id}}}&message={{text:'{message}'}}&messaging_type=RESPONSE&access_token={ACCESS_TOKEN}"
        rq.post(send_message_url)

        return True

    return True
