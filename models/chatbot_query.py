from pydantic import BaseModel


class ChatbotQueryInput(BaseModel):
    text: str
    session_id: str


class ChatbotQueryOutput(BaseModel):
    input: str
    output: str


class TrainingDataInput(BaseModel):
    text: str
