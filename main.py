#pip install fastapi[all] uvicorn pydantic

import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types 
from dotenv import load_dotenv 

load_dotenv()

API_KEY = os.environ["API_KEY"]
client = genai.Client(api_key=API_KEY)

app = FastAPI(
    title="챗봇 API",
    description="TXT 파일 지식을 기반으로 답변하며, JSON으로 대화 내역을 기억하는 챗봇 API입니다.",
    version="1.0.0"
)

HISTORY_FILE = "chat_history.json"
KNOWLEDGE_FILE = "멋사규칙.txt"

def load_knowledge():
    os.path.exists(KNOWLEDGE_FILE)
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return f.read()

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

knowledge_context = load_knowledge()
sys_instruct = f"""
너는 제공된 [참고 문서]의 내용을 기반으로 질문에 답변하는 전문 안내 AI 도우미야.
[답변 규칙]
1. 반드시 아래 제공된 [참고 문서]에 있는 사실에만 기반해서 답변해줘.
2. 문서에 없는 내용은 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 정중하게 답변해.
3. 항상 친절하고 상냥한 존댓말을 사용해.
[참고 문서]
{knowledge_context}
"""

class ChatRequest(BaseModel):
    user_input: str

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="질문")

    chat_history = load_history()

    chat_history.append({
        "role": "user",
        "parts": [{"text": request.user_input}]
    })

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=chat_history, 
            config=types.GenerateContentConfig(
                system_instruction=sys_instruct
            )
        )

        ai_reply = response.text
        chat_history.append({
            "role": "model",
            "parts": [{"text": ai_reply}]
        })
        save_history(chat_history)

        return {
            "status": "success",
            "response": ai_reply
        }

    except Exception as e:
        chat_history.pop()
        raise HTTPException(status_code=500, detail=f"에러: {str(e)}")