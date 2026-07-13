from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="ML Lab")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Список доступных моделей (кнопки на главной)
MODELS = [
    {"id": "classification", "name": "Классификация", "description": "Задачи классификации"},
    {"id": "regression", "name": "Регрессия", "description": "Задачи регрессии"},
    {"id": "clustering", "name": "Кластеризация", "description": "Кластеризация данных"},
    {"id": "nlp", "name": "NLP", "description": "Обработка естественного языка"},
    {"id": "cv", "name": "Computer Vision", "description": "Компьютерное зрение"},
    {"id": "recommendation", "name": "Рекомендации", "description": "Рекомендательные системы"},
]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"models": MODELS},
    )
