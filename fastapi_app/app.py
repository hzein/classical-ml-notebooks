from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils import get_preds

class Email(BaseModel):
    text: str = None

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    email_text=""
    return templates.TemplateResponse(request=request, name="index.html", context={'request': request, 'email_text': email_text})

@app.post("/predict")
def predict(request: Request, email_text: str = Form(default="")):
    predictions = get_preds(email_text=email_text)
    return templates.TemplateResponse('index.html', context={'request': request, 'email_text': email_text, 'predictions': predictions})

@app.post("/api/predict")
def api_predict(email: Email):
    email_text = email.text
    predictions = get_preds(email_text=email_text)
    return {"email": email_text, "predictions": predictions}