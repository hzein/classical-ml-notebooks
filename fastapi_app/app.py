from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = FastAPI()

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    email_text=""
    return templates.TemplateResponse(request=request, name="index.html", context={'request': request, 'email_text': email_text})

@app.post("/predict")
def predict(request: Request, email_text: str = Form(default="")):
    print(f"EMAIL TEXT: {[email_text]}")
    tokenized_email = tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return templates.TemplateResponse('index.html', context={'request': request, 'email_text': email_text, 'predictions': predictions})