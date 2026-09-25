from fastapi import FastAPI

from .graph import ask
from .models import QueryRequest, AssistantResponse

app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AssistantResponse)
def ask_endpoint(request: QueryRequest):
    return ask(request.query)
