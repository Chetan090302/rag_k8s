from fastapi import FastAPI,HTTPException
from rag_pipeline import trigger_graph
from pydantic import BaseModel
from typing import Optional

class Input(BaseModel):
    question:str
    answer:str=""
    documents:list=[]

class Output(BaseModel):
    response:str
    status_code:int

app=FastAPI()

@app.get("/root")
def get_root():
    return {"response":"Hello"}

@app.get("/")
def get_root():
    return {"response":"Hello world"}

@app.post("/pass_question",response_model=Output)
def chat_llm_model(input:Input):
    input_data=input.model_dump()
    print(f"input {input_data}")
    res=trigger_graph(input_data)
    if res:
        return Output(response=res,status_code=200)
    return Output(response="Exception occured",status_code=404)

