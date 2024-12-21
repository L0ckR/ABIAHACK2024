from fastapi import FastAPI, HTTPException, Query, File, Body, UploadFile
from typing import List, Dict
import os
from config import Config
from vector_db import RAG


from langchain_core.documents import Document


if not os.path.exists(Config.INPUT_DIR):
    os.makedirs(Config.INPUT_DIR)

if not os.path.exists(Config.OUTPUT_DIR):
    os.makedirs(Config.OUTPUT_DIR)

if not os.path.exists(Config.VB_PATH):
    os.makedirs(Config.VB_PATH)

rag = RAG(Config)
print('ok')

app = FastAPI()


@app.get("/search")
def search_endpoint(
    query: str = Query(..., description="Поисковый запрос"),
    k: int = Query(30, ge=1, description="Количество результатов"),
    threshold: float = Query(0.92, ge=0.0, le=1.0, description="Пороговая оценка")
):
    try:
        documents = [rag.dump_to_json(doc, i) for i, doc in enumerate(rag.search_VB(query, k=k, threshold=threshold))]
        if not documents:
            raise HTTPException(status_code=404, detail="Нет релевантных документов")
        return {"results": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/invoke_llm")
def invoking(
    query: str = Query(..., description="Поисковый запрос"),
    documents: List[Dict] = Body(..., description="docs"),
    k: int = Query(15, ge=1, description="Количество результатов"),
    threshold: float = Query(0.8, ge=0.0, le=1.0, description="Пороговая оценка")
):
    try:
        query = rag.send_req(query, 'preprocess_query', 0.2)
        if len(documents) <= 1:
            documents = [rag.dump_to_json(doc, i) for i, doc in enumerate(rag.search_VB(query, k=k, threshold=threshold))]
            answer = rag.send_final_req(query, documents)
        else:
            answer = rag.send_final_req(query, documents)
        return {"contexts":documents, "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate")
def invoking(
    query: str = Query(..., description="Поисковый запрос"),
    prompt: str = Query(..., description="Поисковый запрос"),
    temperature: float = Query(0.3, ge=0.0, le=2.0, description="temp ")
):
    try:
        answer = rag.send_req(query, prompt, temperature)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rebuild")
def rebuild_endpoint():
    try:
        rag = rag.build_with_parse()
        return {"message": "База данных успешно перестроена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Загружаемый файл должен быть в формате PDF")

        file_path = os.path.join(Config.INPUT_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return {"message": f"Файл {file.filename} успешно загружен в {Config.INPUT_DIR}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))