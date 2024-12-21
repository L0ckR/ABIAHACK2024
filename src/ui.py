import streamlit as st
import requests
import json
from typing import List, Tuple, Optional

from pydantic import BaseModel

class Instruction(BaseModel):
    title: str
    done:bool
    description: Optional[str]
    image_url: Optional[str]

class CheckList(BaseModel):
    name:str
    instruction:List[Instruction]

BACKEND_URL = "http://localhost:7777"

def search_VB(query: str, top_k: int = 10, threshold: float = 0.5):
    response = requests.get(
        f"{BACKEND_URL}/search",
        params={
            "query": query,
            "k": top_k,
            "threshold": threshold
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        contexts = [repr(json.dumps(item, ensure_ascii=False)) for item in data["results"]]
        return contexts
    else:
        return []

def send_req(user_query: str, k: int = 5, threshold: float = 0.5) -> Tuple[str, List]:
    response = requests.get(
        f"{BACKEND_URL}/invoke_llm",
         params={
            "query": user_query,
            "k": k
            },
        json=list(dict()), verify=False)
    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
        llm_answer = response_data.get("response", "Ошибка при запросе к модели")
        contexts = [repr(json.dumps(item, ensure_ascii=False)) for item in response_data.get("contexts", [])]
    else:
        response_data = response.json()
        print(response_data)
        llm_answer = "Ошибка при запросе к модели"
        contexts = []
    return (llm_answer, contexts)

def generate_checklist(query: str, prompt : str, temperature: float) -> str:
    response = requests.get(
        f"{BACKEND_URL}/generate",
        params={
            "query": query, 
            "prompt": prompt,
            "temperature": temperature
            }
    )
    if response.status_code == 200:
        return response.text
    else:
        return "Ошибка при генерации чек-листа"

def clean_context(ctx):
    cleaned_ctx = ctx.replace("page_content='", "").replace("'", "").strip()
    return cleaned_ctx

st.set_page_config(page_title="Муравьиная ферма - AI-помощник", layout="wide")

st.title("Муравьиная ферма - AI-помощник инженера")

threshold = 0.8

left_col, right_col = st.columns([1, 2], gap="medium")

if 'contexts' not in st.session_state:
    st.session_state.contexts = []
if 'model_answer' not in st.session_state:
    st.session_state.model_answer = ""
if 'checklist_result' not in st.session_state:
    st.session_state.checklist_result = ""

with left_col:
    st.markdown("### Запрос к модели")
    user_query = st.text_input("Введите ваш вопрос:")
    if st.button("Отправить запрос"):
        if user_query.strip():
            answer, found_contexts = send_req(user_query, k = 2)
            st.session_state.contexts = found_contexts
            st.session_state.model_answer = answer
            st.session_state.checklist_result = ""
        else:
            st.warning("Пожалуйста, введите вопрос.")

    if st.session_state.model_answer:
        if st.button("Сформировать чек-лист"):
            st.session_state.checklist_result = generate_checklist(user_query)

with right_col:
    st.markdown("### Результаты поиска и ответа")

    if st.session_state.contexts:
        num_contexts = len(st.session_state.contexts)
        with st.expander(f"Контексты ({num_contexts}):", expanded=False):
            for i, ctx in enumerate(st.session_state.contexts):
                cleaned_ctx = clean_context(ctx)
                if st.checkbox(f"Контекст №{i+1}", key=f"context_{i}"):
                    st.markdown(f"> {cleaned_ctx}")
    else:
        st.write("Контексты не найдены или пока нет результатов.")

    if st.session_state.model_answer:
        with st.expander("Ответ от модели", expanded=True):
            st.write(st.session_state.model_answer)

    if st.session_state.checklist_result:
        with st.expander("Ваш чек-лист", expanded=True):
            st.write(st.session_state.checklist_result)

st.sidebar.markdown("© 2024 profagro")
st.sidebar.markdown("Сделано для удобства трактористов и инженеров")
