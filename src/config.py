import os
from pathlib import Path
class Config:
    LLM_MODEL_URL: str = os.getenv('LLM_MODEL_URL', default='https://api.groq.com/openai/v1')
    LLM_MODEL_NAME: str = os.getenv('LLM_MODEL_NAME', default='mixtral-8x7b-32768')
    LLM_MODEL_API:str = os.getenv('LLM_MODEL_API', default='')
    INPUT_DIR:Path = os.getenv('INPUT_DIR', default=Path(r'C:/Users/User/Desktop/mena/avia_hack/aviahack/input_docs/'))
    OUTPUT_DIR:Path = os.getenv('OUTPUT_DIR', default=Path(r'C:/Users/User/Desktop/mena/avia_hack/aviahack/output_docs/'))
    VB_PATH:Path = os.getenv('VB_PATH', default=Path(r'C:/Users/User/Desktop/mena/avia_hack/aviahack/vector_base/'))
    EMBED_MODEL_NAME:str = os.getenv('EMBED_MODEL_NAME', default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    PROMPTS_DIR:Path = os.getenv('PROMPTS_DIR', default=Path(r'C:/Users/User/Desktop/mena/avia_hack/aviahack/prompts/'))
