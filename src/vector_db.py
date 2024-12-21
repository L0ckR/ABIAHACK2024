import re
import os
import json

from typing import List

import pandas as pd
import numpy as np

import torch
import faiss

from parser_pdf import pdf_to_json
from overlap import process_json_file

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors import FlashrankRerank

from openai import OpenAI

    
class RAG():
    def __init__(self, Config):  
        self.actualConfig = Config
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.VB_PATH = Config.VB_PATH
        self.INPUT_DIR = Config.INPUT_DIR
        self.OUTPUT_DIR = Config.OUTPUT_DIR
        self.PROMPTS_DIR = Config.PROMPTS_DIR
        self.LLM_MODEL_NAME=Config.LLM_MODEL_NAME
        self.model = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        with open(os.path.join(self.PROMPTS_DIR, 'multi_query.txt'), 'r') as file:
            prompt_template = file.read()
        self.prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template
        )

        self.llm_client = OpenAI(
            base_url=Config.LLM_MODEL_URL,
            api_key=Config.LLM_MODEL_API)
        
        self.llm_model = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            base_url=Config.LLM_MODEL_URL,
            openai_api_key=Config.LLM_MODEL_API,
            temperature=0
        ).configurable_fields(
            temperature=ConfigurableField(
            id="llm_temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
            )
        )

            
        try:
            self.df = pd.read_csv(os.path.join(self.VB_PATH, 'docs_info.csv'))
            self.load()
        except Exception as e:
            if os.listdir(self.INPUT_DIR) != os.listdir(self.OUTPUT_DIR):
                self.build_with_parse()
            else:
                self.build()
                
    def load(self):
        self.vb = FAISS.load_local(os.path.join(self.VB_PATH, 'vector_database'), embeddings=self.model, allow_dangerous_deserialization=True)
        return self

    def build(self):
        print("END OF PARSING")
        self.vb = self.VB_build(self.OUTPUT_DIR)
        return self
    
    def build_with_parse(self):
        self.parse_pdf(self.INPUT_DIR, self.OUTPUT_DIR)
        print("END OF PARSING")
        self.vb = self.VB_build(self.OUTPUT_DIR)
        return self

    
    def parse_pdf(self, inp_dir, outp_dir):
        files = [f for f in os.listdir(inp_dir) if os.path.isfile(os.path.join(inp_dir, f)) and f.endswith('.pdf')]
    
        for i in files:
            pdf_to_json(os.path.join(inp_dir, i), os.path.join(outp_dir, "output_"+i))
        
        dirs = [f for f in os.listdir(outp_dir) if os.path.isdir(os.path.join(outp_dir, f)) ]
        for i in dirs:
            files = [f for f in os.listdir(os.path.join(outp_dir, i)) if os.path.isfile(os.path.join(outp_dir, i, f)) and f.endswith('.json')]
            for file in files:
                process_json_file(os.path.join(outp_dir, i, file), context_size=100)
           
    def chunk_split(self, outp_dir):
        rows = []
        dirs = [f for f in os.listdir(outp_dir) if os.path.isdir(os.path.join(outp_dir, f)) ]
        for i in dirs:
            files = [f for f in os.listdir(os.path.join(outp_dir, i)) if os.path.isfile(os.path.join(outp_dir, i, f)) and f.endswith('.json')]
            for file in files:
                with open(os.path.join(outp_dir, i, file), 'r', encoding='utf-8') as f:
                    js = json.load(f)
                    for section in js:
                        rows.append(section)
        return pd.DataFrame(rows)
    

    def VB_build(self, outp_dir):
        df = self.dataframe_preprocess(self.chunk_split(outp_dir))
        print('vb building')
        sample_text = "Sample text."
        embedding = self.model.embed_query(sample_text)
    
        dimension = len(embedding)
        index = faiss.IndexFlatIP(dimension)
        df['embedded_text'] = df['embedded_text'].apply(lambda x: np.array(x, dtype=np.float32))
        embeddings = np.stack(df['embedded_text'].tolist())
        faiss.normalize_L2(embeddings) 
        index.add(embeddings)

        documents = [
            Document(page_content=text, metadata={'filename': sf , "image_paths": pths, "section_number": sect})
            for text, sf, pths, sect in zip(df['chunk'], df['filename'], df['image_paths'], df['section_number'])
        ]
        
        docstore = InMemoryDocstore(dict(enumerate(documents)))
        
        self.vb = FAISS(
            embedding_function=self.model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=dict(zip(range(len(embeddings)), range(len(embeddings)))),
        )
        self.vb.save_local(os.path.join(self.VB_PATH, 'vector_database'))
        self.df = df
        self.df.to_csv(os.path.join(self.VB_PATH, 'docs_info.csv'), index=False)
        
        return self.vb
    
    def search_VB(self, query, k=20, threshold=0.8, multi_query: bool = False, rerank: bool = True):

        
        if multi_query:
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=self.vb.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': k*2, 'lambda_mult': 0.35}
                ),
                llm=self.llm_model,
                parser_key="lines",
                prompt=self.prompt
            )
            if rerank:        
                compressor = CrossEncoderReranker(model=self.rerank_model, top_n=k)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=retriever_from_llm
                )
                docs = compression_retriever.invoke({query})

            else:
                docs = retriever_from_llm.invoke({query})
        else:
            if rerank:        
                compressor = CrossEncoderReranker(model=self.rerank_model, top_n=k)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=self.vb.as_retriever()
                )
                
                docs = compression_retriever.invoke(
                    query
                )
            else:
                docs = self.vb.as_retriever().invoke({query})
        return docs


            
    
    
    def dataframe_preprocess(self, df):
        df = df.dropna().reset_index(drop=True)
    
        df['preprocessed_text'] = df['chunk'].apply(lambda x: self.preprocess_text(x))
    
        df['embedded_text'] = df['preprocessed_text'].apply(lambda x: self.model.embed_query(x))
    
        return df

    def send_req(self, user_input: str, prompt:str, temperature: float = 0.3) -> str:
        with open(os.path.join(self.PROMPTS_DIR, prompt+'.txt'), 'r') as file:
            GROUNDED_SYSTEM_PROMPT = file.read()
            sample_history = [
                ('system',GROUNDED_SYSTEM_PROMPT), 
                ('user',user_input)
            ]
            final_answer = self.llm_model.with_config(
                configurable={
                    "llm_temperature": temperature
                }).invoke(
                sample_history
            ).content
            return final_answer
    
    def send_final_req(self, user_input: str, documents: List[Document]) -> str:
        with open(os.path.join(self.PROMPTS_DIR, 'prompt.txt'), 'r') as file:
            GROUNDED_SYSTEM_PROMPT = file.read()
            sample_history = [
                {'role': 'system', 'content': GROUNDED_SYSTEM_PROMPT}, 
                {'role': 'documents', 'content': json.dumps(documents, ensure_ascii=False)},
                {'role': 'user', 'content': user_input}
            ]
            
            relevant_indexes = self.llm_client.chat.completions.create(
                messages=sample_history,
                model=self.LLM_MODEL_NAME,
                temperature=0.3
            ).choices[0].message.content
        
            final_answer = self.llm_client.chat.completions.create(
                model=self.LLM_MODEL_NAME,
                messages=sample_history + [{'role': 'assistant', 'content': relevant_indexes}],
                temperature=0.0
            ).choices[0].message.content
            return final_answer

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'[^\w\s.,?!;:«»"\'\-]+', '', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text) 
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'\.{5,}', 'на странице', text)
        
        text = text.lower()

        return text

    
    @staticmethod
    def dump_to_json(x: Document, i: int):
        x = x.model_dump()
        return x

if __name__ == "__main__":

    from config import Config
    rag = RAG(Config)
    print(rag.search_VB("как починить"))
    print(rag.send_final_req("починить", []))