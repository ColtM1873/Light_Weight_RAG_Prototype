# every dict have keys:
# path, name, mtime, version_date, file_prefix, title, id, content

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 模型已在本机缓存，默认离线加载，避免每次联网检查；需要联网时设 HF_HUB_OFFLINE=0
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import re
from pathlib import Path
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from assign_ids_and_versiondate import do_we_have_file_change, return_pipeline_list
from split_into_chunks_and_save import split_it_up


project_dir = Path(__file__).resolve().parent.parent

embedding_model_name = "BAAI/bge-small-zh-v1.5"

model = SentenceTransformer( embedding_model_name)
vector = model.encode( ["维度"] , normalize_embeddings = True)
d_model = vector.shape[1]

path_index_sentences = project_dir / "data" / "index_sentences.faiss"
path_index_chunks = project_dir / "data" / "index_chunks.faiss"

def build_index_sentences( solid_sentences: list[dict] ):
    list_content = [ s["content"] for s in solid_sentences ]
    index_here = faiss.IndexFlatIP(d_model)
    list_embed = model.encode( list_content , normalize_embeddings = True )
    index_here.add(list_embed)
    # faiss.write_index 的 C++ fopen 不支持中文路径，改用序列化后由 Python 写文件
    path_index_sentences.write_bytes(faiss.serialize_index(index_here).tobytes())

def build_index_chunks ( solid_chunks: list[dict] ):
    list_content = [ s["content"] for s in solid_chunks ]
    index_here = faiss.IndexFlatIP(d_model)
    list_embed = model.encode( list_content , normalize_embeddings = True )
    index_here.add(list_embed)
    # faiss.write_index 的 C++ fopen 不支持中文路径，改用序列化后由 Python 写文件
    path_index_chunks.write_bytes(faiss.serialize_index(index_here).tobytes())

def get_index_ready():
    if do_we_have_file_change():
        sorted_pileline_file_list = return_pipeline_list()
        solid_sentences , solid_chunks = split_it_up(sorted_pileline_file_list)
        build_index_sentences(solid_sentences)
        build_index_chunks(solid_chunks)



