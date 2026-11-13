# every dict have keys:
# path, name, mtime, version_date, file_prefix, title, id, content

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    faiss.write_index ( index_here , str(path_index_sentences) )

def build_index_chunks ( solid_chunks: list[dict] ):
    list_content = [ s["content"] for s in solid_chunks ]
    index_here = faiss.IndexFlatIP(d_model)
    list_embed = model.encode( list_content , normalize_embeddings = True )
    index_here.add(list_embed)
    faiss.write_index ( index_here , str(path_index_chunks) )

def get_index_ready():
    if do_we_have_file_change():
        sorted_pileline_file_list = return_pipeline_list()
        solid_sentences , solid_chunks = split_it_up(sorted_pileline_file_list)
        build_index_sentences(solid_sentences)
        build_index_chunks(solid_chunks)



