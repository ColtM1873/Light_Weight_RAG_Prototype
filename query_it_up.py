# every dict have keys:
# path, name, mtime, version_date, file_prefix, title, id, content

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool as langchain_tool
from typing import Annotated, Literal


project_dir = Path(__file__).resolve().parent.parent

embedding_model_name = "BAAI/bge-small-zh-v1.5"

model = SentenceTransformer( embedding_model_name)
vector = model.encode( ["维度"] , normalize_embeddings = True)
d_model = vector.shape[1]

path_index_sentences = project_dir / "data" / "index_sentences.faiss"
path_index_chunks = project_dir / "data" / "index_chunks.faiss"
path_sentences = project_dir / "data" / "solid_sentences.json"
path_chunks = project_dir / "data" /  "solid_chunks.json"

query_prefix = "为这个句子生成表示以用于检索相关文章："

# **需要修改**
already_queried_accumulate_times = {}  #  需要是会话的context或者config其他什么地方透传进tool包裹的函数，不能只作为一个全局变量（这里的会话 指langgraph 中串起一条线的那个state）
# **需要修改**


# **需要修改**
solid_sentences : list[dict] = json.loads(
    path_sentences.read_text(encoding="utf-8")
)
solid_chunk : list[dict] = json.loads(
    path_chunks.read_text(encoding="utf-8")
)
index_sentences = faiss.read_index(str(path_index_sentences))
index_chunks = faiss.read_index(str(path_index_chunks))
# 这四个应该一次运行中，常驻于内存中，直接在这个python文件写一个全局变量是否合适？
# **需要修改**

async def get_merged_list_dict(s_ids : list[int] , c_ids: list[int]):
    chunk_list = []
    sentence_list = []
    # path, name, mtime, version_date, file_prefix, title, id, content

    for id_here in c_ids:
        origin_dict = solid_chunk[id_here]
        working_dict = {}
        working_dict["document_id"] = origin_dict["id"]
        working_dict["piece_content"] = origin_dict["content"]
        working_dict["piece_id"] = id_here
        working_dict["piece_type"] = "chunk"
        chunk_list.append(working_dict)

    def not_in_chunks(s_id : int) -> bool:
        content_str = solid_sentences[s_id]["content"]
        for dict_here in chunk_list:
            if content_str in dict_here["piece_content"]:
                return False
        return True
    
    for id_here in s_ids:
        if not not_in_chunks(id_here):
            continue
        origin_dict = solid_sentences[id_here]
        working_dict = {}
        working_dict["document_id"] = origin_dict["id"]
        working_dict["piece_content"] = origin_dict["content"]
        working_dict["piece_id"] = id_here
        working_dict["piece_type"] = "sentence"
        sentence_list.append(working_dict)

    return chunk_list + sentence_list

async def raw_query_func (query_str :str , already_num : int , new_num : int ) -> dict|str:
    query_vector = model.encode([query_prefix + query_str], normalize_embeddings = True)
    s_scores , s_ids = index_sentences.search( query_vector , already_num + new_num)
    c_scores , c_ids = index_chunks.search( query_vector , 
    already_num + new_num)
    # query只有一个，只取第一行
    s_ids = s_ids[0]
    c_ids = c_ids[0]


    if len(s_ids) <= already_num and len(c_ids) <= already_num:
        return "没有（更多的）匹配项。（匹配项条数为零条）"
    
    s_ids = s_ids[already_num : already_num + new_num]
    c_ids = c_ids[already_num : already_num + new_num]
    list_dict = await get_merged_list_dict(s_ids,c_ids)
    return list_dict

@langchain_tool
async def query_the_docs(
    query_str: Annotated[str, "要在文档库里面查询的内容"],
    number_of_retrieval: Annotated[int , "打算返回的条数，实际返回条数在这个数的一倍和两倍之间（匹配文档不足时可能返回数目的少于这个数）"],
    first_try_on_same_query: Annotated[bool , "是否是某个查询内容的第一次查询。如果传入False，会自动扩大召回条数并去重。比如第一次和第二次都查询“x”，第一次选择返回4条，第二次选择返回6条，那么第二次在这里传入False之后，实际就会检索10条，然后去掉优先级靠前的4条后进行返回。传入False要求查询内容完全一样。默认值为True。对于已经查询过的内容传入True进行查询，会重置累计查询次数。"] = True,
) -> str|list[dict]:
    """使用RAG方式返回文档库中的匹配内容，并返回最匹配的若干条。会从单句和小段落两个层级进行匹配，并自动去重单句属于小段落的返回内容，所以最终的返回条数会在选择返回条数的一倍到两倍之间（匹配文档充足情况下）。
    如果工具调用没有出错的话，会返回一个list[dict]，其中每个dict的字段的解释如下： 
    document_id ：这条返回内容出自于哪一个文档（以ID号形式标明）
    piece_content：该条检索结果的具体内容
    piece_id：该条检索结果自身的ID
    piece_type：是单句和小段落中的哪一类返回结果"""
    if not first_try_on_same_query:
        if query_str not in already_queried_accumulate_times:
            return "这条检索内容是第一次检索，不应该在first_try_on_same_query传入False。"
        else:
            try:
                return_dict =  await raw_query_func(query_str= query_str , already_num = already_queried_accumulate_times[query_str] , new_num= number_of_retrieval)
                already_queried_accumulate_times[query_str] += number_of_retrieval 
                return return_dict
            except Exception as e:
                return f"工具调用失败，异常名称：{e}"
        
    elif first_try_on_same_query:
        try:
            return_dict =  await raw_query_func(query_str= query_str , already_num = 0 , new_num= number_of_retrieval)
            already_queried_accumulate_times[query_str] =  number_of_retrieval
            return return_dict
        except Exception as e:
            return f"工具调用失败，异常名称：{e}"

async def retrieve_context( piece_id : int , number_of_context_to_retrieve : int , solid_list : list[dict]) -> list[str] | str:
    rear_half = []
    head_half = []
    if not  0 <= piece_id < len(solid_list):
        return "piece_id 超出范围，请复核"
    cur_doc_id = solid_list[piece_id]["id"]
    bump_head = False
    bump_rear = False
    # to rear
    half_len = number_of_context_to_retrieve // 2
    for i in range(half_len):
        target_id = piece_id + i + 1
        target_doc_id = solid_list[target_id]["id"]
        if target_doc_id != cur_doc_id:
            bump_rear = True
            break
        else:
            rear_half.append(solid_list[target_id]["content"])
    if not solid_list[target_id + 1]["id"] == cur_doc_id:
        bump_rear = True

    rear_id = target_id + 1

    for i in range(number_of_context_to_retrieve - len(rear_half) - 1):
        target_id = piece_id - i - 1
        target_doc_id = solid_list[target_id]["id"]
        if target_doc_id != cur_doc_id:
            bump_head = True
            break
        else:
            head_half.append(solid_list[target_id]["content"])
    if not solid_list[target_id - 1]["id"] == cur_doc_id:
        bump_head = True

    head_id = target_id - 1

    if bump_rear and not bump_head:
        for i in range(number_of_context_to_retrieve - len(rear_half) - len(head_half) - 1):
            target_id = head_id - i 
            target_doc_id = solid_list[target_id]["id"]
            if target_doc_id != cur_doc_id:
                bump_head = True
                break
            else:
                head_half.append(solid_list[target_id]["content"])
        if not solid_list[target_id - 1]["id"] == cur_doc_id:
            bump_head = True

    elif bump_head and not bump_rear:
        for i in range(number_of_context_to_retrieve - len(rear_half) - len(head_half) - 1):
            target_id = rear_id + i
            target_doc_id = solid_list[target_id]["id"]
            if target_doc_id != cur_doc_id:
                bump_rear = True
                break
            else:
                rear_half.append(solid_list[target_id]["content"])
        if not solid_list[target_id + 1]["id"] == cur_doc_id:
            bump_rear = True
    head_str = "（原文档前面还有内容）" if not bump_head else "（已经抵达原文档开头）"
    rear_str = "（原文档后面还有内容）" if not bump_head else "（已经抵达原文档末尾）"
    head_half.reverse()
    return_list = [head_str] + head_half + solid_list[piece_id]["content"] + rear_half + [rear_str]
    return return_list


@langchain_tool
async def retrieve_piece_context(
    piece_id: Annotated[int, "要返回上下文的检索结果ID"],
    piece_type: Annotated[Literal["sentence" , "chunk"] , "要返回上下文的检索结果类型，必须跟piece_id对应，只能是 sentence 和 chunk 中的一种"],
    number_of_context_to_retrieve: Annotated[int , "返回上下文的数量（长度）"] ,
) -> list[str] | str:
    """如果使用query_the_docs得到了感兴趣的内容，需要进一步查看相关内容的上下文，就调用这个工具。注意，传入的piece_id 和 piece_type要对应。至于返回上下文的数量（长度），原则上前后各一半，如果是文档开头结尾则做动态调整。如果piece_type传入sentence，就会返回piece_id对应的检索结果上下的 返回上下文长度 个句子；如果piece_type传入chunk，就会返回这么多个小段落。文档前后还有没有内容，会明确在返回内容里标出来。"""
    # solid_sentences 
    # solid_chunk 
    # path, name, mtime, version_date, file_prefix, title, id, content
    if piece_type == "sentence":
        return await retrieve_context(piece_id= piece_id , number_of_context_to_retrieve= number_of_context_to_retrieve , solid_list= solid_sentences)
    elif piece_type == "chunk":
        return await retrieve_context(piece_id= piece_id , number_of_context_to_retrieve= number_of_context_to_retrieve , solid_list= solid_chunk)
    else:
        return "piece_type 错误，只能是 sentence 和 chunk 中的一种"


@langchain_tool
async def read_full_document(
    document_id: Annotated[int, "要阅读全文的文档ID"],
) -> str:
    """用来阅读文档的全文。（可能会很长）"""
# **需要修改** 
# 需要实现
# 如果文档hash值跟 solid_xxx 里面的对不上了，在在返回str的最开头加一句 “（文件hash值已经变动，可能跟检索结果有出入/不可信）”
# **需要修改** 

@langchain_tool
async def get_version_date(
    document_id: Annotated[int, "要查阅版本日期的文档ID"],
) -> str:
    """用来查阅文档的 版本日期，可以再两个文档内容冲突的时候调用这个工具。"""
# **需要修改** 
# 需要实现
# **需要修改** 

async def get_tools():
    return [query_the_docs , retrieve_piece_context, read_full_document, get_version_date]
        
