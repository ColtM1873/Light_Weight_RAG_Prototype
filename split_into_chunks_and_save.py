# every dict have keys:
# path, name, mtime, version_date, file_prefix, title, id

import json
import re
from pathlib import Path
from datetime import datetime

project_dir = Path(__file__).resolve().parent.parent

ref_files_dir = project_dir /"data"/"ref_files"
path_non_ids_map = project_dir/ "data" / "non_ids_map.json"
re_version_date = re.compile(r"^版本日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})")

path_sentences = project_dir / "data" / "solid_sentences.json"
path_chunks = project_dir / "data" /  "solid_chunks.json"

CHUNK_SIZE = 300
MAX_SIZE_CAPABLE = 512

def get_sentences( raw_text: str ) -> list[str]:
    sentences = re.split(r"(?<=[，。！？；\n])" , raw_text)
    sentences =  [ sentence.strip() for sentence in sentences if sentence.strip() ]
    final_sentences : list[str] = []
    for i in range(len(sentences)):
        s = sentences[i]
        if len(s) > MAX_SIZE_CAPABLE:
            pieces_divide = len(s) // MAX_SIZE_CAPABLE + 1
            piece_len = len(s) // pieces_divide
            start_indice = 0
            end_indice = piece_len - 1
            while end_indice < len(s):
                final_sentences.append( s[start_indice : end_indice + 1] )
                end_indice += piece_len
                start_indice += piece_len
            if s[start_indice:] :
                final_sentences.append( s[start_indice :] ) 
        else:
            final_sentences.append(s)
    return final_sentences


def chunk_up( sentences : list[str] ):
    chunks_already = []
    buf = ""
    last_s_of_last_chunk = ""
    last_s_of_this_chunk = ""
    for s in sentences:
        if len(last_s_of_last_chunk) + len(buf) + len(s) <= CHUNK_SIZE:
            buf += s
        else:
            if buf:  # this chunk already has something
                # means its safe to contai last_s_of_last_chunk
                # and last_s_of_this_chunk isnt None
                chunks_already.append(last_s_of_last_chunk + buf)
                buf = ""
                last_s_of_last_chunk = last_s_of_this_chunk
                last_s_of_this_chunk = ""

                buf += s
                last_s_of_this_chunk = s
            else:
                # this chunk has nothing
                # thus len(last_s_of_this_chunk) + len(s) > CHUNK_SIZE
                chunks_already.append(s)
                buf = ""
                last_s_of_last_chunk = ""
                last_s_of_this_chunk = ""
    if buf: 
        chunks_already.append(last_s_of_last_chunk + buf) if len(last_s_of_last_chunk) + len(buf) <= MAX_SIZE_CAPABLE else chunks_already.append(buf)
    return chunks_already



def split_it_up (sorted_pileline_file_list : list[dict]) -> tuple[list[dict], list[dict]]:
    solid_sentences :list[dict] = []
    solid_chunks : list[dict] = []
    for i in range(len(sorted_pileline_file_list)):
        file_here = sorted_pileline_file_list[i]

        # to peel off the first version_date line if exists
        raw_text_here = file_here["path"] . read_text(encoding = "utf-8")
        lines = raw_text_here.splitlines()
        if not lines: # no content
            continue
        first_line = lines[0].strip().lstrip("\ufeff")
        match_here = re_version_date.match(first_line)
        if match_here:
            raw_text_here = "\n".join(lines[1:])

        sentences = get_sentences(raw_text_here)
        chunks = chunk_up(sentences)
        solid_sentences_here = [ {"content" : s} + file_here for s in sentences ]
        solid_chunks_here = [ {"content" : chu} + file_here for chu in chunks ]
        solid_sentences += solid_sentences_here
        solid_chunks += solid_chunks_here
    path_sentences.parent.mkdir( parents = True , exist_ok = True )
    path_sentences.write_text(
            json.dumps( solid_sentences , ensure_ascii = False,
                       indent = 2),
            encoding = "utf-8"
            )
    path_chunks.write_text(
            json.dumps( solid_chunks , ensure_ascii = False,
                       indent = 2),
            encoding = "utf-8"
            )
    return solid_sentences, solid_chunks







