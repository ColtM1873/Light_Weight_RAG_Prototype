# this file only does two things and two things only,
# one, write and id_map json in disk
# two, return a list of file indicators sorted by ids for chunk
# return a list[dict]
# every dict have keys:
# path, name, mtime, version_date, file_prefix, title, id

import json
import re
from pathlib import Path
from datetime import datetime
import hashlib

set_file_suffix = {".txt" , ".md"}

project_dir = Path(__file__).resolve().parent.parent

ref_files_dir = project_dir /"data"/"ref_files"
path_non_ids_map = project_dir/ "data" / "non_ids_map.json"
path_hash_file = project_dir / "data" / "ref_files_hash.hash"

non_ids_starter = 1000

re_format_required = re.compile(r"^D(\d+)_(.+)$")
re_version_date = re.compile(r"^版本日期\s*[:：]\s*(\d{4}-\d{2}-\d{2})")

def validate_date(s : str) -> bool:
    try:
        datetime.strptime(s , "%Y-%m-%d")
        return True
    except ValueError:
        return False

def get_version_date( path : Path):
    with path.open(encoding = "utf-8" ) as file:
        first_line = file.readline().strip().lstrip("\ufeff")
        match_here = re_version_date.match(first_line)
        if match_here:
            version_date = match_here.group(1)
            if validate_date(version_date):
                return version_date
            else:
                return None
        else:
            return None

def get_sha256_of_file ( path: Path , chunk_size : int = 1024 * 1024 ) -> bytes :
    hash_here = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hash_here.update(chunk)
    return hash_here.digest()

def calc_total_hash ( docs_dir : str | Path = ref_files_dir ) -> list[dict]:
    docs_dir = Path(docs_dir)
    hash_here = hashlib.sha256()

    for path in sorted( docs_dir.iterdir() ):
        if not path.is_file() or path.suffix.lower() not in set_file_suffix:
            continue
        hash_here.update( get_sha256_of_file(path) )

    return hash_here.hexdigest()

def do_we_have_file_change() -> bool:
    if not path_hash_file.exists():
        return True
    else:
        cur_hash : str = calc_total_hash()
        with path_hash_file.open("rt", encoding="utf-8") as file:
            old_hash = file.read().strip().lstrip("\ufeff")
            if cur_hash == old_hash:
                return False
            else:
                return True

def save_cur_hash():
    cur_hash: str = calc_total_hash()
    with path_hash_file.open("wt", encoding="utf-8") as file:
        file.write(cur_hash)

def scan_files( docs_dir : str | Path = ref_files_dir ) -> list[dict]:
    docs_dir = Path(docs_dir)
    list_files = []
    for path in sorted( docs_dir.iterdir() ):
        if not path.is_file() or path.suffix.lower() not in set_file_suffix:
            continue
        mtime = path.stat().st_mtime
        version_date = get_version_date(path)
        list_files.append(
                {"path" : str(path),
                 "name" : path.name,
                 "mtime": mtime,
                 "version_date" : version_date,}
                 # **需要修改**  "file_hash" : get_sha256_of_file(path).hex() }
                )
    return list_files

# **需要修改** 
# 需要有一个存储 dict_path_file_hash 的过程
# 也就是 文档 path（str化） 到 file_hash 的映射
# 方便后面读取 文档全文时，在需要的时候提示文件被修改
# **需要修改** 


def assign_ids (list_files_nonids : list[dict] , ids_occupied : set[int]):
    cur_id = non_ids_starter
    id_map = {}
    for i in range(len(list_files_nonids)):
        while cur_id in ids_occupied:
            cur_id += 1
        file = list_files_nonids[i]
        file["id"] = cur_id
        file["file_prefix"] = f"D{cur_id}"
        id_map[cur_id] = file["path"]
        cur_id += 1
    path_non_ids_map.parent.mkdir( parents = True , exist_ok = True )
    path_non_ids_map.write_text(
            json.dumps(id_map , ensure_ascii = False ,
                      indent = 2),
            encoding = "utf-8"
            )

def classify( list_files : list[dict] ) -> tuple[list[dict] , list[dict] ]:
    list_files_ids = [] 
    # file_prefix : str
    # id : int
    # title : str
    list_files_nonids = []
    ids_occupied = set()
    for file in list_files:
        file_name = Path( file["name"] ) .stem # without suffix
        match_here = re_format_required.match(file_name)
        if match_here:
            file["file_prefix"] = f"D{match_here.group(1)}" # with pre-zeros saved
            file["title"] = match_here.group(2)
            id_here = int(match_here.group(1))
            file["id"] = id_here
            list_files_ids.append(file)

            ids_occupied.add(id_here)
        else:
            file["title"] = file_name
            list_files_nonids.append(file)
    assign_ids(list_files_nonids , ids_occupied)
    sorted_pileline_file_list = list_files_ids + list_files_nonids
    sorted_pileline_file_list.sort(key = lambda x : x["id"])
    return sorted_pileline_file_list


def return_pipeline_list():
    save_cur_hash()
    list_files = scan_files()
    return classify(list_files)














