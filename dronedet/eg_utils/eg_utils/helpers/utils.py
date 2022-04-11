import os
from os.path import exists, expanduser, join, realpath
from pathlib import Path
from typing import List, Optional


def find_path(file_path: str, root_path: str = "", check_existence: bool = True) -> str:
    if len(root_path) == 0:
        root_path = os.getcwd()
    if "~" == file_path[0]:
        ret = expanduser(file_path)
        if check_existence and not exists(ret):
            raise Exception("'{}' not found".format(ret))
        return ret
    else:
        joined = False
        if len(file_path) > 1 and "./" == file_path[:2]:
            file_path = join(root_path, file_path[2:])
            joined = True
        elif len(file_path) > 2 and "../" == file_path[:3]:
            file_path = join(str(Path(root_path).parent), file_path[3:])
            joined = True

        if check_existence:
            if exists(file_path):
                return file_path
            elif exists(realpath(file_path)):
                return realpath(file_path)
            elif not joined:
                return join(root_path, file_path)
            else:
                raise Exception("'{}' not found".format(file_path))
        else:
            return file_path


def listdir(path: str, base: Optional[str] = None) -> List[str]:
    all_files = []
    for file in os.listdir(path):
        tmp_base = file if base is None else os.path.join(base, file)
        tmp_path = os.path.join(path, file)
        if os.path.isdir(tmp_path):
            all_files += listdir(path=tmp_path, base=tmp_base)
        else:
            all_files.append(tmp_base)
    return all_files
