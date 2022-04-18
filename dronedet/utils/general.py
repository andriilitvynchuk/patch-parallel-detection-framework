from typing import Any, List, Optional


def get_index(element: Any, element_list: List[Any]) -> Optional[Any]:
    try:
        index_element = element_list.index(element)
        return index_element
    except ValueError:
        return None
