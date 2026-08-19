import numpy as np

def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    left_data = left.get('data', 0) if isinstance(left, dict) else left
    right_data = right.get('data', 0) if isinstance(right, dict) else right
    computed_data = left_data + right_data
    return {
        "id": output_id,
        "data": computed_data,
        "grad": 0,
        "op": "+",
        "parents": [left, right]
    }