"""
This file should contain any utility functions, that help with processing collections of any kind.

"""

import numpy as np


def extract_feature_values(list_of_dicts, key, dtype=None) -> np.ndarray:
    """
    Will extract a given feature (using parameter `key`) from a list of dictionaries, and will put all the values into
    a single Numpy array. It can, possibly, cast values to required dtype.

    :param list_of_dicts:
    :param key:
    :param dtype:
    :return:
    """

    res = np.array([x.get(key, None) for x in list_of_dicts])
    if dtype is not None:
        res = res.astype(dtype)

    return res


def list_of_dict_to_dict_of_features(list_of_dicts: list[dict], dtype=None) -> dict[str, np.ndarray]:
    """
    This function will convert list of dictionaries into a dictionary of features, where each key is a name of feature
    from the original item-dictionary, and value is a Numpy array of feature values of all items.

    The first dictionary is taken as a 'template', from which the list of keys (features) is extracted.

    :param list_of_dicts:
    :return:
    """

    if len(list_of_dicts) <= 0:
        raise ValueError("list_of_dicts can NOT be empty! Features can not be determined.")

    d0 = list_of_dicts[0]
    features = d0.keys()

    result = {}
    for feat_name in features:
        result[feat_name] = extract_feature_values(list_of_dicts, key=feat_name, dtype=dtype)


    return result


