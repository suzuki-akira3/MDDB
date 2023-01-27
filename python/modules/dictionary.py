def append_dictionary(dic, key, terms, extend=False):
    """
    appent terms into term list
    Different order from tdm-corpora !!!
    Args:
        terms (list): list of terms
        dic (dict): {key: [list of terms]}
        key: dict key
        extend (bool): True:append term, False:append list(term)

    Returns:
        dict: {key: [list of terms]}
    """
    if not extend:
        return dic.setdefault(key, list()).append(terms)
    else:
        return dic.setdefault(key, list()).extend(terms)

