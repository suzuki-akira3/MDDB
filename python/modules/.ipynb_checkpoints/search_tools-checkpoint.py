import re
from itertools import chain

import numpy as np
import pandas as pd

from python.annotation import df_tools
from python.annotation.addmodules import AddLayers
from python.annotation.dataframe_matching import DataFrameMatchingRe
from python.annotation.df_tools import split_sentences
from python.annotation.read_write_tsv_file import tsvfile2dataframe
from python.parameters.search_parameters import SearchParameters, RegexMaterials


def ng_materials(df_results):
    def remove_ng(phrase):
        m = re.search(r'^0_0_0$|^\(_T_\)$|^\(_0_A_\)$|^\(_0_V_\)$|^\(_0_T_\)$', str(phrase))
        return bool(m)

    # Remove NG
    if not df_results.empty:
        indexes_ng1 = df_results[df_results['lemma'].apply(remove_ng)].index
        indexes_mat = df_results[df_results['label'] == 'materials'].index
        ng_file = 'python/parameters/materials_NG.csv'
        df_ng = pd.read_csv(ng_file, keep_default_na=False)
        indexes_ng2 = df_results[df_results['lemma'].isin(df_ng['lemma'])].index

        drop_index = indexes_mat & (indexes_ng1 | indexes_ng2)
        return df_results.drop(drop_index)


def check_materials(df_src):
    RE_in = any([any(df_src['lemma'].str.contains(elem)) for elem in RegexMaterials.RE_elements])
    required_in = all(
        [any(df_src['lemma'].str.contains(elem)) for elem in RegexMaterials.required_elements])
    return RE_in and required_in


def rm_overlapped(df_src, labels):
    df_parts = df_src[df_src['label'].isin(labels)].copy()
    idx = df_parts.index
    df_parts = df_tools.remove_overlap(df_parts)
    df_src.drop(idx, inplace=True)
    return df_append(df_src, df_parts)


def check_symbol(df_src, prop, ok, ng):
    indexes = df_src[df_src['label'] == prop].index
    for index in indexes:
        phrase = df_src.loc[index, 'phrase']
        if prop[:-3] not in phrase and (ng in phrase and
                                        (ok not in phrase and ok.upper() not in phrase)):
            df_src.drop(index, inplace=True)

    return df_src


def restruct_results(results):
    add_groupdict = SearchParameters.add_groupdict
    core_columns = ['pid', 'tids', 'label', 'lemma', 'phrase']
    columns = core_columns + ['dict'] if add_groupdict else core_columns

    df_results = pd.DataFrame(results, columns=columns)

    # Check properties
    df_results = check_symbol(df_results, 'coercivity', 'Hc', 'H')
    df_results = check_symbol(df_results, 'remanence', 'Br', 'B')
    df_results = check_symbol(df_results, 'remanence', 'Mr', 'M')
    df_results = check_symbol(df_results, 'magnetization', 'Ms', 'M')

    # Remove overlapped (materials only) and sort
    df_results['stid'] = df_results['tids'].apply(lambda x: int(x.split('-')[0]))
    df_results['etid'] = df_results['tids'].apply(lambda x: int(x.split('-')[1]))
    # Materials, Properties
    df_results = rm_overlapped(df_results, ['materials'])
    df_results = rm_overlapped(df_results, ['coercivity', 'remanence', 'BHmax',
                                            'magnetization', 'properties'])

    df_results.sort_values(['pid', 'stid', 'etid'], inplace=True)

    if not add_groupdict and 'dict' not in df_results:
        df_results['dict'] = ''

    return df_results[columns]


def add_fileinfo(df_src, publisher, csvfile):
    columns = ['publisher', 'filename'] + df_src.columns.tolist()
    df_src['publisher'] = publisher
    df_src['filename'] = csvfile.stem
    return df_src[columns]


def search_regex(match, search_phrase, check_phrase=False):
    """
    Search phrases using a regular expression
    Args:
        match(class):
        search_phrase(tuple): (search_name, re.cpmpile)
        check_phrase(bool):

    Returns:
        result(tuple): ('pid', 'tids', 'label, 'lemma', 'phrase', ('dict'))
    """
    add_groupdict = SearchParameters.add_groupdict
    result = match.lemma_to_token(search_phrase[1], out='tuple', check_phrase=check_phrase,
                                  add_groupdict=add_groupdict, groupdict_raw=True, last_truncated=True)
    return [tuple(list(tupl)[:2] + [search_phrase[0]] + list(tupl)[2:]) for tupl in result]


def search_all(match):
    """
    Search all phrases using regular expressions
    Args:
        match(class):

    Returns:
        results(list): [('pid', 'tids', 'label, 'lemma', 'phrase', ('dict'))]
    """
    results = []
    for search_phrase in SearchParameters.search_phrases:
        check_phrase = True if search_phrase[0] in SearchParameters.check_phrase_labels else False

        result = search_regex(match, search_phrase, check_phrase=check_phrase)
        if result:
            results += result
    return results


def check_property(df_src):
    properties = [search[0] for search in SearchParameters.search_phrases[:5]]
    return bool(set(properties) & set(df_src['label']))


def df_append(df_src1, df_src2):
    dic_join = {}
    dtypes = df_src2.dtypes
    columns = df_src2.columns
    for column in columns:
        src = df_src1[column] if column in df_src1.columns else []
        dic_join[column] = np.append(src, df_src2[column].values)
    return pd.DataFrame(dic_join, columns=columns).astype(dtypes)


def search_from_tsv(tsvfile):
    columns = SearchParameters.columns

    # Read WebAnnoTSV file
    df_spanset, df_relationset, df_tokenlines = tsvfile2dataframe(tsvfile)
    class_tsv = AddLayers(df_spanset, df_relationset, df_tokenlines)
    class_tsv.pre_treatment()
    df_clean = class_tsv.tokenlines

    sentences_gen = map(split_sentences, [tupl[1] for tupl in df_clean.groupby('pid')])
    sentences_gen = chain.from_iterable([list(x) for x in sentences_gen])

    class_gen = map(DataFrameMatchingRe, sentences_gen)
    results = [x for x in map(search_all, [*class_gen]) if x]
    return pd.DataFrame(chain.from_iterable(results), columns=columns[2:])


def publisher_results(df_results, df_results_publisher, publisher, tsvfile):
    parts = map(restruct_results, [tupl[1] for tupl in df_results.groupby('pid')])
    df_results_article = pd.concat([*parts]).reset_index().drop('index', axis=1)
    df_results_article = add_fileinfo(df_results_article, publisher, tsvfile)
    if not df_results_article.empty \
            and check_materials(df_results_article) \
            and check_property(df_results_article) \
            :
        df_results_publisher = df_append(df_results_publisher, df_results_article)

    return df_results_publisher
