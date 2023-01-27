from functools import reduce
import pandas as pd
import numpy as np
from python.parameters.common_parameters import CommonParameters


split_char = CommonParameters.split_char


def split_sentences(df_src):
    """
    Split DataFrame by sentence
    """
    with_space = (df_src.shift(-1)['start'] > df_src['end'])
    with_space_idx = with_space[with_space].index

    next_capitals = df_src.shift(-1)['token'].str.match(r'^[A-Zμ].*?$')
    next_capitals_idx = next_capitals[next_capitals == True].index

    period_idx = df_src[df_src['token'] == '.'].index

    sep_idx = reduce(np.intersect1d, (period_idx, with_space_idx, next_capitals_idx))

    idx_number = df_src.index.isin(sep_idx)
    if any(idx_number):
        sep_numbers = np.where(idx_number)[0] + 1
        group_idxes = np.split(df_src.index, sep_numbers)

        return (df_src.loc[indexes, :] for indexes in group_idxes)
    else:
        return (df_src,)


def remove_overlap(df_src, grouping=None, inverse=False):
    """
    Remove overlapped rows
    :param df_src:
    :param grouping:
    :param inverse:
    :return:
    """
    def extract_overlap(df_src_inloop):
        overlaps = []
        for tupl in df_src_inloop.itertuples():
            # All stid and etid except self
            ex_tid_offset = df_src_inloop.set_index(['stid', 'etid']).drop(
                [(tupl.stid, tupl.etid)]).index.values
            tids_list = [list(range(ex_tids[0], ex_tids[1] + 1)) for ex_tids in
                         ex_tid_offset]
            # Finde overlaps in df_match
            overlaps += [
                any([set(range(tupl.stid, tupl.etid + 1)) <= set(tids) for tids in tids_list])]

        return overlaps

    df_src_no_overlappped = pd.DataFrame()

    # Remove overlapped by a grouping column
    if grouping:
        for group_column, df_src_grouping in df_src.groupby(grouping):
            overlaps = extract_overlap(df_src_grouping)
            # Default
            if not inverse:
                not_overlaps = [not bool(overlap) for overlap in overlaps]
                df_src_no_overlapped_grouping = df_src_grouping.loc[not_overlaps, :]
            else:
                df_src_no_overlapped_grouping = df_src_grouping.loc[overlaps, :]

            df_src_no_overlappped = pd.concat([df_src_no_overlappped, df_src_no_overlapped_grouping])
    else:
        overlaps = extract_overlap(df_src)
        if not inverse:
            not_overlaps = [not bool(overlap) for overlap in overlaps]
            df_src_no_overlappped = df_src.loc[not_overlaps, :]
        else:
            df_src_no_overlappped = df_src.loc[overlaps, :]

    return df_src_no_overlappped.sort_values(['stid'])
