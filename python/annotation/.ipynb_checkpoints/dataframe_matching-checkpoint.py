import re

import numpy as np
import pandas as pd

from python.annotation import tokenizertools
from python.parameters.common_parameters import CommonParameters

split_char = CommonParameters.split_char


class DataFrameMatching:

    def __init__(self, df_src):
        """Initializer

        Args:
            df_src (pd.DataFrame): Token lines of Webanno TSV
        """
        self.df_src = df_src
        self.df_src_lemma = self.df_src[self.df_src['lemma'] != '']
        self.columns = ['pid', 'tid', 'start', 'end', 'lemma', 'token']

    @staticmethod
    def lemma_text(df_src):
        return split_char.join(
            [token for token in df_src.lemma.tolist() if token])

    @staticmethod
    def raw_text(df_src):
        return tokenizertools.token_to_text(df_src)


class DataFrameMatchingPhrase(DataFrameMatching):
    def __init__(self, searh_phrase, df_src):
        super().__init__(df_src)
        # Remove the last _
        self.searh_phrase = searh_phrase.rstrip(split_char)
        self.search_phrase_list = self.searh_phrase.split(split_char)
        self.len_search_phrase = len(self.search_phrase_list)

    @property
    def first_indexes(self, ):
        return self.df_src_lemma[self.df_src_lemma['lemma'] == self.search_phrase_list[0]].index

    def check_match_with_phrase(self, first_index, last_truncated=False):
        matches = []
        # Check from the start to previous of the last
        for i in range(1, self.len_search_phrase - 1):
            search = self.search_phrase_list[i]
            text = self.df_src_lemma.shift(-i).loc[first_index, 'lemma']
            if not text:
                continue
            matches.append(search == text)

        # Check the last token
        j = self.len_search_phrase - 1
        text = self.df_src_lemma.shift(-j).loc[first_index, 'lemma']
        if isinstance(text, str):
            if last_truncated:
                matches.append(self.search_phrase_list[j].startswith(text))
            else:
                matches.append(self.search_phrase_list[j] == text)
        else:
            matches.append(False)

        return all(matches)

    # @stop_watch
    def lemma_matched_tokens(self, first_index, add_tag='sub', last_truncated=False):
        if self.check_match_with_phrase(first_index, last_truncated):
            idx_df_src_end = self.df_src.index[-1]
            idx_array = self.df_src_lemma.index.values
            new_idx = np.where(idx_array == first_index)[0] + self.len_search_phrase - 1
            end_index = idx_array[new_idx][0]

            if add_tag:
                if idx_df_src_end >= end_index + 1 \
                        and add_tag in self.df_src.loc[end_index + 1, 'style_tag']:
                    add_idx = 5 if idx_df_src_end > end_index + 5 else idx_df_src_end - end_index
                    # print(first_index, idx_df_src_end, new_idx, end_index)

                    df_style = self.df_src.loc[end_index + 1: end_index + add_idx, 'style_tag'].copy()
                    # print(df_style, df_style[~df_style.str.contains(add_tag)])
                    add_indexes = df_style[~df_style.str.contains(add_tag)].index
                    if len(add_indexes):
                        end_index = add_indexes[0] - 1
                    else:
                        sub_indexes = df_style[df_style.str.contains(add_tag)].index
                        end_index += len(sub_indexes)

            # Add symboles before a number
            symbols = '- ≈ ∼ ~ < > ⩽ ⩾ ≤ ≥ \+ ±'.split()
            if self.df_src.loc[first_index, 'lemma'] == '0':
                _token = self.df_src.loc[first_index - 1:first_index, 'token'].iloc[0]
                prev_token = tokenizertools.replace_simchars([_token])
                # print(prev_token[0], prev_token[0] in symbols)
                if prev_token[0] in symbols:
                    first_index -= 1
                    # print(self.df_src.loc[first_index: end_index, self.columns])

            return self.df_src.loc[first_index: end_index, self.columns]
        else:
            return pd.DataFrame()

    def lemma_matched_tokens_all(self, add_tag='sub', out='dataframe', check_phrase=False, last_truncated=False):
        """

        Args:
            add_tag:
            out:
            check_phrase:

        Returns:
            (pid, tids, lemma_text, raw_text)
        """
        indexes = self.first_indexes

        if out == 'dataframe':
            results = pd.DataFrame()
        else:
            results = []

        for i, index in enumerate(indexes):
            df_result = self.lemma_matched_tokens(index, add_tag, last_truncated)
            if df_result.empty:
                continue

            lemma_text = self.lemma_text(df_result)

            if out == 'dataframe':
                df_result['group'] = f'{self.searh_phrase}-{i}'
                new_columns = ['group'] + self.columns
                # results = results.append(df_result[new_columns])
                results = pd.concat([results, df_result[new_columns]])

            else:
                pid = df_result.iloc[0, 0]
                tids = '-'.join([str(df_result.iloc[0, 1]), str(df_result.iloc[-1, 1])])
                raw_text = self.raw_text(df_result)

                results.append((pid, tids, lemma_text, raw_text))

        return results


class DataFrameMatchingRe(DataFrameMatching):

    def lemma_to_token(self, re_phrase, out='dataframe', add_tag='sub', add_groupdict=False, groupdict_raw=False,
                       check_phrase=False, last_truncated=False):
        """Find raw tokens with matched lemmartized tokens

        Args:
            re_phrase:
            out
            add_groupdict:

        Returns:

        """
        matches = re.finditer(re_phrase, self.lemma_text(self.df_src))
        tupls = [(match.group(), match.groupdict()) for match in matches]

        phrases = [tupl[0] for tupl in tupls]
        if not phrases:
            if out == 'dataframe':
                pd.DataFrame()
            else:
                return []
        set_phrases, idx = np.unique(phrases, return_index=True)

        grpdict = [tupl[1] for tupl in tupls]
        if grpdict:
            set_grpdict = np.array(grpdict)[idx]
        else:
            set_grpdict = [{}] * len(set_phrases)

        if out == 'dataframe':
            results = pd.DataFrame()
        else:
            results = []

        for i, (phrase, grpdict) in enumerate(zip(set_phrases, set_grpdict)):

            m = DataFrameMatchingPhrase(phrase, self.df_src)

            if out == 'dataframe':
                df_result = m.lemma_matched_tokens_all(out='dataframe', add_tag=add_tag,
                                                       check_phrase=check_phrase, last_truncated=last_truncated)
                if not df_result.empty:
                    # TODO: check add groupdict
                    if add_groupdict:
                        for label, phrase in grpdict.items():
                            if phrase:
                                mdict = DataFrameMatchingPhrase(phrase, df_result)
                                df_dic = mdict.lemma_matched_tokens_all(out='dataframe', add_tag=add_tag,
                                                                        check_phrase=check_phrase,
                                                                        last_truncated=last_truncated)
                                df_dic['dict'] = label
                                df_result = df_result.merge(df_dic, on='token', how='left', suffixes=['', '_y'])

                    rm_columns = df_result.filter(regex='_y$', axis=1).columns
                    results = results.append(df_result.drop(rm_columns, axis=1))
            else:
                result_all = m.lemma_matched_tokens_all(out='tuple', add_tag=add_tag, check_phrase=check_phrase,
                                                        last_truncated=last_truncated)

                for result in result_all:
                    if add_groupdict:
                        grpdict_clean = {k: v for k, v in grpdict.items() if v}
                        if groupdict_raw and len(grpdict_clean):
                            grpdict_clean = self.groupdict_token(result, grpdict_clean)

                        new_result = tuple(list(result) + [str(grpdict_clean)])
                        results += [new_result]

        return results

    def groupdict_token(self, result, grpdict_clean):
        """
        Replace lemma token to raw token in groupdict
        Args:
            result(tuple): Result of lemma_matched_tokens_all
            grpdict_clean: Result of groupdict except None values

        Returns:

        """
        pid, tids, lemma, phrase = result
        stid, etid = map(int, tids.split('-'))
        df_match = self.df_src[(stid <= self.df_src['tid']) & (self.df_src['tid'] <= etid)]

        esc_dict = {key: tokenizertools.escape_symbols(val).rstrip(split_char) for key, val in grpdict_clean.items()}
        expressions = [f'(?P<{key}>{value})' for key, value in esc_dict.items()]
        all_expression = '(?P<gg1>.*?)'.join(expressions)
        num_extra = all_expression.count('P<gg1>')
        for i in range(num_extra):
            all_expression = all_expression.replace('P<gg1>', f'P<gg{i}>', 1)

        new_match = re.match(all_expression, lemma)
        if new_match:
            new_gdict = new_match.groupdict()

            keys, values = new_gdict.keys(), new_gdict.values()
            array = [*map(lambda x: len([y for y in x.split(split_char) if y]), values)]
            array_e = np.cumsum(array)
            array_s = array_e.copy()
            array_s[1:] = array_e[:-1]
            array_s[0] = 0

            ivac = df_match.index[0]
            indexes = [i for i, lemma in zip(df_match.index, df_match['lemma']) if lemma == '']
            indexes_sub = [i for i, style in zip(df_match.index, df_match['style_tag']) if 'sub' in style]
            results = []
            for v1, v2 in zip(array_s, array_e):
                new_v1 = v1 + ivac
                new_v2 = v2 + ivac
                for index in indexes:
                    if new_v1 <= index <= new_v2:
                        new_v2 += 1
                        ivac += 1

                # add sub tag tokens
                for index_sub in indexes_sub:
                    if new_v2 == index_sub:
                        new_v2 += 1

                results.append((new_v1, new_v2))

            return {k: self.raw_text(df_match.loc[v1:v2 - 1])
                    for k, (v1, v2) in zip(keys, results) if not k.startswith('gg') and not df_match.loc[v1:v2 - 1].empty}
