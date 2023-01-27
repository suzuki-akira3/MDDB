import re
import unicodedata
import functools

import numpy as np

from python.annotation import tokenizertools
from python.parameters.common_parameters import CommonParameters

split_char = CommonParameters.split_char


class Normalization:

    def __init__(self, df_src):
        self.df_src = df_src

    @property
    def dataframe(self, ):
        return self.df_src

    @property
    def token_list(self):
        return self.df_src.token.tolist()

    @property
    def lemma_list(self):
        return self.df_src.lemma.tolist()

    def add_lemma(self):
        # WhitespaceTokenizer -> lemmatized_text -> normalize_chars -> make_tokens_lower
        lemma_list = tokenizertools.lemmatized_text(self.df_src.token.tolist())
        self.df_src['lemma'] = lemma_list

    def remove_tag(self, tagsets):

        for tagset in tagsets:
            for column, taglist in tagset.items():
                esc_tag_string = '|'.join(map(tokenizertools.escape_symbols, taglist))
                # esc_tag_string = token_tools.escape_symbols(tag_string)
                re_tag = re.compile(rf'({esc_tag_string}\d*)(\[\d+\])?')
                # print(self.df_src.loc[self.df_src[column].str.match(re_tag, na=False), ['token', 'lemma']])
                self.df_src.loc[self.df_src[column].str.match(re_tag, na=False),
                                'lemma'] = ''

    def remove_token(self, rm_tokens):
        re_not_sup = re.compile(r'(?!sup(\[\d+\])?)')

        # Remove 's
        index = self.df_src[(self.df_src['lemma'] == "'") & (self.df_src.shift(-1)['lemma'] == 's')].index
        if not index.empty:
            indexes = np.concatenate([index.values, index.values + 1])
            self.df_src.loc[indexes, 'lemma'] = ''

        if isinstance(rm_tokens, list):
            rm_token_text = '|'.join(sorted(
                map(tokenizertools.escape_symbols, rm_tokens),
                key=len, reverse=True))
            re_rmtoken = re.compile(rf'^({rm_token_text})$')

            self.df_src.loc[self.df_src['style_tag'].str.match(re_not_sup, na=False),
                            'lemma'] = self.df_src.loc[
                self.df_src['style_tag'].str.match(re_not_sup, na=False),
                'lemma'].str.replace(re_rmtoken, '')

    def normalize_number(self, ):
        """Make all numbers to 0 except those with sup tag

        Returns:

        """
        re_sup_tag = re.compile(r'(?!sup(\[\d+\])?)')
        re_number = re.compile(r'^\d+(.\d+)?$')
        self.df_src.loc[self.df_src['style_tag'].str.match(re_sup_tag, na=False),
                        'lemma'] = self.df_src.loc[
            self.df_src['style_tag'].str.match(re_sup_tag, na=False),
            'lemma'].str.replace(re_number, '0')

    def normalize_word(self, dic):
        self.df_src['lemma'] = self.df_src['lemma'].replace(dic)

    def unicode_normalize(self):
        uni_norm = functools.partial(unicodedata.normalize, "NFKC")
        self.df_src['lemma'] = self.df_src['lemma'].apply(uni_norm)

    def make_abbrev_single(self, ):
        """ Make abbreviation words to single (AAAs -> AAA)

        Returns:

        """

        def make_single(term):
            return re.sub(r'([A-Z]{2,})s', r'\1', term)

        self.df_src['lemma'] = self.df_src['lemma'].apply(make_single)

    def out_df(self):
        return self.df_src


def pretreatment(df_src):
    rm_tags = CommonParameters.rm_tags
    rm_token = CommonParameters.rm_tokens
    dic = CommonParameters.dic_normalize

    df_class = Normalization(df_src)
    if not 'lemma' in df_src.columns.to_list():
        df_class.add_lemma()

    df_class.remove_tag(rm_tags)
    df_class.remove_token(rm_token)
    df_class.normalize_number()
    df_class.unicode_normalize()
    df_class.normalize_word(dic)

    # AAAs -> AAA
    df_class.make_abbrev_single()

    return df_class.out_df()


def replace_tokens(token_list, old_tokens, new_tokens):
    """
    Replace token split manually
    :param token_list:
    :param old_tokens:
    :param new_tokens:
    :return:
    """
    token_list_set = set(token_list)
    search_tokens = old_tokens.split(split_char)
    search_token_set = set(search_tokens)
    len_search_tokens = len(search_tokens)

    if search_token_set <= token_list_set:
        first_indexes = [i for i, token in enumerate(token_list) if token == search_tokens[0]]

        for first_index in first_indexes[::-1]:
            if first_index > len(token_list) - len_search_tokens:
                continue

            check = [token_list[first_index + i] == search_tokens[i] for i in range(1, len_search_tokens)]
            if not all(check):
                continue

            # print(first_index, token_list[first_index:first_index+len_search_tokens])
            new_token_list = token_list[:first_index] + [new_tokens] + token_list[first_index + len_search_tokens:]
            # print(new_token_list[first_index-1:first_index + len_search_tokens +1])
            token_list = new_token_list

    return token_list
