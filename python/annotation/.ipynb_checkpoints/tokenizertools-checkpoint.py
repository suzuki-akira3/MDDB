import re
from functools import lru_cache

import numpy as np
import pandas as pd

from python.annotation import spacy_tokenizer
from python.parameters.common_parameters import CommonParameters


split_char = CommonParameters.split_char


class SimilarSymbols:
    def __init__(self):
        self.hyphens = (
            '\u002d',
            set('\u2010\u2011\u2012\u2013\u2014\u2015\u2043\u02d7\u207b\u2212\u00ad\uff0d'),
        )
        self.slashes = ('\u002f', set('\u2044\u2215'))
        self.varbar = ('\u007c', set('\u2223'))
        self.aposts = ('\u0027', set('\u2019\u055a\u05F3\u2032\ua78b\uff07'))
        self.pluses = ('\u002b', set('\uff0b\u207a'))
        self.nearly = ('\u007E', set('\u223C\u2243\u2248'))
        self.less = ('\u003C', set('\u2A7D\u2264'))
        self.greater = ('\u003E', set('\u2A7E\u2265'))
        

class PeriodicTable:
    # re_elements = re.compile(
    #     r'(H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|'
    #     r'K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|'
    #     r'Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|'
    #     r'Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
    #     r'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Lv)'
    # )
    def __init__(self):
        self.periodic = pd.read_csv('python/annotation/periodic_table.csv', encoding='utf-8')

    def __repr__(self):
        return str(self.periodic)

    @property
    def all_elements(self):
        return self.periodic.Element.tolist()

    @property
    def re_elements(self):
        return re.compile('|'.join(self.periodic.Element.tolist()))

    @property
    def element_name(self):
        return {tupl.Element: tupl.en for tupl in self.periodic.itertuples()}

    def use_elements(self, use_list):
        use_index = self.periodic.query(f'Element in {use_list}').index
        self.periodic = self.periodic.iloc[use_index]

    def remove_elements(self, rm_list):
        drop_index = self.periodic.query(f'Element in {rm_list}').index
        self.periodic.drop(drop_index, inplace=True)


def replace_simchars(terms):
    """replace simchar <- should be first step

    Args:
        terms (list): token text

    Returns:
        list: replaced text
    """

    sim = SimilarSymbols()
    simchar_map = {rep_symbol:tupl[0]
                   for tupl in sim.__dict__.values()
                   for rep_symbol in tupl[1]}

    def replace_term(term):
        return ''.join([simchar_map.get(char) or char
                        for char in term])

    return [replace_term(term)
            if any([char in list(simchar_map.keys()) for char in term]) else term
            for term in terms]


@lru_cache(maxsize=1048576)
def can_lemmatize(token):
    """Lemmatizeable

    Args:
        token (str): A single token

    Returns:
        bool: True if text can lemmatize
    """
    periodic = PeriodicTable()
    re_elements = periodic.re_elements
    # TODO: Check all units
    units = [
                'Pa', 'Oe', 'Wb', 'Gy', 'Sv', 'Gal', 'St', 'Da', 'Ns', 'Nor', 'Hz', 'Gs',
                'Np', 'Ws', 'Wh', 'Ci', 'Dptr', 'Bq', 'Mb'
            ] + [
                'Ln', 'Nu', 'Re', 'Ma', 'Pr', 'Le', 'Kn', 'Gr', 'Fr'
            ] + ['Di', 'Cp', 'Ms', 'Mr', 'Hc', 'Br']
    units_str = '|'.join(units)
    re_units = re.compile(f'{units_str}')

    return token[1:].islower() and \
           not re.fullmatch(re_elements, token) and \
           not re.fullmatch(re_units, token) and \
           '-' not in token


def lemmatized_text(terms):
    """Lemmatize

    Args:
        terms (list): List of terms

    Returns:
        list: Lemmatized terms
    """

    def out(lemma, token, cond):
        """

        Args: Select raw or lemmatized token
            lemma: lemmatized tokens
            token: raw tokens
            cond (bool): Condition

        Returns:
            str: lemmatized token
        """
        return lemma if cond else token

    def lower(lemma, cond):
        """

        Args: Select raw or lemmatized token
            lemma: lemmatized tokens
            cond (bool): Condition

        Returns:
            str: lower token
        """
        return lemma.lower() if cond else lemma

    # TODO: move replace \xad in tokenizer
    terms = replace_simchars([term.replace('\xad', '-') for term in terms])

    # Make matched tokens lower
    can_lower = np.array([can_lemmatize(token) for token in terms])
    terms_lower = np.frompyfunc(lower, 2, 1)(terms, can_lower)

    doc = spacy_tokenizer.tokenize_by_ws(' '.join(terms_lower))
    token_lemmas = [token.lemma_ for token in doc]

    can_pos = np.array([token.pos_ != 'VERB' for token in doc])
    can_lemma = can_pos & can_lower

    # Lemmatize can_lemma tokens else raw tokens
    return np.frompyfunc(out, 3, 1)(token_lemmas, terms, can_lemma)


def token_to_text(df_src) -> str:
    """
    Get paragraph using tokens and offsets
    Args:
        df_src: DataFrame should contain ['start', 'end', 'token']

    Returns:
        Paragraph text
    """
    df_src = df_src.reset_index()
    start = df_src.loc[df_src.index[0], 'start']
    end = df_src.loc[df_src.index[-1], 'end']

    text_list = [' '] * (end - start)
    for tupl in df_src.itertuples():
        s = tupl.start - start
        e = tupl.end - start
        text_list[s:e] = tupl.token

    return ''.join((text_list))


def lemma_to_text(df_src) -> str:
    """
    Get paragraph using tokens, lemma and offsets
    Args:
        df_src: DataFrame

    Returns:
        Paragraph text
    """
    df_src_copy = df_src.copy()

    # Replace hyphens to space
    sim = SimilarSymbols()
    hyphen_indexes = df_src_copy[
        (df_src_copy['token'].isin([sim.hyphens[0]] + list(sim.hyphens[1])))
        & (~df_src_copy['style_tag'].str.contains(r'sup'))].index
    df_src_copy.loc[hyphen_indexes, 'lemma'] = ' '

    token_len = df_src_copy['token'].apply(len)
    lemma_len = df_src_copy['lemma'].apply(len)
    diff = token_len - lemma_len
    cumsum = diff.cumsum().astype(int)
    shift_cumsum = cumsum.shift().fillna(0).astype(int)

    df_src_copy['start'] -= shift_cumsum
    df_src_copy['end'] -= cumsum
    df_src_copy['token'] = df_src_copy['lemma']

    text = token_to_text(df_src_copy)
    return re.sub(r'\s+', ' ', text).strip()


def spacy_tokenizer_from_text(text):
    """Convert from text to token list. Spaces in the text was normalized in advance.

    Args:
        text (str): Text

    Returns:
        list: List of tuple(text-with-ws, lemmatized-text, pos)
    """
    doc = spacy_tokenizer.tokenize(text)
    spacy_token = [token.text_with_ws for token in doc]
    spacy_lemma = lemmatized_text([token.lemma_ for token in doc])
    spacy_pos = [token.pos_ for token in doc]

    return [*zip(spacy_token, spacy_lemma, spacy_pos)]


def escape_symbols(term):
    symbols = CommonParameters.esc_symbols
    for symbol in symbols:
        term = term.replace(symbol, '\\' + symbol)
    return term
