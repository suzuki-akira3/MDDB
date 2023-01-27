import re

import numpy as np

from python.parameters.common_parameters import CommonParameters

split_char = CommonParameters.split_char


class CheckBaseWord:
    def __init__(self, keytype):
        self.keytype = keytype

    def get_baseword(self, phrase):

        # Skip tokens for keytype = 'first'
        skip_tokens = CommonParameters.skip_tokens

        if split_char not in phrase:
            return phrase

        if self.keytype in ['base', 'abbrev', 'synonym']:
            if phrase[-1] in [')', '|', '}', ']']:
                if self.keytype in ['base']:
                    match = re.match(r'.*?[(|{\[]_([^()|{}]+?)_[)|}\]]$', phrase)
                else:
                    # 'coercivity_(_0_Oe_)' ?
                    match = re.match(r'[^()]+?\(_([^()]+?)_\)$', phrase)
                if match:
                    return match.group(1).split(split_char)[-1]
            else:
                return phrase.split(split_char)[-1]
        elif self.keytype in ['first', 'abbrev2', 'synonym2']:
            if phrase[0] in ['(', '|', '{', '[']:
                if self.keytype in ['first']:
                    match = re.match(r'^[(|{\[]_([^()|{}]+?)_[)|}\]].*?', phrase)
                else:
                    match = re.match(r'^\(_([^()]+?)_\)_[^()]+?', phrase)
                if match:
                    return match.group(1).split(split_char)[0]
            else:
                first = phrase.split(split_char)[0]
                if first in skip_tokens:
                    return phrase.split(split_char)[1]
                else:
                    return first
        else:
            return None

    def has_baseword(self, baseword, phrase):
        """
        return True(bool) if phrase contains a baseword
        """
        return baseword == self.get_baseword(phrase)

    def get_baseword_indexes(self, phrase):
        """
        return indexes(list) of baseword in a phrase
        """
        baseword = self.get_baseword(phrase)
        return [i for i, token in enumerate(phrase.split(split_char))
                if token in baseword.split(split_char)]

    def match_baseword(self, baseword, phrases):
        """
        return phrases(list) with a baseword
        """
        return [phrase for phrase in phrases
                if self.has_baseword(baseword, phrase)]

    def check_in_bw_list(self, phrase, basewords=None) -> bool:
        if basewords is not None:
            baseword = self.get_baseword(phrase)
            if baseword:
                return baseword in basewords

    def get_term(self, ):
        """
        Get a term of abbreviation
        :return:
        """
        pass


def balanced_blacket(phrase, brackets):
    """Check unless the number of brackets is odd
        No brackets in phrase, or both '(' and ')' in a phrase and the order is '(', ')'
    Args:
        phrase (str): Phrase extracted from articles
        brackets (str): Pairs of brackets

    Returns:
        (bool)
    """
    #
    if set(brackets).isdisjoint(set(phrase)) or \
            (set(brackets) <= set(phrase) and list(brackets) == [x for x in phrase if x in brackets]):
        return True


def check_balanced_blacket(phrase):
    """Check if brackets in a phrase are correct

    Args:
        phrase (str): Phrase extracted from articles

    Returns:
        (bool)
    """
    round_br = balanced_blacket(phrase, '()')
    square_br = balanced_blacket(phrase, '[]')
    angle_br = balanced_blacket(phrase, '⟨⟩')
    curly_br = balanced_blacket(phrase, '{}')
    return all([round_br, square_br, angle_br, curly_br])


def remove_stop_phrase(phrases, base_phrase=False, keytype='base'):
    # keytype = MatDicParameters.keytype

    ng_list_top = MatDicParameters.ng_list_top
    ng_list_next = MatDicParameters.ng_list_next
    ng_list_prev = MatDicParameters.ng_list_prev
    ng_list_bottom = MatDicParameters.ng_list_bottom

    def check_ng(phrase):
        if split_char not in phrase or re.search(rf'{split_char}{split_char}+', phrase) \
                or phrase.startswith(split_char):
            return False
        else:
            return True

    def check_base(phrase):
        word_list = phrase.split(split_char)
        if keytype in ['base', 'abbrev']:
            if len(word_list) > 1:
                top_word = word_list[0]
                return not (top_word in ng_list_top)
            else:
                return False
        elif keytype in ['first']:
            if len(word_list) > 1:
                bottom_word = word_list[-1]
                return not (bottom_word in ng_list_bottom)
            else:
                return False

    def check_next(phrase):
        word_list = phrase.split(split_char)
        if keytype in ['base', 'abbrev']:
            if len(word_list) > 1:
                next_word = word_list[1]
                return not (next_word in ng_list_next)
            else:
                return False
        elif keytype in ['first']:
            if len(word_list) > 1:
                prev_word = word_list[-2]
                return not (prev_word in ng_list_prev)
            else:
                return False

    balanced = np.array([*map(check_balanced_blacket, phrases)])
    ngchecked = np.array([*map(check_ng, phrases)])
    ngbase = np.array([*map(check_base, phrases)])
    ngnext = np.array([*map(check_next, phrases)]) if base_phrase \
        else np.array([True] * len(phrases))

    check_ok = (balanced & ngchecked & ngbase & ngnext)
    new_phrases = np.array(phrases)[check_ok].tolist()

    # Exceptional
    if '(_BH_)' in phrases:
        new_phrases.append('(_BH_)')
    elif '(_B_H_)' in phrases:
        new_phrases.append('(_B_H_)')

    return new_phrases
