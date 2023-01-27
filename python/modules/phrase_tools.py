import re

from parameters.common_parameters import CommonParameters

split_char = CommonParameters.split_char


class CheckBaseWord:
    def __init__(self, keytype):
        self.keytype = keytype

    def get_baseword(self, phrase):

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
