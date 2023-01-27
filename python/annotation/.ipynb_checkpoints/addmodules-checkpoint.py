import re
from itertools import chain

import pandas as pd

from python.annotation.tsv_components import WebannoLabel, WebAnnoTSV
from python.annotation.normalization import pretreatment


def combine_label(current_label, add_label):
    current_components = WebannoLabel(current_label).components(only_label=True)
    add_components = WebannoLabel(add_label).components(only_label=True)

    if current_label == '_':
        return add_label
    elif add_components not in current_components:
        return '|'.join([current_label, add_label])
    else:
        return current_label


def combine_other_label(current_other_label, num):
    new_label = '*' + f'[{num}]' if num else '*'
    if current_other_label == '_':
        return new_label
    else:
        return '|'.join([current_other_label, new_label])


class LayerTagset:
    def __init__(self, layer, tags):
        """
        Layer, tagset one line
        """
        self.layer = layer
        self.tags = sorted(tags)

    @property
    def out_tuple(self, ):
        return self.layer, self.tags

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.layer == other.layer, self.tags == other.tags])


class DFspanset:
    def __init__(self, df_spanset):
        self.df_spanset = df_spanset

    @staticmethod
    def sort_tuple(tupl):
        a = LayerTagset(*tupl)
        return a.out_tuple

    @property
    def out_tuples(self, ):
        layer_tags = sorted([tuple(val) for val in self.df_spanset.values], key=lambda x: x[0])
        return [*map(self.sort_tuple, layer_tags)]

    @property
    def out_df(self, ):
        return self.df_spanset

    def other_tags(self, tag):
        tag_list = self.df_spanset['tagset'][self.df_spanset['tagset'].apply(''.join).str.contains(tag)]
        ds = tag_list.apply('|'.join).str.split('|', expand=True).iloc[0]
        return ds[~ds.str.contains(tag)].tolist()

    @staticmethod
    def add_tuples(tupls1, tupls2):
        """
        Add new layer_tags_tuples excluding duplicates using pd.DataFrame
        """
        all_tuples = tupls1 + tupls2

        def serialize_tags(tupl):
            return [tupl[0]] + tupl[1]

        def restore_tuple(serialized_tags):
            return serialized_tags[0], [x for x in serialized_tags[1:] if x]

        df = pd.DataFrame(map(serialize_tags, all_tuples))
        all_tuple_list = df.drop_duplicates().values.tolist()
        return [*map(restore_tuple, all_tuple_list)]

    def add_spanset(self, layer_tagset_tuples):
        all_tuples = self.add_tuples(self.out_tuples, layer_tagset_tuples)
        out_tuples = [*map(self.sort_tuple, sorted(all_tuples, key=lambda x: x[0]))]
        self.df_spanset = pd.DataFrame(out_tuples, columns=['layer', 'tagset'])

    @property
    def new_columns(self, ):
        return [*chain.from_iterable(self.df_spanset['tagset'])]


class DFrelationset:
    def __init__(self, df_relationset):
        self.df_relationset = df_relationset

    @property
    def out_tuples(self, ):
        layer_tags = sorted([tuple(val) for val in self.df_relationset.values], key=lambda x: x[0])
        return layer_tags

    @property
    def out_df(self, ):
        return self.df_relationset

    def add_relationset(self, layer_tag_link_tuples, out_type='dataframe'):
        all_tuples = sorted((self.out_tuples + layer_tag_link_tuples), key=lambda x: x[0])
        self.df_relationset = pd.DataFrame(all_tuples, columns=['layer', 'tagset', 'link'])

    @property
    def new_columns(self, ):
        return [*chain.from_iterable(self.df_relationset[['tagset', 'layer']].values.tolist())]


class AddLayers(WebAnnoTSV):
    """
    def __init__(self, df_spanset, df_relationset, df_tokenlines)
    """

    def add_blank_tag(self, layer_tagset_tuples, layer_tag_link_tuples, clear=False):
        span = DFspanset(self.df_spanset)
        span.add_spanset(layer_tagset_tuples)
        self.df_spanset = span.out_df

        relation = DFrelationset(self.df_relationset)
        relation.add_relationset(layer_tag_link_tuples)
        self.df_relationset = relation.out_df

        columns = self.df_tokenlines.columns.tolist()
        head_columns = self.df_tokenlines.columns[:5].tolist()
        tag_columns = span.new_columns + relation.new_columns
        all_columns = head_columns + tag_columns
        new_columns_set = set(all_columns) - set(columns)

        df_new_tokenlines = self.df_tokenlines.copy()
        for column in new_columns_set:
            df_new_tokenlines[column] = ['_'] * len(df_new_tokenlines)

        if clear:
            span_tags = set(chain.from_iterable([tupl[1]
                                                 for tupl in layer_tagset_tuples]))
            relation_tags = set(chain.from_iterable([tupl[1]
                                                     for tupl in layer_tag_link_tuples]))
            for column in span_tags | relation_tags:
                df_new_tokenlines[column] = ['_'] * len(df_new_tokenlines)

        self.df_tokenlines = df_new_tokenlines[all_columns]

    def spanset_except_tag(self, tag):
        return DFspanset(self.df_spanset).other_tags(tag)

    def add_label(self, pid, tids, add_column, add_label, num=None):
        if num:
            add_label += f'[{num}]'
        s, e = tids
        if e - s > 1 and not num:
            print('Check number!')

        # Dummy
        _pid = pid
        indexes = self.df_tokenlines.query('(pid == @_pid) & (@s <= tid <= @e)').index
        for index in indexes:
            current_label = self.df_tokenlines.loc[index, add_column]
            self.df_tokenlines.loc[index, add_column] = combine_label(
                current_label, add_label)

            # add * to other columns in the same layer
            other_columns = self.spanset_except_tag(add_column)
            if other_columns:
                for other_column in other_columns:
                    current_other_label = self.df_tokenlines.loc[index,
                                                                 other_column]
                    new_other_label = combine_other_label(
                        current_other_label, num)
                    self.df_tokenlines.loc[index,
                                           other_column] = new_other_label

    def pre_treatment(self, ):
        self.df_tokenlines = pretreatment(self.df_tokenlines)

    def drop_lemma(self, ):
        if 'lemma' in self.df_tokenlines.columns:
            self.df_tokenlines.drop('lemma', axis=1, inplace=True)

    @staticmethod
    def extract_relation_numbers(tsv_labels, anno_label):
        """
        Extract common numbers
        """
        match_labels = [[label for label in labels.split('|') if anno_label in label] for labels in tsv_labels]
        matches = [*map(lambda x: re.finditer(r'\[(\d+)\]', x), map(lambda x: '|'.join(x), match_labels))]
        numbers = [[m.group(1) for m in match] for match in matches]
        # print(numbers, set(numbers[0]).intersection(*[set(x) for x in numbers]))
        return set(numbers[0]).intersection(*[set(x) for x in numbers])

    def get_label_number(self, index, label):
        """
        Get label number
        """
        pid, stid, etid = index
        parent = label.split('-')[0]
        column = parent.lower() + '_tag'
        anno_label = label.split('-')[-1]

        if etid - stid < 0:
            return None
        elif etid - stid == 0:
            number = '0'
        else:
            df_target_tsv = self.df_tokenlines[(self.df_tokenlines['pid'] == pid) &
                                               (self.df_tokenlines['tid'] >= stid) & (
                                                           self.df_tokenlines['tid'] <= etid)]
            df_tsv_labels = df_target_tsv[column]
            number = self.extract_relation_numbers(df_tsv_labels, anno_label)

        if number:
            return anno_label, list(number)[-1]

    def add_relation(self, src_index, src_labels, dst_index, dst_labels, add_column_label, add_column_link):
        """
        Add relation in TSV
        Args:
            src_index(tuple): source index (pid, (stid, etid))
            src_column(str): source column for label
            dst_index(tuple): destination index
            dst_column(str): destination column for label
            add_column_label: 'relation_tag'
            add_column_link: 'webanno.custom.Relations'

        Returns:
        """

        src_pid, src_stid, src_etid = src_index
        src_label, src_num = self.get_label_number(src_index, src_labels)
        dst_label, dst_num = self.get_label_number(dst_index, dst_labels)

        if src_num and dst_num:
            add_index = self.df_tokenlines[(self.df_tokenlines['pid']==dst_index[0]) &
                                           (self.df_tokenlines['tid']==dst_index[1])].index

            # Add link info in the first line of distination
            link_info = f'{src_pid}-{src_stid}[{src_num}_{dst_num}]'
            self.df_tokenlines.loc[add_index, add_column_link] = link_info
            link_label = f'{src_label}-{dst_label}'
            self.df_tokenlines.loc[add_index, add_column_label] = link_label
