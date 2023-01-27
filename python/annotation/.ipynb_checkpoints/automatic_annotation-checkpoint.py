import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import urllib.parse

from python.annotation.dataframe_matching import DataFrameMatchingPhrase
from python.parameters.common_parameters import CommonParameter
from python.annotation import df_tools

split_char = CommonParameter.split_char


def read_phrase_info(filepath):
    """
    Read phrase information from file
    """
    df_phrase_info = pd.read_csv(filepath, keep_default_na=False)
    df_phrase_info['Labels'] = df_phrase_info['Labels'].apply(eval)
    df_phrase_info['Relations'] = df_phrase_info['Relations'].apply(eval)
    data_columns = ['Phrase', 'Labels', 'Relations']
    return {phrase: (label, relation) for phrase, label, relation in df_phrase_info[data_columns].values}


def check_phrase(phrase, df_tokenlines):
    return phrase in split_char.join([lemma for lemma in df_tokenlines['lemma'].tolist() if lemma]) 


def enumerate_multi_tupl(tupls):
    """
    (a, x), (b, y), (a, z) -> (0, 0, a, x), (1, 0, b, y), (0, 1, a, z)
    """
    ds = pd.DataFrame(tupls)[[0]].value_counts()
    ind = {tupl[0]:0 for tupl in tupls}

    results = []
    for i, tupl in enumerate(tupls):   
        result = [(i, ind[tupl[0]], *tupl[0:])]
        if ds[tupl[0]] > 1:
            ind[tupl[0]] += 1

        results += result
        
    return results


def find_matched_phrases(dic_usephrase_label, df_tsv):
    # Matching process
    # From longer phrase to shorter one
    phrases = sorted(dic_usephrase_label.keys(), key=len, reverse=True)
    
    results = []
    for phrase in tqdm(phrases):
        label_relation = dic_usephrase_label.get(phrase)
        label, relation = label_relation
        match = DataFrameMatchingPhrase(phrase, df_tsv)
        result = match.lemma_matched_tokens_all(out='tupl', add_tag='sub')
        # print(phrase)
        # print(result)
        if not result:
            continue
                        
        for res in result:
            # Add label and relation for phrase
            results += [list(res) + [label, relation]]
                
    df_match = pd.DataFrame(results, columns=['pid', 'tids', 'lemma', 'phrase', 'label', 'relation'])
    if not df_match.empty:        
        # tids -> stid, etid
        df_stid = df_match['tids'].str.split('-', expand=True)
        df_match = pd.concat([df_match, df_stid], axis=1).drop('tids', axis=1)
        df_match.rename(columns={0: 'stid', 1: 'etid'}, inplace=True)
        return df_match.astype({'stid':int, 'etid':int}).fillna('')
    


def find_matched_basephrases(df_match, df_tsv):
    # Extract basephrase and label from phrases
    columns=['pid', 'stid', 'etid', 'lemma', 'phrase', 'label', 'relation']
    
    iter_results = []
    for phrase_tupl in df_match.itertuples():
        # Add label and relation of phrase
        iter_results.append([phrase_tupl.pid, phrase_tupl.stid, phrase_tupl.etid, 
                    phrase_tupl.lemma, phrase_tupl.phrase, '', []])
        

        # Dataframe of phrase
        df_phrase = df_tsv[(df_tsv['pid']==phrase_tupl.pid) & 
                           (df_tsv['tid']>=phrase_tupl.stid) & (df_tsv['tid']<=phrase_tupl.etid)]
        # print(phrase_tupl)
        # print(df_phrase)

        # Match results of each base phrase
        basephrase_labels = phrase_tupl.label
        basephrases = set([tupl[0] for tupl in basephrase_labels])
        dic_bp_result = {}
        for basephrase in basephrases:
            bp_match = DataFrameMatchingPhrase(basephrase, df_phrase)
            bp_results = bp_match.lemma_matched_tokens_all(out='tupl', add_tag='sub')
            # print('bp_results', bp_results)
            bp_results = [(bp_result[0],  int(bp_result[1].split('-')[0]),  int(bp_result[1].split('-')[1]),
                           bp_result[2], bp_result[3]) for bp_result in bp_results]
            if len(bp_results):
                dic_bp_result[basephrase] = bp_results
                
            # print('2: ', bp_results)
                
        # Find result for each base phrase
        bp_label_results = []
        new_basephrase_labels = enumerate_multi_tupl(basephrase_labels)
        for tupl in new_basephrase_labels:
            bp_index, bp_multi, bp, bp_label = tupl
            result = dic_bp_result.get(bp)   
            
            if result:
                bp_label_results += [list(result[bp_multi]) + [bp_label, []]]
            else:
                bp_label_results += []
                
            # print('3: ', [list(result[bp_multi]) + [bp_label, []]])

        # Add relation (dst, pid, tids)
        basephrase_relations = phrase_tupl.relation
        if len(basephrase_relations):
            for i, basephrase_relation in enumerate(basephrase_relations):
                src, dst = basephrase_relation
                src_ptid = bp_label_results[src]
                dst_result = bp_label_results[dst]
                
                if len(src_ptid) and len(dst_result):
                    bp_label_results[dst][-1].append(src_ptid[:3])
                    # print(bp_label_results)

        # Combine all results      
        for bp_label_result in bp_label_results:
            # print(len(columns), bp_label_result)
            if len(bp_label_result) == len(columns):
                iter_results.append(bp_label_result)
    
    return pd.DataFrame(iter_results, columns=columns)


def add_matdic_uri(tupl, num, tsv):
    lemma_term = tupl.lemma
    label = CommonParameter.base_host + urllib.parse.quote(lemma_term)
    tsv.add_label(tupl.pid, (tupl.stid, tupl.etid), 'matdic_uri', label, num)
    num += 1
    return num, tsv


def execute_addlabel(df_match, tsv):
    num_max = tsv.get_max_number
    num = num_max + 1

    for pid, df_match_pid in df_match.groupby('pid'):
        for tupl in df_match_pid.itertuples():

            # matdic_uri
            if tupl.label == '':
                num, tsv = add_matdic_uri(tupl, num, tsv)               

            # materials
            else:
                for one_label in tupl.label.split('|'):
                    add_column, add_label = one_label.split('-')
                    add_column = add_column.lower() + '_tag'
                    # print(add_column, add_label, tupl.label, tupl.lemma)

                    if add_column in tsv.tokenlines.columns:
                        if tupl.etid - tupl.stid == 0:
                            tsv.add_label(pid, (tupl.stid, tupl.etid), add_column, add_label, num=None)
                        elif tupl.etid - tupl.stid > 0:
                        # multi line
                            tsv.add_label(pid, (tupl.stid, tupl.etid), add_column, add_label, num)
                            num += 1

    return tsv


def execute_relation(df_match, tsv):
    df_match = df_match.astype({'pid': int, 'stid': int, 'etid': int, 'relation': str})
    # Only basephrase
    df_match = df_match[~(df_match['label']=='')]
    for pid, df_match_pid in df_match.groupby('pid'):
        for tupl in df_match_pid.itertuples():
            dst_pid = tupl.pid
            dst_stid = tupl.stid
            dst_etid = tupl.etid
            dst_index = (dst_pid, dst_stid, dst_etid)
            dst_labels = tupl.label

            relations = eval(tupl.relation)

            for relation in relations:
                src_pid = relation[0]
                src_stid = int(relation[1])
                src_etid = int(relation[2])
                df_src_match = df_match_pid[(df_match_pid['pid']==src_pid) & (df_match_pid['stid']==src_stid) & (df_match_pid['etid']==src_etid)]
                if df_src_match.empty:
                    continue
                src_index = (src_pid, src_stid, src_etid)
                src_labels = df_src_match['label'].iloc[0] 
                
                labels = src_labels.split('|')
                for src_label in labels:
                    if len(dst_labels.split('|')) == 1:
                        tsv.add_relation(src_index, src_label, dst_index, dst_labels, 'relation_tag', 'Relations')

    return tsv

def off_to_index(phrase, offset):
    """
    Calc indexes from offset
    """
    start, end = offset
    phrase_list = phrase.split(split_char)
    phrase_list_with_split = re.split(rf'(?<={split_char})', phrase)
    len_list = [*map(len, phrase_list_with_split)]
    s = np.cumsum(len_list) - len_list
    e = np.cumsum(len_list)
    return np.where(start <= s)[0][0], np.where(end <= e)[0][0]


def check_overlap(indexes, base_index=0):
    set_0 = set(indexes[base_index])
    check_indexes = [i for i in range(len(indexes)) if i != base_index]
    results = []
    for i in check_indexes:
        if bool(set_0.issubset(indexes[i])):
            results.append(('upperset', [indexes[i][0], indexes[i][-1]]))
        elif bool(set_0.issuperset(indexes[i])):
            results.append(('subset', [indexes[base_index][0], indexes[base_index][-1]]))
        elif bool(set_0.intersection(indexes[i])):
            results.append(('overlap', [indexes[base_index][0], indexes[i][-1]]))
        elif indexes[base_index][-1] + 1 == indexes[i][0]:
            results.append(('sequential', [indexes[base_index][0], indexes[i][-1]]))
        else:
            results.append(('disjoint', [indexes[base_index][:], indexes[i][:]]))
        
    return results


def common_labels(labels, add_qnt=False):
    """
    add_qnt: True Property-QNT and Property-Value
    """
    if len(labels) == 1:
        return None
    
    set_labels = [*map(lambda x: set(x.split('|')), labels)]
    
    if add_qnt and any([set(['Property-Value']) & set_label for set_label in set_labels]):
        for i in range(len(set_labels)):
            if 'Property-QNT' in set_labels[i]:
                set_labels[i] = set_labels[i] | set(['Property-Value'])
    
    set_label = set_labels[0]
    for i in range(1, len(set_labels)):
        set_label = set_label.intersection(set_labels[i])
    
    if set_label:
        return '|'.join(set_label)
   
    
def make_relations(src_label, dst_labels, bp_labels):
    """
    src_label: Relation from one (e.g. 'QNT')
    dist_labels: Relation to ones (e.g. ['Value', 'Condition'])
    bp_labels: [(basephrase, class-label), ]
    """
    labels = [(i, tupl[1].split('-')[-1]) for i, tupl in enumerate(bp_labels)]
    src_labels = [label for label in labels if label[1] == src_label]
    len_src_labels = len(src_labels)
    
    dic_dist_indexes = {}
    if len_src_labels:
        for dst_label in dst_labels:
            dist_indexes = [i for i, (j, l) in enumerate(labels) if l == dst_label]
            dic_dist_indexes[dst_label] = dist_indexes

        # print(src_label, dic_dist_indexes)
    
    relations = []
    for i, (j, label) in enumerate(src_labels):
        for dst_label, dst_indexes in dic_dist_indexes.items():
            len_dist = len(dst_indexes)
            # print('  ', i, label, dst_label, dst_indexes)
            if len_dist:
                # Src 1, Dst 1
                if len_dist == 1: 
                    k = dst_indexes[0] 
                    relations.append((j, k))
                # Src > 1, Dst > 1 -> (s1, d1), (s2, d2), ...
                elif len_dist > i and len_src_labels > 1:
                    k = dst_indexes[i]
                    relations.append((j, k))
                # Src == 1, Dst > 1 -> (s1, d1), (s1, d2), ...
                elif len_dist > i and len_src_labels == 1:
                    for kk in dst_indexes:
                        relations.append((j, kk))                
    return relations

def get_sentence_id(df_sentence_id, pid, stid, etid):
    df_pid =  df_sentence_id[(df_sentence_id['pid']==pid)].copy()
    df_pid.loc[:, 'stid'] = df_pid['tids'].apply(lambda x: x[0])
    df_pid.loc[:, 'etid'] = df_pid['tids'].apply(lambda x: x[-1])
    return df_pid.loc[(df_pid['stid'] <= stid) & (df_pid['etid'] >= etid), 'sid'].iloc[0]


def combine_phrase(first, second, sep=split_char):
    first_list = first.split(sep)
    second_list = second.split(sep)
    k = 0
    for i in range(len(second_list)):
        for j in range(len(first_list)):
            if first_list[j] == second_list[i]:
                k += 1

    # print(i, j, k, first_list, second_list[k:])
    return sep.join(first_list + second_list[k:])


def add_phrase_relation(relation, df_src):
    if len(relation):
        for pairs in relation:
            new_position = df_src.loc[pairs[0], ['pid', 'stid', 'etid']].values.tolist()
            row = df_src.loc[pairs[1]].copy()
            relation = row[-1]
            relation.append(new_position)
            relation = list(map(eval, set(map(str, relation))))
            row[-1] = relation
            df_src.loc[pairs[1]] = np.array(row, dtype=object)
            
    return df_src


def material_qnt_relation(df_src, material):
    materials = df_src[df_src['label']==material].index
    qnts = df_src[df_src['label']=='Property-QNT'].index
 
    material_material = []
    material_qnt = []
    if len(materials) == 1:
        material_qnt = [[materials[0], qnt] for qnt in qnts]
    elif len(materials) == 2:
        if len(qnts) == 2:
            material_qnt = [*zip(materials, qnts)]
        elif len(qnts) == 1:
            material_material = [[materials[0], materials[1]]]
            # ToDo: use distance
            material_qnt = [[materials[0], qnts[0]]]
    elif len(materials) == 3:
        if len(qnts) == 3:
            material_clas = []
            material_qnt = [*zip(materials, qnts)]
        if len(qnts) == 2:
            material_clas = [materials[0], materials[1]]
            material_qnt = [*zip(materials[1:], qnts)]
        if len(qnts) == 1:
            material_material = [[materials[0], materials[1]], [materials[1], materials[2]]]
            # ToDo: use distance
            material_qnt = [[materials[0], qnts[0]]]
    elif len(materials) > 3 and len(qnts) > 3:
            material_clas = []
            material_qnt = [*zip(materials, qnts)]        
            
    return material_material, material_qnt


def qnt_qnt_relation(df_src):
    qnts = df_src[df_src['label']=='Property-QNT'].index
    values = df_src[df_src['label']=='Property-Value'].index

    qnt_qnt = []
    qnt_value = []
    if len(qnts) == 1:
        if len(values) > 0:
            qnt_value = [(qnts[0], value) for value in values]
    elif len(qnts) == 2:
        if len(values) == 1:
            qnt_qnt = [qnts]
            qnt_value = [[qnts[1], values[0]]]
        elif len(values) == 2:
            qnt_value = [*zip(qnts, values)]

    return qnt_qnt, qnt_value


def value_condition_relation(df_src):
    values = df_src[df_src['label']=='Property-Value'].index
    conditions = df_src[df_src['label']=='Method-Condition'].index

    value_condition = []
    if len(values) == 1:
        if len(conditions) == 1:
            value_condition = [[values[0], conditions[0]]]
    elif len(values) < 1:
        if len(conditions) == 1:
            value_condition = [[value, conditions[0]] for value in values]
        elif len(values) == len(conditions):
            value_condition = [*zip(values, conditions)]

    return value_condition


def sequential_label(pid, df_src, df_sentence_id):
    """
    Define label with neighboring phrases
    """
    df_phrase = df_src[(df_src['label']=='')].copy()
    df_bp = df_src[~(df_src['label']=='')].copy()
    df_bp = df_bp.reset_index(drop=True)
    columns = df_bp.columns

    sid_results = []
    for tupl in df_bp.itertuples():
        sid = get_sentence_id(df_sentence_id, tupl.pid, tupl.stid, tupl.etid)
        sid_results.append((tupl.pid, sid, *tupl[2:]))
    df_bp_sid = pd.DataFrame(sid_results, columns = [columns[0], 'sid', *columns[1:]])
    df_bp_sid = df_bp_sid.reset_index(drop=True)

    df_results = pd.DataFrame()
    for sid, df_sid in df_bp_sid.groupby('sid'):
        for index in df_sid.index[1::]:
            df_indexes = df_sid.loc[index-1: index].copy()
            if len(df_indexes) != 2:
                continue
            first_index, second_index = df_indexes[['stid', 'etid']].values.tolist()
            first_lemma, second_lemma = df_indexes['lemma'].values.tolist()
            first_phrase, second_phrase = df_indexes['phrase'].values.tolist()
            first_label, second_label = df_indexes['label'].values.tolist()
            first_relation, second_relation = df_indexes['relation'].astype(str).apply(eval).values.tolist()
            
            overlap_indexes = check_overlap([first_index, second_index], base_index=0)
            # print('1: ', overlap_indexes, first_lemma, second_lemma)
            if overlap_indexes[0][0] in ['sequential', 'overlap', 'subset']:  
                # 
                if first_label == 'Property-Value':
                    continue
                if overlap_indexes[0][0] in ['sequential'] and first_label != second_label:
                    # print('sequential: ', first_label , second_label)
                    continue
                
                common_lemma = combine_phrase(first_lemma, second_lemma)
                common_phrase = combine_phrase(first_phrase, second_phrase, sep=' ')
                common_label = common_labels([first_label , second_label]) or second_label

                common_relation = first_relation + second_relation
                result = (pid, sid, *overlap_indexes[0][-1], common_lemma, common_phrase, common_label, common_relation)
                # print(overlap_indexes)
                # print(result)
                # Replace DataFrame row
                df_sid.loc[index, :] = np.array(result, dtype=object)
                df_sid.drop(index=index-1, inplace=True)

            elif overlap_indexes[0][0] in ['upperset']:
                common_lemma = second_lemma
                common_phrase = second_phrase
                common_label = common_labels([first_label , second_label]) or second_label
                common_relation = first_relation + second_relation
                result = (pid, sid, *overlap_indexes[0][-1], common_lemma, common_phrase, common_label, common_relation)
                # print(overlap_indexes)
                # print(result)                
                # Replace DataFrame row
                df_sid.loc[index-1, :] = np.array(result, dtype=object)
                df_sid.drop(index=index, inplace=True)

            elif overlap_indexes[0][0] in ['disjoint']:
                common_label = common_labels([first_label , second_label], add_qnt=True)
                if common_label == 'Property-Value' and first_label == 'Property-QNT':
                    common_label = None
                    first_label = 'Property-QNT'
                    second_label = 'Property-Value'
                elif common_label == 'Property-Value' and second_label == 'Property-QNT':
                    common_label = None
                    first_label = 'Property-Value'
                    second_label = 'Property-QNT'

                # print(sid, index, overlap_indexes, first_label, second_label, common_label) 
                df_sid.loc[index-1, :] = np.array([pid, sid, *first_index, first_lemma, first_phrase, common_label or first_label, first_relation], dtype=object)
                df_sid.loc[index, :] = np.array([pid, sid, *second_index, second_lemma, second_phrase, common_label or second_label, second_relation], dtype=object)

        # Make relation
        classes = df_sid[df_sid['label']=='Material-Class'].index
        elements = df_sid[df_sid['label']=='Material-Elem'].index
        qnts = df_sid[df_sid['label']=='Property-QNT'].index
        values = df_sid[df_sid['label']=='Property-Value'].index
        conditions = df_sid[df_sid['label']=='Method-Condition'].index


        # Material-QNT
        material_labels = ['Material-Class', 'Material-Element', 'Material-Shape', 'Material-Structure']
        material_results = []
        for material_label in reversed(material_labels):
        # for material_label in (material_labels):
            material_material, material_qnt = material_qnt_relation(df_sid, material_label)
            if len(material_qnt):
                material_results = (material_material, material_qnt)
                
                # print(material_material, material_qnt)
                
        for material_result in material_results:
            df_sid = add_phrase_relation(material_result, df_sid)
        
        # QNT-QNT, QNT-Value
        qnt_qnt, qnt_value = qnt_qnt_relation(df_sid)
        df_sid = add_phrase_relation(qnt_qnt, df_sid)
        df_sid = add_phrase_relation(qnt_value, df_sid)    
        
        # Value-condition
        value_condition = value_condition_relation(df_sid)
        df_sid = add_phrase_relation(value_condition, df_sid)     

        # Append all sid
        df_results = pd.concat([df_results, df_sid])


    phrase_results = []
    for tupl in df_phrase.itertuples():
        sid = get_sentence_id(df_sentence_id, tupl.pid, tupl.stid, tupl.etid)
        phrase_results.append((tupl.pid, sid, *tupl[2:]))
    df_phrase_sid = pd.DataFrame(phrase_results, columns = [columns[0], 'sid', *columns[1:]])

    df =  pd.concat([df_phrase_sid, df_tools.remove_overlap(df_results)]
                    ).astype({'pid':int, 'sid': int, 'stid':int, 'etid': int}).sort_values(
        ['pid', 'sid', 'stid', 'label']).reset_index(drop=True)
    
    return df.drop('sid', axis=1)
