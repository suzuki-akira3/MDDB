import re
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

from create_database.dataset.dictionary import append_dictionary
from create_database.annotation.phrase_tools import CheckBaseWord
from create_database.dataset import construction
from create_database.dataset.construction import Component, MatDicGraph, Convert2RDF

# Add base words obtained from phrases
domain = 'magtdm'
archive = 'mddb'
archive_dir = Path('archives') / domain / archive

data_dir = Path(archive_dir / 'data')

# Initial KB
df_base_word = pd.read_csv(data_dir / f't2_initial_KB.csv', keep_default_na=False).astype(str)
df_base_word['All base words'] = df_base_word['All base words'].apply(lambda x: x.split(', '))
df_base_word = df_base_word.explode('All base words')
df_base_word['root'] = 'Dic'
dictionary = df_base_word['root'].str.cat(df_base_word['Category'], sep='-')
dic_class = df_base_word['Category'].str.cat(df_base_word['Annotation label'], sep='-')
class_base = df_base_word['Annotation label'].str.cat(df_base_word['All base words'], sep='-')

# Structured phrases from rule-based
df_structured_phrases = pd.read_csv(data_dir / 't5_structured_phrases.csv', keep_default_na=False)
df_structured_phrases['Phrase'] = df_structured_phrases['phrase']
df_structured_phrases['Labels'] = df_structured_phrases['label'].apply(construction.get_contents)
df_structured_phrases['Relations'] = df_structured_phrases['relations'].apply(construction.get_contents, relation=True)

# Structured phrases from pattern matching
df_phrase_label = pd.read_csv(data_dir / 't9_pattern_matched.csv', keep_default_na=False)
df_phrase_label['label'] = df_phrase_label['Pattern'].apply(construction.get_label_from_pattern)
df_phrase_label['Base_phrase'] = df_phrase_label['Base_phrase'].apply(construction.get_base_phrase)
df_phrase_label['Labels'] = [[*zip(bps, lbls)] for bps, lbls in df_phrase_label[['Base_phrase', 'label']].values]
df_phrase_label['Relations'] = df_phrase_label['Relations'].apply(construction.get_contents, relation=True)
columns = ['Phrase', 'Labels', 'Relations']
df_phrase_label = pd.concat([df_structured_phrases[columns], df_phrase_label[columns]])

base = CheckBaseWord('base')
first = CheckBaseWord('first')

phrase_edges = []
basephrase_edges = []
phrase_label = []
dic_basephrase_label = {}
dic_basephrase_relation = {}
for tupl in tqdm(df_phrase_label.iloc[:].itertuples()):
    # print(tupl)
    phrase = tupl.Phrase
    labels = tupl.Labels
    relations = tupl.Relations

    a = Component(phrase, labels, relations)

    phrase_edges += a.phrase_basewords(class_base)
    basephrase_edges += a.phrase_check_basewords(class_base)
    a.get_relations()

    phrase_label += a.phrase_label

    dic_basephrase_label[phrase] = [str(labels)]
    dic_basephrase_relation[phrase] = [str(relations)]

basephrase_edges = [*map(construction.label_arrange, basephrase_edges)]
dic_relation = a.dic_relations

nxgraph = MatDicGraph()
nxgraph.add_edges(phrase_edges)
nxgraph.add_edges(basephrase_edges)
nxgraph.set_attributes('Relation', dic_relation)
nxgraph.set_attributes('Annotation_label',
                       {f"{v[0].split('-')[-1]}-{k}": v for k, v in phrase_label if len(v) == 1})

nxgraph.set_attributes('basephrase_label', dic_basephrase_label)
nxgraph.set_attributes('basephrase_relation', dic_basephrase_relation)

# Add base structure
nxgraph.add_edges(zip(dic_class, class_base))
nxgraph.add_edges(zip(dictionary, dic_class))

nxgraph.set_attributes('baseword', {baseword: ["True"] for baseword in class_base})
nxgraph.set_attributes('phrase', {phrase: ["True"] for phrase in df_phrase_label['Phrase']})

basephrases = [tupl[0] for tupl in basephrase_edges if not re.search(r'[A-Z][A-Za-z]+?\-', tupl[0])]
basephrases = set(basephrases)
nxgraph.set_attributes('basephrase', {basephrase: ["True"]
                                      for basephrase in basephrases})

# Remove isolated base_words
nxgraph.remove_isolate_baseword(class_base)
nxgraph.remove_isolate_baseword(dic_class)
nxgraph.remove_isolate_baseword(dictionary)

graph = nxgraph()
rdf = Convert2RDF(domain, archive, graph, None)
rdf.convert_process()
rdf_file = data_dir / f'matdic_{domain}_{archive}.owl'
rdf.save_rdf(str(rdf_file))
