import re
import networkx as nx
import rdflib
from rdflib import URIRef, RDF, OWL
from rdflib.namespace import RDFS

from modules.rdf_base_functions import RDFBaseFunc
from modules.phrase_tools import CheckBaseWord
from parameters.common_parameters import CommonParameters
from modules.dictionary import append_dictionary

split_char = CommonParameters.split_char


class Component:
    base = CheckBaseWord('base')
    first = CheckBaseWord('first')
    dic_relations = {}

    def __init__(self, phrase, labels, relations):
        self.phrase = phrase
        self.labels = labels
        self.relations = relations

        # tuple (phrase, <label>)
        self.basewords = [(tupl[0], tupl[1]) for tupl in self.labels if split_char not in tupl[0]]
        self.basephrases = [(tupl[0], tupl[1]) for tupl in self.labels if split_char in tupl[0]]
        self.phrase_label = [(phrase, label) for phrase, label in self.labels if phrase == self.phrase]

    def get_relations(self, ):
        results = []
        for tupl in self.relations:
            src_phrase = self.labels[tupl[0]][0]
            src_label = self.labels[tupl[0]][1]
            dst_phrase = self.labels[tupl[1]][0]
            dst_label = self.labels[tupl[1]][1]

            if len(src_label.split('|')) == 1 and len(dst_label.split('|')) == 1:
                results.append(('-'.join([src_label, src_phrase]), '-'.join([dst_label, dst_phrase])))

        [append_dictionary(self.dic_relations,
                           re.sub(r'<([A-Z][A-Za-z]+)>-', r'\1-', tupl[0]),
                           re.sub(r'<([A-Z][A-Za-z]+)>-', r'\1-', tupl[1])
                           ) for tupl in results]
        self.dic_relations = {k: list(set(v)) for k, v in self.dic_relations.items()}

    @classmethod
    def all_relations(cls):
        return cls.dic_relations

    def phrase_basewords(self, class_base):
        phrases = []
        for tupl in self.basewords:
            # Baseword check
            clas = [c.split('-')[0] for c in class_base if c.split('-')[-1] == tupl[0]]
            labels = tupl[1].split('|')
            for label in labels:
                if label[1:-1] in clas:
                    phrases += [('-'.join([label[1:-1], tupl[0]]), self.phrase)]

        return set(phrases)

    def phrase_check_basewords(self, class_base):
        base = []
        first = []
        phrases = []
        for tupl in self.basephrases:
            baseword_base = self.base.get_baseword(tupl[0])
            clas = [c.split('-')[0] for c in class_base if c.split('-')[-1] == baseword_base]
            labels = tupl[1].split('|')
            for label in labels:
                if label[1:-1] in clas:
                    base += [('-'.join((label[1:-1], baseword_base)), tupl[0])]
                    if self.phrase != tupl[0]:
                        phrases += [(tupl[0], self.phrase)]

            baseword_first = self.first.get_baseword(tupl[0])
            clas = [c.split('-')[0] for c in class_base if c.split('-')[-1] == baseword_first]
            labels = tupl[1].split('|')
            for label in labels:
                if label[1:-1] in clas:
                    first += [('-'.join((label[1:-1], baseword_first)), tupl[0])]
                    if self.phrase != tupl[0]:
                        phrases += [(tupl[0], self.phrase)]

        return set(base + first + phrases)


class MatDicGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def __call__(self):
        return self.graph

    def add_edges(self, tupls, weight=1):
        edges = ((parent, child, weight) for parent, child in tupls)
        self.graph.add_weighted_edges_from(edges)

    def set_attributes(self, attribute, appen_dic):
        # print({k: {attribute: '|'.join(v_list)} for k, v_list in appen_dic.items()})
        nx.set_node_attributes(self.graph, values={k: {attribute: '|'.join(v_list)} for k, v_list in appen_dic.items()})

    def reverse(self):
        nx.reverse(self.graph)

    def remove_isolate_baseword(self, class_base):
        rm_nodes = [node for node in class_base if not [*self.graph.successors(node)]]
        self.graph.remove_nodes_from(rm_nodes)

    def save_graph(self, filename):
        nx.write_graphml_lxml(self.graph, filename)


class Convert2RDF(RDFBaseFunc):
    attribs = ['Annotation_label', 'Relation', 'basephrase_label', 'basephrase_relation', 'baseword', 'basephrase',
               'phrase']
    attrib_url = ['Annotation_label', 'Relation']
    attrib_str = ['basephrase_label', 'basephrase_relation', 'baseword', 'basephrase', 'phrase']

    def __init__(self, domain, archive, nxgraph, rdf_graph):
        super().__init__(domain, archive, rdf_graph)
        self.rdf_graph = rdflib.Graph()
        self.rdf_graph.add((URIRef(self.base_prefix), RDF.type, OWL.Ontology))
        self.nxgraph = nxgraph

    def convert_process(self, ):
        # Make base structure
        df = nx.to_pandas_edgelist(self.nxgraph)
        self.add_dataflame_to_graph(['target', 'source'], RDFS.subClassOf, df)

        name_list = set(df.source) | set(df.target)
        self.add_label_to_graph(name_list)

        for attrib in self.attribs:
            if attrib in self.attrib_url:
                conv = self.make_uri
            else:
                conv = self.make_string
            phrase_types = nx.get_node_attributes(self.nxgraph, attrib)

            if attrib == 'Relation':
                for node, value in phrase_types.items():
                    for string in value.split('|'):
                        triples = [
                            (self.make_uri(node), self.make_uri(attrib), conv(string))
                        ]
                        self.add_triples_to_graph(triples)

            else:
                triples = [
                    (self.make_uri(node), self.make_uri(attrib), conv(string))
                    for node, string in phrase_types.items()]

                self.add_triples_to_graph(triples)


def get_contents(text, relation=False):
    tupls = re.finditer(r'\((.+?), (.+?)\)', text)
    if relation:
        return [*map(lambda x: (int(x.group(1)), int(x.group(2))), tupls)]
    else:
        return [*map(lambda x: x.groups(), tupls)]


def get_base_phrase(text):
    return text[1:-1].split(', ')


def get_label_from_pattern(text):
    tupls = re.finditer(r'<(.+?)>', text)
    return [*map(lambda x: x.group(), tupls)]


def label_arrange(tupl):
    return [*map(lambda x: re.sub(r'<([A-Z][A-Za-z]+)>-', r'\1-', x), tupl)]
