import pathlib
import re
import urllib.parse

import pandas as pd
import rdflib
from rdflib import Namespace, URIRef, Literal, XSD, RDFS

from parameters.common_parameters import CommonParameters


class RDFBaseFunc:
    base_host = CommonParameters.base_host

    def __init__(self, domain, archive, rdf_graph):
        self.base_prefix = Namespace(f'{self.base_host}/{domain}/{archive}#')

        self.rdf_graph = rdflib.Graph()
        if isinstance(rdf_graph, rdflib.graph.Graph):
            self.rdf_graph = rdf_graph
        elif isinstance(rdf_graph, pathlib.PosixPath):
            self.rdf_graph.parse(str(rdf_graph))
        elif isinstance(rdf_graph, str):
            self.rdf_graph.parse(rdf_graph)

    def combine_rdf(self, rdf_file):
        if isinstance(rdf_file, pathlib.PosixPath):
            self.rdf_graph.parse(str(rdf_file))
        elif isinstance(rdf_file, str):
            self.rdf_graph.parse(rdf_file)

    def make_uri(self, term):
        return URIRef(self.base_prefix + urllib.parse.quote(str(term)))

    @staticmethod
    def make_string(term):
        return Literal(term, datatype=XSD.string)

    @staticmethod
    def make_number(term):
        return Literal(term, datatype=XSD.integer)

    @staticmethod
    def unique_tuples(df_src, *columns):
        value_list = df_src[[*columns]].values
        return set(map(tuple, value_list))

    def make_triples(self, *triples):
        columns = ['subject', 'predicate', 'object']
        df_triple = pd.DataFrame(columns=columns)
        for i, column in enumerate(columns):
            df_triple.loc[:, column] = triples[i]

        return self.unique_tuples(df_triple, *columns)

    def add_triples_to_graph(self, triple_tupls):
        for triple_tupl in triple_tupls:
            if None not in triple_tupl:
                self.rdf_graph.add(triple_tupl)
            else:
                print("Can't add triple: ", triple_tupl)

    def add_dataflame_to_graph(self, two_columns, predicate, df_src, indexes=slice(None)):
        subject, object = two_columns
        df_uri = df_src.applymap(self.make_uri)

        triple_tupls = self.make_triples(df_uri.loc[indexes, subject], predicate,
                                         df_uri.loc[indexes, object])
        self.add_triples_to_graph(triple_tupls)

    def add_label_to_graph(self, name_list):
        uris = [*map(self.make_uri, name_list)]
        # Change label for QNT_0_K -> 0_K
        new_name_list = [re.sub(r'([A-Z][^_]+?-)((?:[^_]+_)+[^_]+)', r'\2', name)
            if re.match(r'[A-Z][^_]+?-([^_]+_)+[^_]+', name) else name for name in name_list]
        labels = [*map(self.make_string, new_name_list)]
        triple_tupls = self.make_triples(uris, RDFS.label, labels)
        self.add_triples_to_graph(triple_tupls)

    def remove_triples_from_graph(self, triple_tupls):
        for triple_tupl in triple_tupls:
            try:
                self.rdf_graph.remove(triple_tupl)
            except:
                print("triple is not found: ", triple_tupl)

    def save_rdf(self, filename):
        self.rdf_graph.serialize(filename, format='pretty-xml')
        print(f'RDF file: {filename}')
