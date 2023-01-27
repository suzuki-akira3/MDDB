# A material dictionary database to extract information on per-manent magnets from scientific articles
## Dataset
### Table data
- [Table_2_all_data.csv](data/Table_data/Table_2_all_data.csv)
  - Table 2. Construction of the initial KB. We selected the categories and subcategories. The base words are the suffix or prefix words of technical terms linked to subcategories used as annotation labels.
- [Table_3_all_data.csv](data/Table_data/Table_3_all_data.csv)
  - Table3. Extracted phrases through the frequency-based rule and regular expression search. Un-necessary phrases were filtered or manually eliminated. The list is arranged in order of frequency
- [Table_5_all_data.csv](data/Table_data/Table_5_all_data.csv)
  - Table 5. Examples of structured phrases. The relations column indicates a connection between the ith and jth labels of a phrase. Note that i and j start from zero.
- [Table_6_all_data.csv](data/Table_data/Table_6_all_data.csv)
  - Table 6. Frequent label patterns. Phrases that include base phrase relations were selected, and their base phrases were replaced with labels (e.g <QNT>), following which they were counted. The list is arranged in order of frequency.
- [Table_7_all_data.csv](data/Table_data/Table_7_all_data.csv)
  - Table 7. Manually created label patterns. To extract property data, complex patterns, including multiple material names, property names, and values were also considered. The relations column indicates a connection between the ith and jth labels of a phrase. Note that i and j start from zero.
- [Table_8_all_data.csv](data/Table_data/Table_8_all_data.csv)
  - Table 8. Label-related base phrases extracted during the phrase structure construction process.
- [Table_9_all_data.csv](data/Table_data/Table_9_all_data.csv)
  - Table 9. Phrases extracted through the label pattern-matching method. Each label was replaced with all combinations of associated base words or phrases to check whether they were present in the texts. The relations column indicates the connection between the ith and jth labels of a phrase. Note that i and j start from zero.
- [Table_10_all_data.csv](data/Table_data/Table_10_all_data.csv)
  - Table 10. Property data records from the magnet domain using the MDDB. Property data were manually extracted using modified annotations after performing automatic annotation.
### Material dictionary database
- [MDDB.owl](data/Material_dictionary_databese/MDDB.owd)
  - Material dictionary database (MDDB)

## Soruce code
- [createdatabase.py](python/createdatabase.py)
  - Input files:
    - [Table_2_all_data.csv](data/Table_data/Table_2_all_data.csv)
    - [Table_5_all_data.csv](data/Table_data/Table_5_all_data.csv)
    - [Table_9_all_data.csv](data/Table_data/Table_8_all_data.csv)
  - OUtput file:
    - [MDDB.owl](data/Material_dictionary_databese/MDDB.owd)
