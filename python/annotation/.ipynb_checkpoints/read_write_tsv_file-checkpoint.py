from pathlib import Path

from python.annotation.tokenizertools import token_to_text
from python.annotation.tsv_components import *


def tsvfile2dataframe(tsvfile):
    """TSV to dataframes

    Args:
        tsvfile (Path): TSV file

    Returns:
        tuple: (spanset: pd.DataFrame, relationset: pd.DataFrame, tokenlines: pd.DataFrame)
    """
    header_check = Header.header_check
    with tsvfile.open(encoding='utf-8') as f:
        tsv = f.readlines()

    assert tsv[0].startswith(header_check), 'TSV format is not correct'

    span_list = []
    relation_list = []
    paragraph_list = []
    tokenline_list = []
    for line in tsv:
        line = line.rstrip()
        if line.startswith(SpanHeader.head):
            span_list.append(SpanHeader(line))
        elif line.startswith(RelationHeader.head):
            relation_list.append(RelationHeader(line))
        elif line.startswith(ParText.head):
            paragraph_list.append(ParText(line))
        elif re.match(r'^\d+-\d+', line):
            tokenline_list.append(TokenList(line))
        else:
            pass

    # Span set
    span_lines = sorted(map(lambda x: x.components, span_list), key=lambda x: x[0])
    df_spanset = pd.DataFrame(span_lines, columns=['layer', 'tagset'])

    # Relation set
    relation_lines = sorted(map(lambda x: x.components, relation_list), key=lambda x: x[0])
    df_relationset = pd.DataFrame(relation_lines, columns=['layer', 'tagset', 'link'])

    # Token lines
    span_tags = map(lambda x: x[1], span_lines)
    relation_tags = map(lambda x: x[1:], relation_lines)
    columns = ['pid', 'tid', 'start', 'end', 'token'] + [*chain.from_iterable(span_tags)] + [
        *chain.from_iterable(relation_tags)]
    token_lines = [*map(lambda x: x.components, tokenline_list)]
    df_tokenlines = pd.DataFrame(token_lines, columns=columns)

    return df_spanset, df_relationset, df_tokenlines.astype({'pid': int, 'tid': int, 'start': int, 'end': int})


def dataframe2tsvfile(df_spanset, df_relationset, df_tokenlines, tsv_file):
    """

    :param df_spanset:
    :param df_relationset:
    :param df_tokenlines:
    :param tsv_file:
    :return: Write tsvfile
    """
    span_list = [SpanHeader(tuple(row)) for row in df_spanset.values]
    span_text = '\n'.join(map(lambda x: x.text, span_list))

    relation_list = [RelationHeader(tuple(row)) for row in df_relationset.values]
    relation_text = '\n'.join(map(lambda x: x.text, relation_list))

    if relation_text:
        relation_text += '\n\n'
    else:
        relation_text += '\n'

    tokenline_texts = []
    paragraph_texts = []
    for pid, df_pid in df_tokenlines.groupby('pid'):
        tokenline_list = [TokenList(tuple(row)) for row in df_pid.values]
        tokenline_texts += ['\n'.join(map(lambda x: x.text, tokenline_list))]

        paragraph_texts.append(ParText.head + token_to_text(df_pid))

    tokenline_text = '\n\n'.join(['\n'.join([para, token]) for para, token in zip(paragraph_texts, tokenline_texts)])

    with open(tsv_file, encoding='utf-8', mode='w') as fw:
        print('\n'.join([Header.header, span_text, relation_text, tokenline_text]), file=fw)
