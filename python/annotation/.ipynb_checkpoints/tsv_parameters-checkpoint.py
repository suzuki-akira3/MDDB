class TsvParameter:
    split_char = '_'
    esc_symbols = r'\()[]{}*.+^$?|'  # '\\' <>
    base_host = 'http://age.nims.go.jp/magtdm/mddb1#'

    # Pretreatment
    dic_normalize = {'coercivitie': 'coercivity',
                     'coercivities': 'coercivity',
                     'Coercivity': 'coercivity',
                     'alloys': 'alloy',
                     'behaviour': 'behavior',
                     'magnetisation': 'magnetization',
                     'improves': 'improve',
                     'energies': 'energy',
                     'curves': 'curve',
                     'loops': 'loop',
                     'emu·g': 'emug',
                      '⋅': '·',}
    
    rm_tokens = '- “ ” a an the The -PRON-'.split() + "‘ '".split()
    # Remove tokens with the following XML tags,
    rm_tags = [{'label_tag': ['xref', 'xref/xref', 'label', 'ext-link']},
               {'style_tag': ['sub', 'over']},
               {'style_tag3': ['disp-formula', 'equation', 'thead', 'tbody']}]    

class MagTDMTSV:
    # Add spanset
    layer_tagset_tuples = [
        ('MatDic_Items', ['matdic_uri']),
        ('Annotation', ['regex']),
        ('Abbreviation', ['abbreviation']),
        ('Materials',
         ['material_tag', 'method_tag', 'property_tag', 'sample_tag', 'theory_tag'])
    ]

    # Add relationset
    layer_tag_link_tuples = [
        ('Relations', 'relation_tag',
         'Materials'),
        ('Abbreviation_link', 'abbreviation_tag',
         'Abbreviation'),
    ]
