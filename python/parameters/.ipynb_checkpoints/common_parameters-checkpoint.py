from pathlib import Path


class CommonParameters:
    # Path
    domain = 'magtdm'
    archive = 'mddb'
    domain_dir = Path('archives') / domain
    archive_dir = domain_dir / archive
    article_dir = domain_dir / 'data_webannotsv'
    
    base_host = 'http://age.nims.go.jp'

    publishers = 'acs aip aps elsevier iop jjap rsc springer wiley'.split()

    split_char = '_'
    esc_symbols = r'\()[]{}*.+^$?|'  # '\\' <>

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
                     'MG·Oe': 'MGOe',
                     'kGs': 'kG',
                     'Gs': 'G',
                     'oC': '°C',
                     'KOe': 'kOe',
                      '⋅': '·',}

    rm_tokens = '- “ ” a an the The -PRON- ·'.split() + "‘ '".split()
    # Remove tokens with the following XML tags,
    rm_tags = [{'label_tag': ['xref', 'xref/xref', 'label', 'ext-link']},
               {'style_tag': ['sub', 'over']},
               {'style_tag3': ['disp-formula', 'equation', 'thead', 'tbody']}]

    # Skip tokens for keytype = 'first'
    first_letters = ['(', 'as', ')', '/']
    pre_values = ['about', 'above', 'almost', 'approximately', 'around',
                  'below', 'exceed', 'exceeding', 'merely', 'near', 'nearly', 'only', 'over',]
    pre_symbols = [*'~∼≃≈<>⩽⩾≤≥+±']
    pre_qnts = ['avarage', 'corresponding', 'giant', 'maximum', 'minimum',
                'large', 'poor']  #'intrinsic', 'extrinsic', 'high', 'low', 'small'
    skip_tokens = []