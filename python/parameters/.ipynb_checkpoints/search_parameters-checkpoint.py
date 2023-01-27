import re
from pathlib import Path


class RegexMaterials:
    RE_elements = 'Y La Ce Pr Nd Sm Tb Dy'.split(' ')
    RE_elements_string = '|'.join(RE_elements)
    required_elements = 'Fe B'.split()
    doping_elements = 'C O Al Si Ti V Cr Mn Co Ni Cu Ga Ge Zr Nb N Zn'.split(' ')
    all_elements = RE_elements + required_elements + doping_elements

    symbol_elements = ['TM', 'M', 'RE', 'R', 'X', 'A', 'T']
    elements_with_symbols = all_elements + symbol_elements
    elements_with_symbols_string = '|'.join(sorted(elements_with_symbols, key=len, reverse=True))

    all_elements_string = '|'.join(sorted(all_elements, key=len, reverse=True))
    symbol_elements_string = '|'.join(sorted(symbol_elements, key=len, reverse=True))

    initial_symbols = f'(?:[αβγδε]_)'
    doped = f'(?:({all_elements_string})_(co_)?doped_)'
    multi_elems = f'(?:({elements_with_symbols_string})+)'
    blacket_elems = f'(?: ({all_elements_string})+(_,_({all_elements_string})_)+ \)_)'
    separators = f'(?:[_′·⋅/\(\)]*)'
    start_blacket = f'(\[_)? ({multi_elems}_)? (\(_)'
    end_blacket = f'(_\) (_\])?)'
    suffix = f'(?:(_based)?_alloy | (_based)?(_nanocomposite)?(_sintered)?(_permanent)?_magnet | _compound)'
    material_base = f'( ( {blacket_elems}? {multi_elems}{separators} )+ {multi_elems} )'

    re_materials = re.compile(fr'''
        ({start_blacket}|{initial_symbols}|{doped})?
        {material_base}
        {end_blacket}?
        {suffix}?
        ''', re.VERBOSE)

    re_elements = re.compile(fr'''
        (?P<symbol>{symbol_elements_string})_[=:]_
        (?P<elements>(({all_elements_string})_,_)+({all_elements_string})(_,_and_|_,_or_|_,_|_and_|_or_)?({all_elements_string})?)
        (?P<end>_,_etc)?
        ''', re.VERBOSE)

    re_RE = re.compile(fr'''
        (?:({RE_elements_string})_,_)+({RE_elements_string})(?:_,_and_|_,_or_|_,_|_and_|_or_)?(?:{RE_elements_string})?
        ''', re.VERBOSE)

    re_composition = re.compile(r'''
        (?P<symbol>[dvwxyz])_[=]_(?P<numbers>(?:\d+_,_)+(\d+)(?:_,_and_|_,_or_|_,_|_and_|_or_)\d+)
        ''', re.VERBOSE)


class RegexProperties:
    sblacket = '([\(\[\|]_)'
    eblacket = '(_[\)\]\|])'

    def quality(self, dic_key):
        prefix = '|'.join(['(ultra_)?high', 'low', 'maximum', 'minimum', 'intrinsic', 'apparent', 'effective', 'peak',
                           'optimal', 'poor'])
        return f'(?P<{dic_key}>{prefix})'

    def value(self, dic_key):
        return f'''({self.sblacket}_)?(?P<{dic_key}>(0_)?0(_×_0_(-_)?\d+)?) (_{self.eblacket})?
            (_{self.sblacket}?(?P<{dic_key}_conv>(0_)?0(_×_0_(-_)?\d+)?){self.eblacket})?'''

    def material(self, dic_key, length=50):
        # in ?
        return f'(of|for|in|using)_(?P<{dic_key}>[^0≈∼<>⩽⩾≤≥][^\,:;]{{,{length}}})'

    def process(self, dic_key, length=50):
        return f'(after|with)_(?P<{dic_key}>[^0≈∼<>⩽⩾≤≥][^\,:;]{{,{length}}})'

    def change(self, dic_key):
        adverb = ['continuously', 'substantially', 'considerably', 'significantly', 'doubly', 'initially', 'steadily',
                  'firstly', 'steeply', 'dramatically', 'gradually']
        change_adverb = '|'.join(map(lambda x: f'(be_)?{x}', adverb))
        verb = ['enhanced?', 'drop(ped|s)?', '(in|de)crease(s|d)?', 'recover(ed)?', 'falls?', 'var(y|ied|ies)',
                'improved?', 'degrad(ed)?', 'reach(es|ed)?', '(in_)?excess', 'reduced?']
        change_verb = '|'.join(map(lambda x: f'(have_)?(also_)?(be_)?{x}', verb))
        change_adverb_f = '|'.join(map(lambda x: f'{x}', adverb))
        return f'''((?P<{dic_key}>({change_adverb})_)?
                    ({change_verb})
                    (_({change_adverb_f}))?)'''

    @property
    def value_get(self,):
        get = ['report(ed|s)?', 'obtain(ed|s)?', 'be_found(_to)?', 'show(ed)?', 'estimat(ed)?(_to)?',
               'determined?', 'measured?', 'achieved?', 'expect(ed)?(_to)?', 'observed?(_to)?',
               'dive(s)(_to)?']
        return '|'.join(map(lambda x: f'(((that_|which_)?can_)?be_)?{x}', get))

    def value_is(self, dic_key):
        be = ['((not?_)?(as_)?(high|low|large|small|more|less)_)?(than|as)', 'exceed(ing|s)?', '(up_?)?to(_be)?', 'equal(_to)?',
              'in_(range|order)(_of)?', '(maximum|minimum)(_value)?(_of)?', 'same(_as)?', 'equal(_to)?', '(slight_)?peak_of'
                                                                                                         '(very_)?((low|high)_)?(of_)?order_of', 'above', 'by', 'as', 'with', 'be', 'of', 'to',
              '[=≈∼~<>\(]']
        value_be = '|'.join(be)
        return f'''(?P<{dic_key}>(((of_)?)({self.value_get})_)?
            (((which|that)_)?(can|could)_)?(be_)?({value_be}) )'''

    def pre_value(self, dic_key):
        pre = ['over', 'about', 'approximately', '(low|high|avarage)_value', 'only', '(somewhat_)?around(_∼)?',
               'extremely_low_:', 'nearly', 'in_range', '[≈∼~<>⩽⩾≤≥\+±]']
        pres = '|'.join(pre)
        return f'(?P<{dic_key}>{pres}) {self.eblacket}?'

    def units(self, dic_key, group=1):
        # unit_B is before A
        units_A = ['.?T', '.?A[_·⋅]m_-_1', '.?A_/_m', '.?Oe', '.?emu_/_gm?', '.?emu[_·⋅]?gm?_-_1', '.?emu_/_cc', '.?emu_/_.?m_3',
                   '.?emu[_·⋅]?cc_-_1', '.?G(uas)?s?', '.?A_?m_2_.?gm?_-_1', '.?A_?m_2_/_.?g', '.?V_/_.?m', '.?V_.?m_-_1' ]
        units_B = ['.?J_/_.?m_3', '.?J_.?m_-_3', '.?G_?(O|o)e']
        unit = [units_A, units_B, units_B + units_A][group - 1]
        unit_form = f'|'.join(unit)
        return f'(?P<{dic_key}>{unit_form})'

    def conditions(self, dic_key, group=1):
        condition = ['(0_)?0_(K|°_?C)', 'room_temperature', 'room_T', 'RT', '(0_)?0_T']
        form = '|'.join(condition)
        conditions1 = f'{self.sblacket}?(?P<{dic_key}>{form}){self.eblacket}?'
        conditions2 = f'(?P<{dic_key}1>{form})_and_(?P<{dic_key}2>{form})'
        conditions3 = f'(?P<{dic_key}1>{form})_,_(?P<{dic_key}2>{form})_and_(?P<{dic_key}3>{form})'
        return [conditions1, conditions2, conditions3][group - 1]

    def single_value(self, unit=3):
        return f'''(({self.pre_value('pre_value_s1')})_)?
                    ({self.value('value_s1')})_ ({self.units('unit_s1', unit)}) {self.eblacket}?
                    (_({self.material('material_s1')}))?
                    (_(at|@)_({self.conditions('condition_s1', 1)}))?
                    '''

    def range_value(self, unit=3):
        return f'''((rang(ed|ing)_)?from|(in_)?between|either)_ 
            (({self.pre_value('pre_value_r1')})_)? ({self.value('value_r1')})_(({self.units('unit_r1', unit)})_)? 
            (({self.material('material_r1')})_)? ((at|@)_({self.conditions('condition_r1', 1)})_)?
            ((up_)?to|and|or)_ (({self.pre_value('pre_value_r2')})_)? ({self.value('value_r2')})_({self.units('unit_r2', unit)})
            (_({self.material('material_r2')}))? (_(at|@)_({self.conditions('condition_r2', 1)}))?
            '''

    def multi_values_2(self, unit=3):
        return f'''(({self.pre_value('pre_value_m2')})_)? ({self.value('value_m2_1')})_(({self.units('unit_m2_1', unit)})_)? 
            (({self.material('material_m2_1')})_)? ((at|@)_({self.conditions('condition_m2_1', 1)})_)?
            (,_|and_|or_) (({self.value_is('value_is_m2_2')})_)? 
            (({self.pre_value('pre_value_m2_2')})_)? ({self.value('value_m2_2')})_({self.units('unit_m2_2', unit)}) 
            (_({self.material('material_m2_2')}))? (_(at|@)_({self.conditions('condition_m2_2', 1)}))?
            (_(at|@)_({self.conditions('conditions_m2_', 2)}))?    
            '''

    def multi_values_3(self, unit=3):
        return f'''(({self.pre_value('pre_value_m3_1')})_)? ({self.value('value_m3_1')})_(({self.units('unit_m3_1', unit)})_)? 
            (({self.material('material_m3_1')})_)? ((at|@)_({self.conditions('condition_m3_1', 1)})_)?
            (,_|and_|or_) 
            (({self.pre_value('pre_value_m3_2')})_)? ({self.value('value_m3_2')})_(({self.units('unit_m3_2', unit)})_)?
            (_({self.material('material_m3_2')}))? (_(at|@)_({self.conditions('condition_m3_2', 1)}))?
            (,_and_|and_|,_) 
            (({self.pre_value('pre_value_m3_3')})_)? ({self.value('value_m3_3')})_({self.units('unit_m3_3', unit)})
            (_({self.material('material_m3_3')}))? (_(at|@)_({self.conditions('condition_m3_3', 1)}))?
            (_(at|@)_({self.conditions('conditions_m3_', 3)}))? 
            '''

    def contents(self, unit):
        return f'''
            ((values?)_)?
            (({self.value_get})_)?
            (({self.material('material_1')})_)? 
            (({self.process('process_1')})_)?
            (({self.change('change')})_)? 
            (
                ( (({self.value_is('value_is_r')})_)? ({self.range_value(unit)}) ) |         
                ( (({self.value_is('value_is_m')})_)? (({self.multi_values_3(unit)})|({self.multi_values_2(unit)}) ) ) |   
                ( (({self.value_is('value_is_s')})_)?  ({self.single_value(unit)}) )
            )
            {self.eblacket}?
        '''

    def single_value_rev(self, unit=3):
        return f'''( (({self.pre_value('pre_value_rev')})_)?
                     ({self.value('value_1')})_({self.units('unit_1', unit)})
                     (_\(_({self.value('value_2')})_({self.units('unit_2', unit)}))? 
                     {self.eblacket}?)
                '''

    @property
    def re_coercivity(self, ):
        unit = 1
        global coercivity
        name = '''(coercivit(y|ie|ies)|coercive_(magnetic_)?field) (_value)? (_of)? '''
        symbol = '''(Δ_?)?(μ_?(0_)?)? (H(_?(c|C))?|h_?c) [^_]{,3} (_\(_._\))? (_,)?'''
        coercivity = f'''
                (room_temperature_)? (
                    (({name})(_,)?_)? {self.sblacket}? ({symbol}) (_,|{self.eblacket})? 
                    |
                    ({name}) (_at_0_(K|°_?C))?
                    ) '''
        coercivity_rev = f''' (room_temperature_)?
                ( coercivit(y|ie|ies) | coercive_(magnetic_)?field | (Δ_?)?(μ_?(0_)?)?H(_?(c|C)[^_]{{,3}})? )'''
        return re.compile(fr'''
            ( (({self.quality('quality')})_)? (?P<property>{coercivity})_ {self.contents(unit)} ) 
            |
            ( ({self.single_value_rev(unit)}) (_(for|of))? _(?P<property_rev>{coercivity_rev}) )
            ''', re.VERBOSE)

    @property
    def re_remanence(self, ):
        unit = 1
        global remanence
        name = '''(remanence | (remanen(t|ce)|remnant)_magnetization ) (_value)? (_of)? '''
        symbol = '''(Δ_?)?(μ_?(0_)?)? ( (B|M)(_?(r|R))? [^_]{,3} | b_?r | m_?r ) (_\(_._\))? (_,)?'''
        remanence = f'''
            (room_temperature_)? (
                (({name})(_,)?_)? {self.sblacket}? ({symbol}) (_,|{self.eblacket})? 
                |
                ({name}) (_at_0_(K|°_?C))?
            ) '''
        remanence_rev = f'''( (room_temperature_)?
            remanence | (remanen(t|ce)|remnant)_magnetization | (Δ_?)?(μ_?(0_)?)?(B|M)(_?(r|R)[^_]{{,3}})?
            )'''
        return re.compile(fr'''
            ( (({self.quality('quality')})_)? (?P<property>{remanence})_ {self.contents(unit)} ) 
            |
            ( ({self.single_value_rev(unit)}) (_(for|of))? _(?P<property_rev>{remanence_rev}) )
            ''', re.VERBOSE)

    @property
    def re_BHmax(self, ):
        unit = 3
        global BHmax
        name = '''maximum_energy_product (_value)?'''
        symbol = '''(  [\(\[\|]_(B|J)H_[\)\]\|] | (B|J)H ) (_max)? (_value)?'''
        BHmax = f'''
            (room_temperature_)? (
                (({name})(_,)?_)? {self.sblacket}? ({symbol}) (_,|{self.eblacket})? 
                |
                ({name}) (_at_0_(K|°_?C))?
            ) '''
        BHmax_rev = f'''( (room_temperature_)?
                {name} | ({symbol})
                        )'''
        return re.compile(fr'''
            ( (({self.quality('quality')})_)? (?P<property>{BHmax})_ {self.contents(unit)} ) 
            |
            ( ({self.single_value_rev(unit)}) (_(for|of))? _(?P<property_rev>{BHmax_rev}) )
            ''', re.VERBOSE)

    @property
    def re_magnetization(self, ):
        unit = 1
        global magnetization
        name = '''((saturat(ion|ed)_)?magnetization(_saturat(ion|ed))? ) (_value)? (_of)? '''
        symbol = '''(Δ_?)?(μ_?(0_)?)?((0|4)_?π_?)? ( M(_?s)? | m_?s) [^_]{,3} (_\(_._\))?  (_,)?'''
        magnetization = f'''
            (room_temperature_)? (
                (({name})(_,)?_)? {self.sblacket}? ({symbol}) (_,|{self.eblacket})? 
                |
                ({name}) (_at_0_(K|°_?C))?
            ) '''
        magnetization_rev = f'''( (room_temperature_)?
                (saturation_|saturated_)?magnetization(_saturation)? | 
                (Δ_?)?(μ_?(0_)?)?((0|4)_?π_?)?(M|m_?s)(_?(s)[^_]{{,3}})?
                )'''
        return re.compile(fr'''
            ( (({self.quality('quality')})_)? (?P<property>{magnetization})_ {self.contents(unit)} ) 
            |
            ( ({self.single_value_rev(unit)}) (_(for|of))? _(?P<property_rev>{magnetization_rev}) )
            ''', re.VERBOSE)

    @property
    def re_properties(self, ):
        properties = f'({coercivity})|({remanence})|({BHmax})|({magnetization})'
        # 3 -> 2
        return re.compile(fr'''
            (?P<property1>({properties}))_ (,_|and_) (?P<property2>{properties})_  
            (,_and_|and_|,_)? ((?P<property3>{properties})_)?
            (({self.material('material_m')})_)? 
            (({self.change('change')})_)? 
            ({self.value_is('value_is_m')})_ 
            ({self.multi_values_3(unit=3)}|{self.multi_values_2(unit=3)})_?
            ''', re.VERBOSE)

class SearchParameters:
    add_groupdict = True
    columns = ['publisher', 'filename', 'pid', 'tids', 'label', 'lemma', 'phrase']
    if add_groupdict:
        columns += ['dict']

    # Search parameters
    property = RegexProperties()
    search_phrases = [
        ('coercivity', property.re_coercivity),
        ('remanence', property.re_remanence),
        ('magnetization', property.re_magnetization),
        ('BHmax', property.re_BHmax),
        ('properties', property.re_properties),

        ('materials', RegexMaterials.re_materials),
        ('elements', RegexMaterials.re_elements),
        ('RE_elements', RegexMaterials.re_RE),
        ('compositions', RegexMaterials.re_composition),
    ]

    check_phrase_labels = ['materials']
    add_dict_labels = ['properties', 'coercivity', 'Hc',
                       'remanence', 'Ms', 'magnetization', 'Br', 'BHmax',
                       'compositions', 'elements']

def main():
    prop = RegexProperties()
    text_list = '''
maximum
energy
product
[
(
BH
)
]
of
0
MGOe
    '''.split()
    print('Input: ', '_'.join(text_list))
    m = re.match(prop.re_BHmax, '_'.join(text_list))
    # m = re.match(prop.re_magnetization, 'M_value_of_0_emu_/_g')
    if m:
        print('Matched: ', m.group())
        print({k: v for k, v in m.groupdict().items() if v})

if __name__ == '__main__':
    main()
