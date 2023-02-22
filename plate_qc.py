import re

def filter_plates(data, personalized_plates=False):
    ''' Drop readings that are noise readings.

    Personalized plates based on : https://saaq.gouv.qc.ca/immatriculation/plaque-immatriculation-personnalisee

    Normal plates based on : https://saaq.gouv.qc.ca/immatriculation/categories-plaques-immatriculation

    Parameters
    ----------
    personalized_plates: boolean (Default: False)
        Indicator to keep personalized plates in the data. 

    Returns
    -------
    data: pd.DataFrame
        Enhance data filtered.
    '''
    data = data.copy()

    # keep this here in case it will come to use
    to_remove = r'[B8][AR]*R\w*|TR[O0]*[TT]*[0O]*[I1]R\w*|[E]*C[0O][L]*[1I]*E[R]*|[U]*TL[1I]SER|P[O0]{0,1}LICE'

    # personalized plates from 5 to 7 chars
    personized_plates_57 = r'[a-z]{0,2}[a-z]{5}|\d[a-z]{4,6}|[a-z]{4,6}'
    personized_plates_24 = r'\w{2,4}'

    # normal plates
    promenade = r'\d{3}[a-z]{3}|\d{3}H\d{3}|[a-z]{3}\d{3}|[a-z]\d{2}[a-z]{3}'
    promenade_cc_cd = r'c[cd]\d{4}'
    commercial = r'f[a-z]{2}\d{4}'
    trailer = r'r[a-z]\d{4}[a-z]'
    five_digit = r'(a|ae|ap|au)\d{5}'
    six_digit = r'(c|l|f)\d{6}'
    radio = r'(VA2|VE2)[a-z]{2,3}'
    electric = r'[vcl]\d{3}1VE$|c[cd]\d1VE'
    movable =  r'x[0-9a-z]\d{5}'
    other = r'[cf][a-z]{3}\d{3}'

    if personalized_plates:
        patterns = [
            personized_plates_57,
            personized_plates_24,
            promenade,
            promenade_cc_cd,
            commercial,
            trailer,
            five_digit,
            six_digit,
            radio,
            electric,
            movable,
            other
        ]
    else:
        patterns = [
            promenade,
            promenade_cc_cd,
            commercial,
            trailer,
            five_digit,
            six_digit,
            radio,
            electric,
            movable,
            other
        ]

    compiled_plates = re.compile(r'^(' + '|'.join(x for x in patterns) + r')', re.IGNORECASE)    

    # filter plates 
    filter = data.plaque.apply(lambda x: True if re.match(compiled_plates, x) else False)

    return filter
