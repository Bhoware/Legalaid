from .legal_data import LEGAL_DATA

def get_data(case):
    return LEGAL_DATA.get(case, None)