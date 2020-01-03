import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def int_or_none(v):
    if isinstance(v, str) and v == '':
        return None
    try:
        return int(v)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(f'Int value expected. Error occurred: {ex}')

def str_or_none(v):
    if isinstance(v, str) and v:
        return v
    return None