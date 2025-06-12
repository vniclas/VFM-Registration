import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging

import model.mlp as mlp
import model.projection_head as projection_head
import model.resunet as resunets
import model.simpleunet as simpleunets

MODELS = []


def add_models(module):
    MODELS.extend(
        [getattr(module, a) for a in dir(module) if 'Net' in a or 'MLP' in a or 'Head' in a])


add_models(simpleunets)
add_models(resunets)
add_models(mlp)
add_models(projection_head)


def load_model(name):
    '''Creates and returns an instance of the model given its class name.
  '''
    # Find the model class from its name
    all_models = MODELS
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        logging.info(f'Invalid model index. You put {name}. Options are:')
        # Display a list of valid model names
        for model in all_models:
            logging.info('\t* {}'.format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass
