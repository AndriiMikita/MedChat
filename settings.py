import os
from transformers import pipeline

ROOT_DIR_PATH = os.path.realpath('.')
SCRIPT_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'src')
DATA_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'data')
ENV_DIR_PATH = os.path.join(ROOT_DIR_PATH, 'env')