import os
from datetime import datetime
from config.config import RESULT_PATH

def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def get_result_path(dataset_name, arch, seed, subfolder="", postfix=""):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    timestamp = get_timestamp()
    model_id = "{}_{}_{}_{}".format(timestamp, dataset_name, arch, seed)
    model_id = model_id + postfix
    model_path = os.path.join(RESULT_PATH, subfolder, model_id)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path