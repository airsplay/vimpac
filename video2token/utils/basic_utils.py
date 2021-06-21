import os
import json


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def get_basename_no_ext(path):
    """ '/data/movienet/240p_keyframe_feats/tt7672188.npz' --> 'tt7672188' """
    return os.path.splitext(os.path.split(path)[1])[0]
