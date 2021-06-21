import lmdb
import numpy as np
import io


def save_np_array_to_lmdb(lmdb_txn, key, np_array):
    """serialize numpy array and save it lmdb

    Some notes on np array to strings:
        https://stackoverflow.com/questions/25837641/save-retrieve-numpy-array-from-string
    """
    in_mem_file = io.BytesIO()
    np.save(in_mem_file, np_array)
    lmdb_txn.put(key=str(key).encode("utf-8"), value=in_mem_file.getvalue())


def load_np_array_from_lmdb(lmdb_txn, key):
    """Load and deserialize np array from lmdb"""
    lmdb_value = lmdb_txn.get(str(key).encode("utf-8"))
    np_array = np.load(io.BytesIO(lmdb_value))
    return np_array


def save_npz_to_lmdb(lmdb_txn, key, **kwds):
    """serialize numpy array and save it lmdb
    Args:
        lmdb_txn: lmdb txn object
        key: str, lmdb key
        kwds: Keyword Arguments, arrays to save, the same as np.savez

    Some notes on np array to strings:
        https://stackoverflow.com/questions/25837641/save-retrieve-numpy-array-from-string
    """
    in_mem_file = io.BytesIO()
    np.savez(in_mem_file, **kwds)
    lmdb_txn.put(key=str(key).encode("utf-8"), value=in_mem_file.getvalue())


def load_npz_from_lmdb(lmdb_txn, key):
    """Load and deserialize npz file from lmdb"""
    kwds = load_np_array_from_lmdb(lmdb_txn, key)
    return kwds


