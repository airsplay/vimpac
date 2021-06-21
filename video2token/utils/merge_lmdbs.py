import os
from os.path import join, exists
import lmdb
import pprint
from tqdm import tqdm
from video2token.utils.basic_utils import save_json


def merge_lmdb_files(lmdb_paths, save_merged_path, use_idx_as_prefix=False):
    """Merge multiple LMDB file together
    use_idx_as_prefix: bool, add idx of the file (inferred from the input order)
        as the prefix to the original lmdb key.
        This should be used when the LMDB files has the keys.
    """
    env = lmdb.open(save_merged_path, map_size=1024 ** 4 * 2.5)
    txn = env.begin(write=True)

    for lmdb_file_idx, lmdb_path in enumerate(lmdb_paths):
        print(f"Processing {lmdb_file_idx+1}-th file {lmdb_path}")
        _env = lmdb.open(lmdb_path, readonly=True, create=False, lock=False)
        _txn = _env.begin(buffers=False)
        lmdb_iterator = tqdm(
            enumerate(_txn.cursor().iternext()), total=_env.stat()["entries"],
            desc=f"Processing [{lmdb_file_idx+1}/{len(lmdb_paths)}]")
        for idx, (k, v) in lmdb_iterator:
            new_k = f"{lmdb_file_idx}_{k.decode()}" if use_idx_as_prefix else k.decode()
            txn.put(key=new_k.encode("utf-8"), value=v)
            if idx % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.commit()
    env.close()

    with open(join(save_merged_path, "files_merged.txt"), "w") as f:
        f.write("\n".join(lmdb_paths))


def main_merge():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_paths", type=str, nargs="+", help="LMDB files to be merged")
    parser.add_argument("--save_merged_path", type=str, help="path to dir saving the merged LMDB file")
    parser.add_argument("--use_idx_as_prefix", action="store_true",
                        help="add idx prefix to the original lmdb key. "
                             "should be used when the LMDB files shares some keys")
    args = parser.parse_args()
    assert not exists(args.save_merged_path) or len(os.listdir(args.save_merged_path)) == 0, \
        f"Directory {args.save_merged_path} exists and is not empty"
    os.makedirs(args.save_merged_path, exist_ok=True)
    save_json(vars(args), join(args.save_merged_path, "args.json"), save_pretty=True)
    pprint.pprint(vars(args))
    merge_lmdb_files(args.lmdb_paths, args.save_merged_path,
                     use_idx_as_prefix=args.use_idx_as_prefix)


if __name__ == '__main__':
    main_merge()
