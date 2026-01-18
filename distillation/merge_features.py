import lmdb
from tqdm import tqdm
from multiprocessing import Process, Queue


def reader_worker(src_file, out_q):
    env_src = lmdb.open(src_file, readonly=True, lock=False, max_readers=126)
    with env_src.begin() as txn_src, txn_src.cursor() as cursor:
        for key, value in cursor:
            out_q.put((key, value))
    out_q.put(None)  # 结束信号
    env_src.close()

def writer_worker(out_q, env_merged, num_readers, total_entries=None):
    txn_merged = env_merged.begin(write=True)
    txn_batch_size = 1000000  # 每100万条commit一次，可根据内存调整
    count = 0
    end_signals = 0
    pbar = tqdm(total=total_entries, desc="Merging", mininterval=2.0) if total_entries else None

    while end_signals < num_readers:
        item = out_q.get()
        if item is None:
            end_signals += 1
            continue
        key, value = item
        txn_merged.put(key, value)
        count += 1
        if pbar:
            pbar.update(1)
        if count % txn_batch_size == 0:
            txn_merged.commit()
            txn_merged = env_merged.begin(write=True)
    txn_merged.commit()
    if pbar:
        pbar.close()
    print(f"Total entries merged: {count}")

def count_all_entries(files):
    total = 0
    for src_file in files:
        env_src = lmdb.open(src_file, readonly=True, lock=False)
        with env_src.begin() as txn_src:
            total += txn_src.stat()['entries']
        env_src.close()
    return total


def merge_features(model_name):
    source_files = [f"/scratch/timnhaoprj/ycaibt/pretrain_features/{model_name}_rank{i}.lmdb" for i in range(8)]
    merged_file = f'/scratch/timnhaoprj/ycaibt/pretrain_features/{model_name}.lmdb'
    map_size = 3 * 1024 ** 4

    print(f"Start merging {len(source_files)} LMDB files into {merged_file}")
    total_entries = count_all_entries(source_files)
    print(f"Total entries to merge: {total_entries}")

    env_merged = lmdb.open(merged_file, map_size=map_size, writemap=False, sync=True, max_dbs=1)

    q = Queue(maxsize=10000)  # 防止内存爆炸

    # 启动reader进程
    readers = []
    for f in source_files:
        p = Process(target=reader_worker, args=(f, q))
        p.start()
        readers.append(p)

    # 主进程写入
    writer_worker(q, env_merged, len(source_files), total_entries)

    for p in readers:
        p.join()
    env_merged.close()
    print(f"All LMDB files of {model_name} merged successfully.")


if __name__ == "__main__":
    model_list = ['virchow2', 'h-optimus-1', 'uni2']
    for model_name in model_list:
        merge_features(model_name)
    print("All LMDB files merged successfully.")
