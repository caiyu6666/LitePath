import torch
import numpy as np
import lmdb
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import time

from datasets import FeatureExtractDataset
from models import get_model, get_custom_transformer


def save_features_to_lmdb(env, key_list, feature_batch):
    with env.begin(write=True) as txn:
        for key, feat in zip(key_list, feature_batch):
            value = feat.astype(np.float32).tobytes()
            txn.put(key, value)


def find_continue_idx(env):
    with env.begin(write=False) as txn:
        stat = txn.stat()
        entries = stat['entries']
    return entries


def worker(rank, world_size, save_dir, teacher_names, dataset_root):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    save_files_dict = {name: f"{save_dir}/{name}_rank{rank}.lmdb" for name in teacher_names}
    env_dict = {name: lmdb.open(save_files_dict[name], map_size=int(3 * 1024 ** 4), readahead=False) for name in teacher_names}

    teacher_models = {name: get_model(name, device, 1) for name in teacher_names}
    transform_dict = {k: get_custom_transformer(k) for k in teacher_names}
    base_dataset = FeatureExtractDataset(root=dataset_root, transform_dict=transform_dict)

    batch_size = 512
    indices = list(range(rank, len(base_dataset), world_size))
    dataset = Subset(base_dataset, indices)

    # Skip already processed images
    continue_idx = [find_continue_idx(env_dict[name]) for name in teacher_names]
    continue_idx = min(continue_idx)
    print(f"Rank{rank} Continue index: {continue_idx}/{len(dataset)}")
    dataset = Subset(dataset, range(continue_idx, len(dataset)))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    batch_len = len(dataloader)
    key_cache = None
    features_cache = {k: None for k in teacher_names}
    t0 = time.time()
    with torch.inference_mode():
        for i, (keys, images) in enumerate(dataloader):
            key_cache = keys if key_cache is None else key_cache + keys
            images = {k: v.cuda(non_blocking=True) for k, v in images.items()}
            # Extract features
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for name, model in teacher_models.items():
                    features = model(images[name]).cpu().numpy()
                    features_cache[name] = features if features_cache[name] is None else np.concatenate(
                        (features_cache[name], features), axis=0)

            if (i+1) % 20 == 0:
                extract_time = time.time() - t0
                t0 = time.time()
                for name, env in env_dict.items():
                    save_features_to_lmdb(env, key_cache, features_cache[name])
                    features_cache[name] = None
                key_cache = None
                save_time = time.time() - t0
                print(f"Rank {rank}, Batch {i+1}/{batch_len}, Extract Time: {extract_time:.2f}s, Save Time: {save_time:.2f}s")
                t0 = time.time()

        if key_cache is not None:
            t0 = time.time()
            for name, env in env_dict.items():
                save_features_to_lmdb(env, key_cache, features_cache[name])
            save_time = time.time() - t0
            print(
                f"Rank {rank}, Batch {i + 1}/{batch_len}, Extract Time: {extract_time:.2f}s, Save Time: {save_time:.2f}s")

        print(f"Rank {rank} finished extracting features.")

    # env close
    for env in env_dict.values():
        env.close()


def main():
    save_dir = '/scratch/timnhaoprj/ycaibt/pretrain_features/'
    teacher_names = ['virchow2', 'h-optimus-1', 'uni2']
    dataset_root = '/project/vcompath/storage/Pathology/exclude_split1_dict.json'

    world_size = torch.cuda.device_count()
    mp.spawn(worker, args=(world_size, save_dir, teacher_names, dataset_root),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
