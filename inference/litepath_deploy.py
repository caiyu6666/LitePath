import torch
from models import get_model
from models import AdaPatchSelector, DAttention
import time


class ModelDeployment:
    def __init__(self, model_name, n_classes, k_a, k_u, aps_ckpt=None, mil_ckpt=None, buffer_threshold=2500):
        super(ModelDeployment, self).__init__()
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        n_features_dict = {
            'LiteFM': 1024,
        }
        n_aps_features_dict = {
            'LiteFM': 768,
            'LiteFM-L': 1536,
        }
        patch_shape_dict = {
            'LiteFM': (197, 384),
        }

        self.batch_size = 256
        self.stage2_batch_size = 512

        self.patch_shape = patch_shape_dict[model_name]
        self.n_features = n_features_dict[model_name]
        self.n_aps_features = n_aps_features_dict[model_name]
        self.litefm_model = get_model(model_name, self.device)

        self.model_name = model_name
        self.aps = AdaPatchSelector(in_dim=self.n_aps_features, out_dim=1).to(self.device)
        self.mil_model = DAttention(n_classes=n_classes, dropout=0.25, act="relu", n_features=self.n_features).to(self.device)

        aps_state_dict = torch.load(aps_ckpt, map_location=self.device)['state_dict']
        mil_state_dict = torch.load(mil_ckpt, map_location=self.device)['state_dict']
        self.aps.load_state_dict(aps_state_dict)
        self.mil_model.load_state_dict(mil_state_dict)
        print(f"Loaded APS model from {aps_ckpt}")
        print(f"Loaded MIL model from {mil_ckpt}")

        self.k_a = k_a
        self.k_u = k_u
        self.buffer_threshold = buffer_threshold
        assert self.k_a <= self.buffer_threshold, "num_score must be less than or equal to buffer_threshold"
        print(f"batch_size: {self.batch_size}, k_a: {self.k_a}, k_u: {self.k_u}, Buffer threshold: {self.buffer_threshold}")


    def load_models(self):
        if self.aps_ckpt:
            aps_state_dict = torch.load(self.aps_ckpt, map_location=self.device)['state_dict']
            self.aps.load_state_dict(aps_state_dict)
            print(f"Loaded APS model from {self.aps_ckpt}")
        else:
            print(f"APS ckpt not found")

        if self.mil_ckpt:
            mil_state_dict = torch.load(self.mil_ckpt, map_location=self.device)['state_dict']
            self.mil_model.load_state_dict(mil_state_dict)
            print(f"Loaded MIL model from {self.mil_ckpt}")
        else:
            print(f"MIL model not found")


    def infer_litepath(self, uniform_loader=None, attention_loader=None):
        t0 = time.time()
        if uniform_loader is not None:
            uniform_features = self.infer_feat_full(uniform_loader)
            print(f"Uniform features shape: {uniform_features.shape}, time cost: {time.time() - t0}")
        else:
            uniform_features = None
        
        t0 = time.time()
        if attention_loader is not None:
            attention_features = self.infer_feat_selection(attention_loader)
            print(f"Attention features shape: {attention_features.shape}, time cost: {time.time() - t0}")

        if uniform_features is not None and attention_features is not None:
            features_all = torch.cat([uniform_features, attention_features], dim=0)
        elif uniform_features is not None:
            features_all = uniform_features
        elif attention_features is not None:
            features_all = attention_features
        else:
            raise ValueError("No features to concatenate")
        
        t0 = time.time()
        logits = self.mil_model(features_all)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        print(f"MIL model inference time cost: {time.time() - t0}")
        return logits, prob, pred


    def infer_feat_full(self, loader):
        features = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, x in enumerate(loader):
                x = x.cuda()
                x = self.litefm_model(x)
                features.append(x)
        features = torch.cat(features, dim=0)
        return features


    def infer_feat_selection(self, loader):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            score_buffer = torch.empty(self.buffer_threshold+self.batch_size, device="cuda", dtype=torch.float16)
            patch_buffer = torch.empty(self.buffer_threshold+self.batch_size, *self.patch_shape, device="cuda", dtype=torch.float16)
            buffer_size = 0
            fisrt_flag = True

            for i, x in enumerate(loader):
                x = x.cuda()
                x = self.litefm_model.infer_deploy(x, stage="pre")  # bs, num_patches + 1, embed_dim

                patch_buffer[buffer_size:buffer_size+x.shape[0]] = x
                buffer_size += x.shape[0]
                
                if buffer_size >= self.buffer_threshold:
                    if fisrt_flag:
                        start_idx = 0
                        fisrt_flag = False
                    else:
                        start_idx = self.k_a

                    aps_input = torch.cat([patch_buffer[start_idx:buffer_size, 0], patch_buffer[start_idx:buffer_size, 1:].mean(1)], dim=1)  # bs, embed_dim * 2
                    score_buffer[start_idx:buffer_size] = self.aps(aps_input).squeeze(-1)  # [bs]

                    score_buffer[:self.k_a], topk_indices = torch.topk(score_buffer[:buffer_size], self.k_a)
                    patch_buffer[:self.k_a] = patch_buffer[topk_indices]
                    buffer_size = self.k_a
            
                # if i % 100 == 0:
                #     print(f"Processing batch {i} of {len(loader)}")
            
            if buffer_size > self.k_a:
                _, topk_indices = torch.topk(score_buffer[:buffer_size], self.k_a)
                patch_buffer = patch_buffer[topk_indices]


            # extract features of selected patches
            if self.stage2_batch_size is not None:
                # 批处理以避免内存不足
                processed_features = []
                for i in range(0, patch_buffer.shape[0], self.stage2_batch_size):
                    batch_features = patch_buffer[i:i+self.stage2_batch_size]
                    batch_features = self.litefm_model.infer_deploy(batch_features, stage="post")
                    processed_features.append(batch_features)
                features = torch.cat(processed_features, dim=0)
            else:
                features = self.litefm_model.infer_deploy(patch_buffer, stage="post")
        
        return features


    # def infer(self, wsi_loader):
    #     data_size = wsi_loader.dataset.__len__()
    #     all_indices = torch.arange(data_size)
    #     uniform_indices = torch.linspace(0, data_size-1, steps=self.num_uniform).int() if self.num_uniform > 0 else None
    #     print("dataset length: ", data_size)
    #     # remaining_indices = torch.tensor([i for i in all_indices if i not in uniform_indices]) if uniform_indices is not None else None

    #     if self.aps is None:
    #         self.aps = AdaPatchSelector(in_dim=self.n_aps_features, out_dim=1).to(self.device) if 'litepath' in self.model_name else None
    #     with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    #         # -----------------------stage 1: patch selection-----------------------
    #         print(f"Stage 1: start patch selection ...")
    #         uniform_patch = [] if uniform_indices is not None else None

    #         score_buffer = torch.empty(self.buffer_threshold+self.batch_size, device="cuda", dtype=torch.float16)
    #         patch_buffer = torch.empty(self.buffer_threshold+self.batch_size, *self.patch_shape, device="cuda", dtype=torch.float16)
    #         buffer_size = 0
    #         fisrt_flag = True
    #         t0 = time.time()
    #         for i, x in enumerate(wsi_loader):
    #             # x: bs, 3, 224, 224
    #             x = x.cuda()
    #             x = self.litepath.infer_deploy(x, stage="pre")  # bs, num_patches + 1, embed_dim

    #             if uniform_indices is not None:
    #                 start_idx = i*self.batch_size
    #                 end_idx = start_idx + x.shape[0]
    #                 batch_idx = all_indices[start_idx:end_idx]
    #                 mask_uniform = torch.isin(batch_idx, uniform_indices)
    #                 uniform_x = x[mask_uniform]
    #                 score_x = x[~mask_uniform]
    #                 uniform_patch.append(uniform_x)
    #             else:
    #                 score_x = x

    #             # patch_buffer[buffer_size:buffer_size+x.shape[0]] = x
    #             # buffer_size += x.shape[0]
    #             patch_buffer[buffer_size:buffer_size+score_x.shape[0]] = score_x
    #             buffer_size += score_x.shape[0]
                
    #             if buffer_size >= self.buffer_threshold:
    #                 if fisrt_flag:
    #                     start_idx = 0
    #                     fisrt_flag = False
    #                 else:
    #                     start_idx = self.num_score

    #                 aps_input = torch.cat([patch_buffer[start_idx:buffer_size, 0], patch_buffer[start_idx:buffer_size, 1:].mean(1)], dim=1)  # bs, embed_dim * 2
    #                 score_buffer[start_idx:buffer_size] = self.aps(aps_input).squeeze(-1)  # [bs]

    #                 score_buffer[:self.num_score], topk_indices = torch.topk(score_buffer[:buffer_size], self.num_score)
    #                 patch_buffer[:self.num_score] = patch_buffer[topk_indices]
    #                 buffer_size = self.num_score
            
    #             if i % 100 == 0:
    #                 print(f"Processing batch {i} of {len(wsi_loader)}")
            
    #         if buffer_size > self.num_score:
    #             _, topk_indices = torch.topk(score_buffer[:buffer_size], self.num_score)
    #             patch_buffer = patch_buffer[topk_indices]
            
    #         # 保存用于打印的信息（在删除之前）
    #         score_count = patch_buffer.shape[0]
            
    #         if uniform_indices is not None:
    #             uniform_patch.append(patch_buffer)
    #             features = torch.cat(uniform_patch, dim=0)
    #             # # 清理不再需要的缓冲区
    #             # del uniform_patch
    #             # torch.cuda.empty_cache()
    #         else:
    #             features = patch_buffer

    #         # # 清理不再需要的缓冲区
    #         # del patch_buffer, score_buffer
    #         # torch.cuda.empty_cache()

    #         print(f"Stage 1: selected features shape {features.shape} (score: {score_count}), \
    #                 time cost: {time.time() - t0}")

    #         # -----------------------stage 2: finish feature extraction of selected patches-----------------------
    #         if self.stage2_batch_size is not None:
    #             # 批处理以避免内存不足
    #             processed_features = []
    #             for i in range(0, features.shape[0], self.stage2_batch_size):
    #                 batch_features = features[i:i+self.stage2_batch_size]
    #                 batch_features = self.litepath.infer_deploy(batch_features, stage="post")
    #                 processed_features.append(batch_features)
    #             features = torch.cat(processed_features, dim=0)
    #         else:
    #             features = self.litepath.infer_deploy(features, stage="post")
    #         print(f"Stage 2: full feature extraction of selected patches {features.shape}, time cost: {time.time() - t0}")

    #         # -----------------------stage 3: finish MIL model inference-----------------------
    #         logits = self.mil_model(features)
    #         spend_time = time.time() - t0
    #         print(f"Stage 3: MIL model inference time cost: {spend_time}s")

    #     return logits, spend_time


    # def infer_all(self, wsi_loader):
    #     features = []
    #     # try:
    #     t0 = time.time()
    #     with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    #         for i, x in enumerate(wsi_loader):
    #             if i % 100 == 0:
    #                 print(f"Processing batch {i} of {len(wsi_loader)}")
    #             x = x.cuda()
    #             out = self.litepath(x)
    #             # features.append(out.cpu())
    #             features.append(out)
    #         features = torch.cat(features, dim=0)

    #         # features = features.cuda()
    #         logits = self.mil_model(features)
        
    #     spend_time = time.time() - t0
    #     print(f"time cost: {spend_time}s")
        
    #     return logits, spend_time


    # def infer_aps(self, uniform_loader, att_loader):
    #     uniform_features = []
    #     with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    #         if uniform_loader is not None:
    #             for i, x in enumerate(uniform_loader):
    #                 x = x.cuda()
    #                 x = self.litepath(x)
    #                 uniform_features.append(x)

    #         if att_loader is not None:
    #             print(f"Stage 1: start patch selection ...")
    #             score_buffer = torch.empty(self.buffer_threshold+self.batch_size, device="cuda", dtype=torch.float16)
    #             patch_buffer = torch.empty(self.buffer_threshold+self.batch_size, *self.patch_shape, device="cuda", dtype=torch.float16)
    #             buffer_size = 0
    #             fisrt_flag = True
    #             t0 = time.time()
    #             for i, x in enumerate(att_loader):
    #                 # x: bs, 3, 224, 224
    #                 x = x.cuda()
    #                 x = self.litepath.infer_deploy(x, stage="pre")  # bs, num_patches + 1, embed_dim

    #                 patch_buffer[buffer_size:buffer_size+x.shape[0]] = x
    #                 buffer_size += x.shape[0]
    #                 if buffer_size >= self.buffer_threshold:
    #                     if fisrt_flag:
    #                         start_idx = 0
    #                         fisrt_flag = False
    #                     else:
    #                         start_idx = self.select_num

    #                     aps_input = torch.cat([patch_buffer[start_idx:buffer_size, 0], patch_buffer[start_idx:buffer_size, 1:].mean(1)], dim=1)  # bs, embed_dim * 2
    #                     score_buffer[start_idx:buffer_size] = self.aps(aps_input).squeeze(-1)  # [bs]

    #                     score_buffer[:self.select_num], topk_indices = torch.topk(score_buffer[:buffer_size], self.select_num)
    #                     patch_buffer[:self.select_num] = patch_buffer[topk_indices]
    #                     buffer_size = self.select_num
                    
    #                 if i % 100 == 0:
    #                     print(f"Processing batch {i} of {len(att_loader)}")
                
    #             if buffer_size > self.select_num:
    #                 _, topk_indices = torch.topk(score_buffer[:buffer_size], self.select_num)
    #                 patch_buffer = patch_buffer[topk_indices]

    #             print(f"Stage 1: selected patch_buffer shape {patch_buffer.shape}, time cost: {time.time() - t0}")

    #             # -----------------------stage 2: finish feature extraction of selected patches-----------------------
    #             features = self.litepath.infer_deploy(patch_buffer, stage="post")
    #             print(f"Stage 2: full feature extraction of selected patches {features.shape}, time cost: {time.time() - t0}")

    #         features = torch.cat(features, dim=0)
