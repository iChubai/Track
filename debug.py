# import torch

# def print_checkpoint_info(checkpoint_path):
#     # 加载checkpoint
#     checkpoint = torch.load(checkpoint_path)
    
#     # 打印整体的checkpoint结构
#     print("Checkpoint Structure:")
#     print(f"Checkpoint contains the following keys: {list(checkpoint.keys())}\n")
    
#     # 遍历checkpoint中的每个元素，重点打印其shape和类型
#     for key, value in checkpoint.items():
#         print(f"Key: {key}")
        
#         if isinstance(value, dict):
#             # 如果值是字典，递归打印内部键值对
#             print(f"  Value is a dictionary, contains keys: {list(value.keys())}")
#             for subkey, subvalue in value.items():
#                 print(f"    Subkey: {subkey}, Type: {type(subvalue)}, Shape: {getattr(subvalue, 'shape', 'N/A')}")
#         else:
#             # 打印普通张量的类型和形状
#             print(f"  Type: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
#         print()

# # 设置Checkpoint文件路径
# checkpoint_path = "/home/ubuntu/fishworld/project/hjd/hjd/ARTrack/output/checkpoints/train/artrackv2/artrackv2_256_got_itpn/ARTrackV2_ep0120.pth.tar"

# # 调用函数打印Checkpoint详细信息
# print_checkpoint_info(checkpoint_path)
import torch

pretrain_path = '/mnt/sda/hjd/ARTrack/lib/train/pretrained_models/fast_itpn_tiny_1600e_1k.pt'

# 加载预训练模型权重
checkpoint = torch.load(pretrain_path, map_location='cpu')

# 打印所有键名
print("Keys in the pretrained model checkpoint:")
for key in checkpoint.keys():
    print(key)

# 检查特定键是否存在
key_to_check = 'net/backbone.patch_embed_true.proj.weight'
if key_to_check in checkpoint:
    print(f"Key {key_to_check} exists in the checkpoint.")
else:
    print(f"Key {key_to_check} does not exist in the checkpoint.")
