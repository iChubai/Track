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

pretrain_path = '/chenyuming/project/hjd/ARTrack/output_02_04/checkpoints/train/artrackv2/artrackv2_256_got_itpn/ARTrackV2_ep0120.pth.tar'

# 加载预训练模型权重
checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)
checkpoint_copy = checkpoint.copy()

# 打印所有键名
print("Keys in the pretrained model checkpoint:")
for key in checkpoint['net'].keys():
    print(key)

# 要删除的键
keys_to_delete = ['backbone.patch_embed_true.proj.weight', 'backbone.patch_embed_true.proj.bias']

# 删除指定的键
for key in keys_to_delete:
    if key in checkpoint_copy['net']:
        del checkpoint_copy['net'][key]

# 检查键是否存在
for key in keys_to_delete:
    if key in checkpoint_copy['net']:
        print(f"Key {key} exists in the checkpoint copy.")
    else:
        print(f"Key {key} does not exist in the checkpoint copy.")
