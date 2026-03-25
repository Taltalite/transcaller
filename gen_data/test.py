import torch
ckpt = torch.load("/home/lijy/workspace/bonito/bonito/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0/weights_1.tar")  # 或者 .pt
print(list(ckpt.keys())[0:5]) 