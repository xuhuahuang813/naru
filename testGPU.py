import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = torch.rand(10000000, 10000000).to(device)
y = x @ x
print("Done on", device)
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
