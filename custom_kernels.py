import torch

kernel1 = torch.tensor([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], dtype=torch.float32)

kernel2 = torch.tensor([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]], dtype=torch.float32)

kernel3 = torch.tensor([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]], dtype=torch.float32)

kernel4 = torch.tensor([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=torch.float32)

kernel5 = torch.tensor([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=torch.float32)

kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]