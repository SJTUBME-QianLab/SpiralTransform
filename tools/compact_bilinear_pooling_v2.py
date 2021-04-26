from __future__ import division
import torch
import torch.nn as nn

import torch.nn.functional as F

torch.manual_seed(0)  # 固定随机数种子
torch.cuda.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# input_dim1 = 256
# input_dim2 = 256
# output_dim = 256*4

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(
            torch.stack([torch.arange(input_dim, out=torch.LongTensor()), rand_h.long()]),
            rand_s.float(),
            [input_dim, output_dim]).to_dense()
        self.sketch_matrix01 = torch.nn.Parameter(
            generate_sketch_matrix(torch.randint(output_dim, size=(input_dim1,)),
                                   2 * torch.randint(2, size=(input_dim1,)) - 1,
                                   input_dim1,
                                   output_dim))
        self.sketch_matrix02 = torch.nn.Parameter(
            generate_sketch_matrix(torch.randint(output_dim, size=(input_dim2,)),
                                   2 * torch.randint(2, size=(input_dim2,)) - 1,
                                   input_dim2,
                                   output_dim))
        # torch.save(sketch_matrix01,'./tool/sketch_matrix1.pth')
        # torch.save(sketch_matrix02,'./tool/sketch_matrix2.pth')
        # sketch_matrix01 = torch.load('./tool/sketch_matrix1.pth')
        # sketch_matrix02 = torch.load('./tool/sketch_matrix2.pth')

    def forward(self, x, y):
        output_dim = self.output_dim
        if y is None:
            y = x
        sketch_matrix1 = self.sketch_matrix01.cuda()
        sketch_matrix2 = self.sketch_matrix02.cuda()
        fft1 = torch.rfft(x.matmul(sketch_matrix1), 1)  # [32, 2, 2, 512, 2]  [32, 4, 8, 1025, 2]
        fft2 = torch.rfft(y.matmul(sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1],
                                   fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
        cbp = torch.irfft(fft_product, 1, signal_sizes=(output_dim,)) * output_dim  # [32, 2, 2, 1024]

        out = cbp.sum(dim=1).sum(dim=1)  # [32, 1024]  [32, 2048]
        out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out))   # out = sign(out)*sqrt(|x|)
        out = torch.nn.functional.normalize(out)   # out=out/||out||2
        return out


class BilinearPooling(nn.Module):
    def forward(self, x, y):
        if y is None:
            y = x
        x = x.view(x.size()[0], x.size()[1], x.size()[2] * x.size()[3])
        y = y.view(y.size()[0], y.size()[1], y.size()[2] * y.size()[3])
        out = torch.bmm(x, torch.transpose(y, 1, 2))  # Bilinear [32, 512, 512]
        out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out))  # out = sign(out)*sqrt(|x|)
        # print(out.max(), out.min())
        out = torch.nn.functional.normalize(out)  # out=out/||out||2
        # print(out.max(), out.min())
        out = out.reshape(out.size(0), -1)  # [32*512*512]
        return out
