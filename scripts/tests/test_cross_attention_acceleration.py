import unittest
import torch
import torch.nn as nn
from ldm.modules.attention import CrossAttention


class TestCrossAttentionAcceleration(unittest.TestCase):

    def test_cross_attention(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("mps") if torch.backends.mps.is_available() else device

        # Generate original and accelerated CrossAttention modules with the same random weights
        cross_attention_orig = CrossAttention(512)
        cross_attention_acc = CrossAttention(512, accelerated=True)
        cross_attention_acc.to_out.in_proj_weight = nn.Parameter(torch.cat([
            cross_attention_orig.to_q.weight.detach().clone(),
            cross_attention_orig.to_k.weight.detach().clone(),
            cross_attention_orig.to_v.weight.detach().clone()
        ]))
        cross_attention_acc.to_out.out_proj.weight = nn.Parameter(cross_attention_orig.to_out.get_parameter("0.weight").detach().clone())
        cross_attention_acc.to_out.out_proj.bias = nn.Parameter(cross_attention_orig.to_out.get_parameter("0.bias").detach().clone())
        cross_attention_orig = cross_attention_orig.to(device)
        cross_attention_acc = cross_attention_acc.to(device)

        # Generate Random Inputs
        data_in = torch.rand((16, 128, 512)).to(device)

        # Generate results of original and accelerated CrossAttention modules
        data_out_orig = cross_attention_orig(data_in).detach().cpu()
        data_out_acc = cross_attention_acc(data_in).detach().cpu()
        self.assertEqual(torch.gt(torch.abs(data_out_orig.data - data_out_acc.data), 1e-6).sum(), 0)


if __name__ == '__main__':
    unittest.main()
