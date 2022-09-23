import argparse
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ckpt_out",
        type=str,
        default="models/ldm/stable-diffusion-v1/model_native_mha.ckpt",
        help="path to the output checkpoint of model using native MHA",
    )
    opt = parser.parse_args()

    config = OmegaConf.load(f"{opt.config}")
    ckpt = f"{opt.ckpt}"
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    weights_to_rename = set()
    for k in pl_sd["state_dict"]:
        if 'attn1' in k:
            weights_to_rename.add(str(k).split('.attn1')[0])

    for k in weights_to_rename:
        pl_sd["state_dict"]['{}.attn1.mha.in_proj_weight'.format(k)] = torch.cat([
            pl_sd["state_dict"]['{}.attn1.to_q.weight'.format(k)],
            pl_sd["state_dict"]['{}.attn1.to_k.weight'.format(k)],
            pl_sd["state_dict"]['{}.attn1.to_v.weight'.format(k)]
        ])
        pl_sd["state_dict"]['{}.attn1.mha.out_proj.weight'.format(k)] = pl_sd["state_dict"]['{}.attn1.to_out.0.weight'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn1.mha.out_proj.bias'.format(k)] = pl_sd["state_dict"]['{}.attn1.to_out.0.bias'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn2.mha.q_proj_weight'.format(k)] = pl_sd["state_dict"]['{}.attn2.to_q.weight'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn2.mha.k_proj_weight'.format(k)] = pl_sd["state_dict"]['{}.attn2.to_k.weight'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn2.mha.v_proj_weight'.format(k)] = pl_sd["state_dict"]['{}.attn2.to_v.weight'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn2.mha.out_proj.weight'.format(k)] = pl_sd["state_dict"]['{}.attn2.to_out.0.weight'.format(k)].clone()
        pl_sd["state_dict"]['{}.attn2.mha.out_proj.bias'.format(k)] = pl_sd["state_dict"]['{}.attn2.to_out.0.bias'.format(k)].clone()

        for i in ['attn1', 'attn2']:
            pl_sd["state_dict"].pop('{}.{}.to_q.weight'.format(k, i))
            pl_sd["state_dict"].pop('{}.{}.to_k.weight'.format(k, i))
            pl_sd["state_dict"].pop('{}.{}.to_v.weight'.format(k, i))
            pl_sd["state_dict"].pop('{}.{}.to_out.0.weight'.format(k, i))
            pl_sd["state_dict"].pop('{}.{}.to_out.0.bias'.format(k, i))

    torch.save(pl_sd, f"{opt.ckpt_out}")


if __name__ == "__main__":
    main()
