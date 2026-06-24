from huggingface_hub import hf_hub_download
from forwardwrapper import forward_wrapper
import timm
import torch.nn as nn
import torch

class ViTModel(nn.Module):
    def __init__(self, extract_attention_maps=True):
        super(ViTModel, self).__init__()

        self.extract_attention_maps = extract_attention_maps

        model_path = hf_hub_download(
            repo_id="Addax-Data-Science/Deepfaune_v1.2",
            filename="deepfaune-vit_large_patch14_dinov2.lvd142m.v2.pt"
        )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}

        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=30
        )

        self.model.load_state_dict(state_dict, strict=False)
        self.model.blocks[-1].attn.forward = forward_wrapper(self.model.blocks[-1].attn)

    def forward(self, x):
        output = self.model(x)

        if self.extract_attention_maps:
            last_attn = self.model.blocks[-1].attn
            cls_attn = last_attn.cls_attn_map
            cls_attn = cls_attn.mean(dim=1)

            grid_size = int(cls_attn.shape[-1] ** 0.5)
            cls_attn = cls_attn.reshape(-1, 1, grid_size, grid_size).detach().cpu().numpy() # [B, 1, 37, 37]    

            return output, cls_attn
        
        return output
    

def load_model(extract_attention):
    model = ViTModel(extract_attention_maps=extract_attention)
    return model
   