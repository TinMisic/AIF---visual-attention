import torch

vae = torch.load("vae-entangled.pt")
torch.save(vae.state_dict(),"vae-entangled_state_dict.pt")

#fc =torch.load("intentions.pt")
#torch.save(fc.state_dict(),"intentions_state_dict.pt")
