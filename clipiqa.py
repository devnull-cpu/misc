import torch
from pyiqa.archs.clipiqa_arch import CLIPIQA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CLIPIQA(model_type='clipiqa+', backbone='RN50', pretrained=True)
model = model.to(device)
model.eval()
model.clip_model[0] = model.clip_model[0].to(device)

example = torch.randn(4, 3, 224, 224).to(device)

with torch.no_grad():
    traced = torch.jit.trace(model, example, check_trace=False)

traced.save("clipiqa.pt")