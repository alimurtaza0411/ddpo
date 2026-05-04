import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

base_model = DummyModel()
config = LoraConfig(target_modules=["linear"])
peft_model = get_peft_model(base_model, config)

print("Fresh PEFT requires_grad:")
for n, p in peft_model.named_parameters():
    if "lora" in n:
        print(f"  {n}: {p.requires_grad}")

peft_model.save_pretrained("dummy_lora")

base_model2 = DummyModel()
resumed_model = PeftModel.from_pretrained(base_model2, "dummy_lora", is_trainable=True)

print("\nResumed PEFT requires_grad with is_trainable=True:")
for n, p in resumed_model.named_parameters():
    if "lora" in n:
        print(f"  {n}: {p.requires_grad}")

