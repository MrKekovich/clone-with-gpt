import pytest
import torch


@pytest.fixture(autouse=True)
def setup_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    print(f"\nPyTorch is using device: {device}\n")
    # for reproducibility
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    # Makes calculations more reproducible, but slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
