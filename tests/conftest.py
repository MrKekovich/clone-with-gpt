import time

import pytest
import torch


@pytest.fixture
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


@pytest.fixture(scope="session", autouse=True)
def setup_test_folder(request, tmp_path_factory):
    import shutil
    print("Setting up test folder")
    test_data_dir = tmp_path_factory.mktemp("test_data")

    yield test_data_dir
    print("Cleaning up test folder")
    shutil.rmtree(test_data_dir)
    print("Test folder cleaned")
