from pathlib import Path

import pytest
import torch
from tokenizers import Encoding

from gpt.tokenizer import GPTTokenizer


class TestGPTTokenizer:

    def test_init_valid(self, tmp_path):
        vocab_path = tmp_path / "vocab.json"
        vocab_path.touch()
        tokenizer = GPTTokenizer(vocab_path)
        assert tokenizer.vocab_path == vocab_path

    def test_init_invalid(self):
        with pytest.raises(TypeError):
            GPTTokenizer(123)

    def test_vocab_path_getter(self):
        tokenizer = GPTTokenizer()
        assert tokenizer.vocab_path is None

    def test_vocab_path_setter_valid(self, tmp_path):
        new_vocab_path = tmp_path / "new_vocab.json"
        new_vocab_path.touch()
        tokenizer = GPTTokenizer()
        tokenizer.vocab_path = new_vocab_path

    def test_vocab_path_setter_invalid(self, tmp_path):
        tokenizer = GPTTokenizer()
        not_a_path = "string.txt"
        new_folder = tmp_path / "vocab_cant_be_a_folder/"
        file_does_not_exist = tmp_path / "i_do_not_exist.json"

        new_folder.mkdir()
        with pytest.raises(TypeError):
            tokenizer.vocab_path = not_a_path
        with pytest.raises(FileNotFoundError):
            tokenizer.vocab_path = file_does_not_exist
        with pytest.raises(ValueError):
            tokenizer.vocab_path = new_folder


    def test_encode_valid(self):
        tokenizer = GPTTokenizer()
        encoding = tokenizer.encode('This is a test')
        assert isinstance(encoding, Encoding)

    def test_encode_invalid(self):
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer.encode(123)
        with pytest.raises(ValueError):
            tokenizer.encode(None)

    def test_decode_valid(self):
        tokenizer = GPTTokenizer()
        text_from_list = tokenizer.decode([1, 2, 3])

        text_from_dtype_short_tensor = tokenizer.decode(torch.tensor([1, 2, 3],
                                                                     dtype=torch.short))
        text_from_tensor = tokenizer.decode(torch.tensor([1, 2, 3]))
        text_from_dtype_long_tensor = tokenizer.decode(torch.tensor([1, 2, 3],
                                                                    dtype=torch.long))
        assert isinstance(text_from_list, str)
        assert isinstance(text_from_dtype_short_tensor, str)
        assert isinstance(text_from_tensor, str)
        assert isinstance(text_from_dtype_long_tensor, str)

    def test_decode_invalid(self):
        tokenizer = GPTTokenizer()
        invalid_string = 'abc'
        invalid_integer = 123
        invalid_matrix_tensor = torch.tensor([[1, 2, 3],
                                              [4, 5, 6]])
        invalid_dtype_float32_tensor = torch.tensor([[1, 2, 3],
                                                     [4, 5, 6]], dtype=torch.float32)
        invalid_dtype_float64_tensor = torch.tensor([[1, 2, 3],
                                                     [4, 5, 6]], dtype=torch.float64)
        with pytest.raises(TypeError):
            tokenizer.decode(invalid_string)
        with pytest.raises(TypeError):
            tokenizer.decode(invalid_integer)
        with pytest.raises(TypeError):
            tokenizer.decode(invalid_matrix_tensor)
        with pytest.raises(TypeError):
            tokenizer.decode(invalid_dtype_float32_tensor)
        with pytest.raises(TypeError):
            tokenizer.decode(invalid_dtype_float64_tensor)

    def test_encode_batch_valid(self):
        tokenizer = GPTTokenizer()
        encodings = tokenizer.encode_batch(['This is a test', 'Second sequence'])
        assert isinstance(encodings, list)
        assert all(isinstance(encoding, Encoding) for encoding in encodings)

    def test_encode_batch_invalid(self):
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer.encode_batch('not a list')

    def test_len(self):
        tokenizer = GPTTokenizer()
        assert isinstance(len(tokenizer), int)

    def test_call_valid(self):
        tokenizer = GPTTokenizer()
        encoding = tokenizer('This is a test')
        assert isinstance(encoding, Encoding)

        encodings = tokenizer(['This is a test', 'Second sequence'])
        assert isinstance(encodings, list)
        assert all(isinstance(encoding, Encoding) for encoding in encodings)

    def test_call_invalid(self):
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer(123)

    def test_save(self, tmp_path):
        tokenizer = GPTTokenizer()
        save_path = tmp_path / 'saves/'
        vocab_path = tokenizer.save(save_path)
        assert vocab_path.exists()

    def test_save_invalid(self, tmp_path):
        string_path = str(tmp_path / "string.txt")
        folder_does_not_exist = tmp_path / "new_folder"
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer.save(string_path)
        with pytest.raises(Exception):
            tokenizer.save(folder_does_not_exist, create_if_not_exist=False)

    def test_train_on_file(self, tmp_path):
        dataset_file = tmp_path / 'dataset.txt'
        dataset_file.write_text('test dataset')

        save_path = tmp_path / 'saves/'
        tokenizer = GPTTokenizer()
        tokenizer.train(dataset_path=dataset_file,
                        save_path=save_path)
        assert tokenizer.vocab_path.exists()
        assert tokenizer.vocab_path == save_path / 'vocab.json'

    def test_train_on_directory(self, tmp_path):
        dataset_folder = tmp_path / "dataset/"
        dataset_file = dataset_folder / 'dataset.txt'

        dataset_folder.mkdir(exist_ok=True)
        dataset_file.write_text('test dataset')

        save_path = tmp_path / "saves/"
        print(f"if this is true it's really bad {save_path.exists()}")
        tokenizer = GPTTokenizer()
        tokenizer.train(dataset_path=dataset_folder,
                        save_path=save_path)
        assert tokenizer.vocab_path.exists()
        assert tokenizer.vocab_path == save_path / 'vocab.json'
