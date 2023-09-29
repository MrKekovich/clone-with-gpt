from pathlib import Path

import pytest
import torch
from tokenizers import Encoding

from gpt.tokenizer import GPTTokenizer
from utils.exceptions import InvalidSaveDirectoryError


class TestGPTTokenizer:

    def test_init_valid(self):
        tokenizer = GPTTokenizer(Path('vocab.json'))
        assert tokenizer.vocab_path == Path('vocab.json')
        assert str(tokenizer.vocab_path) == 'vocab.json'

    def test_init_invalid(self):
        with pytest.raises(TypeError):
            GPTTokenizer(123)

    def test_vocab_path_getter(self):
        tokenizer = GPTTokenizer()
        assert tokenizer.vocab_path is None

    def test_vocab_path_setter(self):
        tokenizer = GPTTokenizer()
        tokenizer.vocab_path = Path('new_vocab.json')
        assert tokenizer.vocab_path == Path('new_vocab.json')

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
        assert isinstance(text_from_tensor, str)

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
        assert isinstance(encodings[0], Encoding)

    def test_encode_batch_invalid(self):
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer.encode_batch('invalid')

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

    def test_save(self):
        tokenizer = GPTTokenizer()
        save_path = Path('test_data/tokenizer/saves/')
        vocab_path = tokenizer.save(save_path)
        assert vocab_path.exists()

    def test_save_invalid(self):
        tokenizer = GPTTokenizer()
        with pytest.raises(TypeError):
            tokenizer.save('string.txt')
        with pytest.raises(InvalidSaveDirectoryError):
            tokenizer.save(Path('test_data/tokenizer/not_a_dir.txt'))

    def test_train(self):
        save_path = Path('test_data/tokenizer/saves/')
        tokenizer = GPTTokenizer()
        tokenizer.train(dataset_path=Path('dataset'),
                        save_path=save_path)
        assert (save_path / 'vocab.json').exists()
        assert tokenizer.vocab_path == save_path / 'vocab.json'
