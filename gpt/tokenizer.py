from pathlib import Path
from typing import Union, Optional, List

import torch
from tokenizers import ByteLevelBPETokenizer, Encoding, InputSequence, EncodeInput

from utils.exceptions import InvalidDatasetPathError, NoFilesFoundError, InvalidSaveDirectoryError


class GPTTokenizer:
    """
    A tokenizer for GPT models.

    Args:
        vocab_path (Path, optional): The path to the vocabulary file. Defaults to None.

    Attributes:
        vocab_path (Path or str): The path to the vocabulary file.
        tokenizer (ByteLevelBPETokenizer): The ByteLevelBPETokenizer instance.
    """

    def __init__(self, vocab_path: Path = None):
        self.vocab_path = vocab_path

    @property
    def vocab_path(self):
        """
        The path to the vocabulary file.
        """
        return self._vocab_path

    @vocab_path.setter
    def vocab_path(self, value: Path):
        if (not isinstance(value, Path)
                and value is not None):
            raise TypeError("Vocabulary path must be instance of Path. "
                            f"{type(value)} : {value} was given.")

        self._vocab_path = value
        if value is None:
            self._tokenizer = ByteLevelBPETokenizer(vocab=None)
        else:
            self._tokenizer = ByteLevelBPETokenizer(vocab=str(value))

    @property
    def tokenizer(self):
        """
        The ByteLevelBPETokenizer instance.
        """
        return self._tokenizer

    def train(self,
              dataset_path: Union[Path],
              vocab_size: int = 53_320,
              min_frequency: int = 2,
              show_progress: bool = True,
              special_tokens: list[str] = None,
              save_path: Optional[Path] = None,
              save_pretty: bool = True) -> None:
        """
        Train the tokenizer.
        Args:
            dataset_path: Path to the dataset folder or file.
            save_path: Path to save the trained tokenizer.
            vocab_size: vocabulary size.
            min_frequency: minimum frequency of a token.
            show_progress: show progress bar.
            special_tokens: special tokens.
            save_pretty: save pretty json.
        """
        self._validate_dataset_path(dataset_path)
        paths = self._get_paths_from_dataset(dataset_path)

        self.tokenizer.train(files=paths,
                             vocab_size=vocab_size,
                             min_frequency=min_frequency,
                             show_progress=show_progress,
                             special_tokens=special_tokens)

        if save_path is not None:
            self.vocab_path = self.save(save_path=save_path,
                                        pretty=save_pretty)

    @staticmethod
    def _validate_dataset_path(dataset_path: Union[Path]) -> None:
        """
        Validate dataset path.
        Args:
            dataset_path: Path: The path to the dataset folder or file.

        Raises:
            TypeError: If the dataset path is not an instance of Path.
            FileNotFoundError: If the dataset path does not exist.
        """
        if not isinstance(dataset_path, Path):
            raise TypeError(f"Dataset path must be instance of Path. "
                            f"{type(dataset_path)}: {dataset_path} was given")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    @staticmethod
    def _get_paths_from_dataset(dataset_path: Union[Path]) -> List[str]:
        """
        Get paths from dataset.
        Args:
            dataset_path: Path: The path to the dataset folder or file.

        Returns:
            List[str]: List of paths

        Raises:
            InvalidDatasetPathError: If the dataset path is invalid (Not a folder nor a file).
            NotFoundError: If no .txt files found at the path address.
        """
        if dataset_path.is_dir():
            paths = [str(path) for path in dataset_path.glob("**/*.txt")]
        elif dataset_path.is_file():
            paths = ([str(dataset_path)]
                     if dataset_path.suffix == ".txt"
                     else [])
        else:
            raise InvalidDatasetPathError(f"Path should be a folder or a file. "
                                          f"{dataset_path} was given")

        if len(paths) == 0:
            raise NoFilesFoundError(f"No txt files found at {dataset_path}")

        return paths

    def save(self, save_path: Path, pretty: bool = True) -> Path:
        """
        Save the tokenizer.
        Args:
            save_path: Path to save the tokenizer.
            pretty: save pretty json.

        Returns:
            Path: The path to the saved tokenizer

        Raises:
            InvalidSaveDirectoryError: If the save path is not a directory
        """
        if not isinstance(save_path, Path):
            raise TypeError(f"Save path must be instance of Path. "
                            f"{type(save_path)}: {save_path} was given")
        if not save_path.is_dir():
            raise InvalidSaveDirectoryError(f"The path should be a directory. {save_path} was given")
        save_path.mkdir(parents=True, exist_ok=True)
        return self.tokenizer.save(path=str(save_path),
                                   pretty=pretty)
        # return save_path / "vocab.json"

    def encode(self,
               sequence: InputSequence,
               pair: Optional[InputSequence] = None,
               is_pretokenized: bool = False,
               add_special_tokens: bool = True,
               ) -> Encoding:
        """
        Encode the given sequence and pair. This method can process raw text sequences as well
        as already pre-tokenized sequences.

        Args:
            sequence: InputSequence:
                The sequence we want to encode. This sequence can be either raw text or
                pre-tokenized, according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            pair: Optional[InputSequence]: An optional input sequence.
            The expected format is the same that for sequence.

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            An Encoding

        Raises:
            ValueError: If `sequence` is `None`.
        """
        return self.tokenizer.encode(sequence=sequence,
                                     pair=pair,
                                     is_pretokenized=is_pretokenized,
                                     add_special_tokens=add_special_tokens)

    def encode_batch(self,
                     inputs: List[EncodeInput],
                     is_pretokenized: bool = False,
                     add_special_tokens: bool = True) -> List[Encoding]:
        """
        Encode the given inputs. This method accept both raw text sequences as well as already
        pre-tokenized sequences.

        Args:
            inputs: List[EncodeInput]:
                A list of single sequences or pair sequences to encode. Each `EncodeInput` is
                expected to be of the following form:
                    `Union[InputSequence, Tuple[InputSequence, InputSequence]]`

                Each `InputSequence` can either be raw text or pre-tokenized,
                according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            A list of Encoding

        Raises:
            ValueError: If `inputs` is `None`.
        """
        return self.tokenizer.encode_batch(inputs=inputs,
                                           is_pretokenized=is_pretokenized,
                                           add_special_tokens=add_special_tokens)

    def decode(self,
               ids: Union[list[int], torch.Tensor],
               skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: Union[list[int], torch.Tensor]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string

        Raises:
            ValueError: If `ids` is `None`
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids=ids,
                                     skip_special_tokens=skip_special_tokens)

    def __call__(self,
                 data: Union[str, list[str]],
                 add_special_tokens: bool = True) -> Union[Encoding, list[Encoding]]:
        """
        Encodes the given data.
        Args:
            data: Union[str, list[str]]: The data to be encoded.
            add_special_tokens: bool: Whether to add the special tokens.

        Returns:
            Union[Encoding, list[Encoding]]: The encoded data
        """
        if isinstance(data, str):
            return self.encode(data, add_special_tokens=add_special_tokens)
        else:
            return self.encode_batch(data, add_special_tokens=add_special_tokens)

    def __len__(self):
        """
        Returns:
            int: The size of the vocabulary
        """
        return self.tokenizer.get_vocab_size(with_added_tokens=True)
