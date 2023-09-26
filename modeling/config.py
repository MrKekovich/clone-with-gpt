from abc import ABC


class GPTConfig(ABC):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 num_heads: int = 12,
                 num_blocks: int = 12,
                 embed_dim: int = 768,
                 attn_dropout: float = 0.1,
                 embed_dropout: float = 0.1,
                 feed_forward_dropout: float = 0.1) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.embed_dropout = embed_dropout
        self.feed_forward_dropout = feed_forward_dropout

    def __str__(self) -> str:
        """
        Return a human-readable representation of the GPTConfig object.

        Returns:
            str: The string representation of the GPTConfig object.
        """
        attributes = [f"{attr}: {getattr(self, attr)}" for attr in dir(self) if not attr.startswith("__")]
        return "\n".join(attributes)

    def __repr__(self) -> str:
        return (f"GPTConfig("
                f"{', '.join(f'{name}={value}' for name, value in self.__dict__.items())}"
                f")")

    def to_dict(self) -> dict:
        """
        Convert the GPTConfig object to a dictionary.

        Returns:
            dict: The dictionary representation of the GPTConfig object.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GPTConfig':
        """
        Create a GPTConfig object from a dictionary.

        Args:
            config_dict (dict): The dictionary containing the configuration parameters.

        Returns:
            GPTConfig: The GPTConfig object.
        """
        return cls(**config_dict)
