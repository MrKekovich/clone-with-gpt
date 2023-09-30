# Path related exceptions
class InvalidDatasetPathError(Exception):
    """
    Raised when the dataset path is invalid.
    """


class InvalidVocabPathError(Exception):
    """
    Raised when the vocabulary path is invalid.
    """


class NoFilesFoundError(Exception):
    """
    Raised when no files are found.
    """


class InvalidSaveDirectoryError(Exception):
    """
    Raised when the save directory is invalid.
    """


# GPT specific exceptions
class InvalidNumHeadsError(Exception):
    """
    Raised when the number of heads is invalid.
    """
