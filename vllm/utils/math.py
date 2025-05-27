import pickle
import hashlib

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()

def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    return (x // y) * y

def sha256(input) -> int:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        An integer representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(),
                          byteorder="big")