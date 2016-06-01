"""
toy compression algorithm with an interface matching stdlib bz2 module.

class StatesCompressor
class StatesDecompressor
class StatesFile

def compress
def decompress
def open
"""

import collections
import io
import os
import struct
import typing

from pprint import pprint

__all__ = [
    'StatesCompressor',
    'StatesDecompressor',
    'StatesFile',
    'compress',
    'decompress',
    'open']

SIGNATURE = b'YWRhbQ'

MetaData = typing.Dict[bytes, int]   # noqa
MetaKeys = typing.MutableSet[bytes]  # noqa
MetaValues = typing.MutableSet[int]  # noqa
OptMetaData = typing.Optional[MetaData]

MetaTree = typing.Dict[int, typing.Union['MetaTree', int]]  # noqa

FlatFmt = typing.List[typing.Optional[int]]  # noqa

ByteLines = typing.List[bytes]  # noqa
ByteLinesIter = typing.Iterable[bytes]  # noqa
Files = typing.Union['StatesFile', typing.TextIO]


def build_followers_size(data: bytes, count: int) -> OptMetaData:
    """map subsequences to next byte if unique."""
    flat_node = {bytes(reversed(data[-count:])): -1}
    for position in range(len(data) - count - 1):
        end = position + count
        key = bytes(reversed(data[position:end])).ljust(count, b'\xFF')
        value = data[end + 1]
        if flat_node.setdefault(key, value) != value:
            return None
    return flat_node


def build_unique_followers(data: bytes) -> MetaData:
    """find shortest size that uniquely maps to next byte."""
    for size in range(len(data)):
        flat_node = build_followers_size(data, size)
        if flat_node:
            return flat_node


def truncate_keys(count: int, flat_node: MetaData) -> MetaKeys:
    """get all keys truncated to length."""
    return set(key[:count] for key in flat_node.keys())


def filter_keys_values(start: bytes, flat_node: MetaData) -> MetaValues:
    """filter values by keys that start with sequence."""
    return set(
        flat_node.get(key)
        for key in flat_node.keys() if key.startswith(start))


def discard_keys(start: bytes, flat_node: MetaData) -> None:
    """remove keys that start with sequence."""
    for key in set(key for key in flat_node.keys() if key.startswith(start)):
        del flat_node[key]


def condense_unique_map(flat_node: MetaData) -> MetaData:
    """get shortest sequence to match each next."""
    flat_node = dict(flat_node.items())
    condense = {}  # type: Dict[bytes, int]
    unique_size = len(next(iter(flat_node.keys()))) + 1
    repeats = iter(range(1, unique_size))
    while flat_node:
        for start in truncate_keys(next(repeats), flat_node):
            possible = filter_keys_values(start, flat_node)
            if len(possible) == 1:
                condense[start] = possible.pop()
                discard_keys(start, flat_node)
    return condense


def strip_tree(tree_root: MetaTree) -> MetaTree:
    """replace value tags from tree with terminal."""
    striped = {}
    for key, value in tree_root.items():
        if value == {}:
            return key
        striped[key] = strip_tree(value)
    return striped


def tree() -> typing.Mapping:
    """autovivification tree."""
    return collections.defaultdict(tree)


def meta_to_tree(flat_node: MetaData) -> MetaTree:
    """convert meta mapping to tree format."""
    tree_root = tree()
    for key, value in flat_node.items():
        ref = tree_root
        for lookback in key:
            ref = ref[lookback]
        ref = ref[value]
        ref.default_factory = None
    return strip_tree(tree_root)


def serialize_branch(tree_root: MetaTree, base_idx: int) -> (FlatFmt, int):
    """convert branch to wire format."""
    flat = [None] * (max(tree_root.keys()) + 1)
    sub_flat = []
    next_idx = base_idx
    for key, value in tree_root.items():
        if isinstance(value, int):
            flat[key] = value
            continue
        seri, key_idx = serialize_branch(value, next_idx)
        flat[key] = key_idx
        next_idx += len(seri)
        sub_flat += seri
    return sub_flat + flat, next_idx


def pack_format(flat: FlatFmt) -> bytes:
    """pack collection of indexes into a tagged byte format."""
    size = len(flat) + 2
    bytes_size = (size.bit_length() // 8) + 1
    if bytes_size > 4:
        format_str = '>Q'
    elif bytes_size > 2:
        format_str = '>L'
    elif bytes_size > 1:
        format_str = '>H'
    else:
        format_str = '>B'
    packer = struct.Struct(format_str)
    fail_idx = packer.pack(size)
    eof_idx = packer.pack(0)

    def real_idx(idx: typing.Optional[int]) -> bytes:
        """handle special fake indexes."""
        if idx is None:
            return fail_idx
        elif idx < 0:
            return eof_idx
        return packer.pack(idx)

    if format_str == '>B':
        format_str.ljust(3, '\x00')
    format_str = format_str.encode()

    return SIGNATURE + format_str + b''.join(real_idx(idx) for idx in flat)


def serialize_meta(start: int, tree_root: MetaTree) -> bytes:
    """convert mapping to wire format."""
    flat = [start] + [None] * 256
    sub_flat = []
    next_idx = len(flat)
    for key, value in tree_root.items():
        if isinstance(value, int):
            flat[key] = value
            continue
        seri, key_idx = serialize_branch(value, next_idx)
        flat[key] = key_idx
        next_idx += len(seri)
        sub_flat += seri
    return pack_format(flat + sub_flat)


class StatesCompressor:
    """compression class for incremental packing.

    compression class for incremental packing
    """

    def __init__(self) -> None:
        """compression class for incremental packing."""
        self._data = io.BytesIO()

    def compress(self, data: bytes) -> bytes:
        """record partial data."""
        self._data.write(data)
        return b''

    def flush(self) -> bytes:
        """end input stream and return compressed form."""
        data = self._data.getvalue()
        flat_node = build_unique_followers(data)
        condense = condense_unique_map(flat_node)
        tree_root = meta_to_tree(condense)
        return serialize_meta(data[0], tree_root)


def deserialize_wire(data: bytes) -> bytes:
    """get collection of indexes from wire format."""
    signature = data[:len(SIGNATURE)]
    if signature != SIGNATURE:
        return b''
    tag_idx = data.index(0)
    unpacker = struct.Struct(data[len(SIGNATURE):tag_idx])
    if unpacker.size == 1:
        tag_idx += 1
    flat = list(unpacker.iter_unpack(data[tag_idx:]))
    output = io.BytesIO(bytes(flat[0]))
    return output.getvalue()


class StatesDecompressor:
    """decompression class for incremental unpacking.

    decompression class for incremental unpacking
    """

    def __init__(self) -> None:
        """decompression class for incremental unpacking."""
        self._data = io.BytesIO()
        self._eof = False
        self._needs_input = True
        self._unused_data = io.BytesIO()

    def decompress(self, data: bytes, max_length: int=-1) -> bytes:
        """get partial reconstruction of stream."""
        self._data.write(data)
        if max_length < 0:
            return deserialize_wire(self._data.getvalue())
        elif max_length == 0:
            return b''
        data = self._data.getvalue()
        return self._data.getvalue()

    @property
    def eof(self) -> bool:
        """get is file end reached."""
        return self._eof

    @property
    def needs_input(self) -> bool:
        """property."""
        return self._needs_input

    @property
    def unused_data(self) -> bytes:
        """property."""
        return self._unused_data.getvalue()


class StatesFile(io.BufferedIOBase, typing.BinaryIO):
    """wrapper for transparent file compression.

    wrapper for transparent file compression
    """

    def __init__(self, filename, mode='r') -> None:
        """wrapper for transparent file compression."""
        super(StatesFile, self).__init__()
        if 'r' in mode:
            self._file = io.FileIO(filename, 'rb')
        else:
            self._file = io.FileIO(filename, 'wb')

    def __enter__(self) -> 'StatesFile':
        """context manager."""
        return self

    def __exit__(self, *args) -> bool:
        """context manager."""
        return self._file.__exit__(*args)

    def __iter__(self):
        """iterable."""
        return self

    def __next__(self) -> bytes:
        """get line."""
        return self.readline()

    def close(self) -> None:
        """finalize."""
        self._file.close()

    def detach(self) -> int:
        """unsupported."""
        raise io.UnsupportedOperation

    def fileno(self) -> int:
        """fileno."""
        return self._file.fileno()

    def flush(self) -> None:
        """flush."""
        return self._file.flush()

    def isatty(self) -> bool:
        """isatty."""
        return self._file.isatty()

    def readable(self) -> bool:
        """readable."""
        return self._file.readable()

    def read(self, size: int=-1) -> bytes:
        """read."""
        return self._file.read(size)

    def read1(self, size: int=-1) -> bytes:
        """unsupported."""
        raise io.UnsupportedOperation

    def readinto(self, b: bytes) -> bool:
        """readinto."""
        return self._file.readinto(b)

    def readinto1(self, b: bytes) -> bool:
        """unsupported."""
        raise io.UnsupportedOperation

    def readline(self, size: int=-1) -> bytes:
        """readline."""
        return self._file.readline(size)

    def readlines(self, hint: int=-1) -> ByteLines:
        """readlines."""
        return self._file.readlines(hint)

    def seek(self, offset: int, whence: int=os.SEEK_SET) -> bool:
        """seek."""
        return self._file.seek(offset, whence)

    def seekable(self) -> bool:
        """seekable."""
        return self._file.seekable()

    def tell(self) -> bool:
        """tell."""
        return self._file.tell()

    def truncate(self, size: int=None) -> int:
        """truncate."""
        return self._file.truncate(size)

    def writable(self) -> bool:
        """writable."""
        return self._file.writable()

    def write(self, b: bytes) -> int:
        """write."""
        return self._file.write(b)

    def writelines(self, lines: ByteLinesIter) -> None:
        """writelines."""
        return self._file.writelines(lines)

    def peek(self, n: int=None) -> bytes:
        """unsupported."""
        raise io.UnsupportedOperation


def compress(data: bytes) -> bytes:
    """compress."""
    states_compressor = StatesCompressor()
    states_compressor.compress(data)
    return states_compressor.flush()


def decompress(data: bytes) -> bytes:
    """decompress."""
    return StatesDecompressor().decompress(data)


def open(filename: str, **kwargs) -> Files:   # noqa
    """open."""
    try:
        mode = kwargs.pop('mode')
    except KeyError:
        return StatesFile(filename)
    text_mode = 't' in mode
    mode = mode.replace('t', '')
    states_file = StatesFile(filename, mode=mode)
    if text_mode:
        return io.TextIOWrapper(states_file, **kwargs)
    return states_file


if __name__ == '__main__':
    with io.FileIO(__file__) as istream:
        orig = istream.read()
        new = compress(orig)
        trip = decompress(new)
        pprint(trip)
        pprint(len(new) / len(orig))
        pprint(orig == trip)
