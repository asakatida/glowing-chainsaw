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
import itertools
import os
import pathlib
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

MetaData = typing.Dict[bytes, int]  # noqa
MetaKeys = typing.MutableSet[bytes]  # noqa
MetaValues = typing.MutableSet[int]  # noqa
OptMetaData = typing.Optional[MetaData]

MetaTree = typing.Dict[int, typing.Union['MetaTree', int]]  # noqa

FlatFmt = typing.List[typing.Optional[int]]  # noqa

ByteLines = typing.List[bytes]  # noqa
ByteLinesIter = typing.Iterable[bytes]  # noqa
Files = typing.Union['StatesFile', typing.TextIO]


class Followers:
    @classmethod
    def from_data(cls, data: bytes) -> 'Followers':
        return cls(data=data)

    @classmethod
    def from_meta(cls, meta: MetaData) -> 'Followers':
        return cls(meta=meta)

    def __init__(self, *, data: bytes=None, meta: MetaData=None) -> None:
        self._data = data
        self._meta = meta

    def _restore(self) -> bytes:
        """restore."""
        count = len(next(iter(self._meta.keys())))
        data = b'\x00' * count
        while True:
            key = bytes(reversed(data[-count:]))
            value = self._meta[key]
            if value == -1:
                break
            data += bytes((value,))
        return data[count:]

    def _analyse(self, count: int) -> OptMetaData:
        """map subsequences to next byte if unique."""
        flat_node = {bytes(reversed(self._data[-count:])): -1}
        for end in range(len(self._data)):
            start = max(end - count, 0)
            key = bytes(reversed(self._data[start:end])).ljust(count, b'\x00')
            value = self._data[end]
            if flat_node.setdefault(key, value) != value:
                return None
        return flat_node

    @property
    def data(self) -> bytes:
        """restore."""
        if self._data is not None:
            return self._data
        self._data = self._restore()
        return self._data

    @property
    def meta(self) -> MetaData:
        """find shortest size that uniquely maps to next byte."""
        if self._meta is not None:
            return self._meta
        for count in range(len(self._data) + 1):
            flat_node = self._analyse(count)
            if flat_node:
                self._meta = flat_node
                return self._meta


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


def meta_to_tree(flat_node: MetaData) -> MetaTree:
    """convert meta mapping to tree format."""
    tree_root = {}
    for key, value in flat_node.items():
        ref = tree_root
        for lookback in key[:-1]:
            ref = ref.setdefault(lookback, {})
        ref[key[-1]] = value
    return tree_root


def pack_format(flat: FlatFmt) -> bytes:
    """pack collection of indexes into a tagged byte format."""
    max_idx = max(flat, key=lambda idx: idx or 0)
    if max_idx > 0xFFFF:
        format_str = '>L'
    elif max_idx > 0xFF:
        format_str = '>H'
    else:
        format_str = '>B'
    packer = struct.Struct(format_str)

    fail_idx = packer.unpack(b'\xFF' * packer.size)[0]
    eof_idx = len(flat)

    def real_idx(idx: typing.Optional[int]) -> bytes:
        """handle special fake indexes."""
        if idx is None:
            return fail_idx
        elif idx < 0:
            return eof_idx
        return idx

    if packer.size == 1:
        format_str = format_str.ljust(len(format_str) + 1, '\x00')
    format_str = SIGNATURE + format_str.encode()

    return format_str + b''.join(packer.pack(real_idx(idx)) for idx in flat)


def freeze_tree(tree_root: MetaTree, found) -> MetaTree:
    root = {}
    for key, value in tree_root.items():
        if isinstance(value, int):
            root[key] = value
            continue
        frozen = freeze_tree(value, found)
        if frozen in found:
            root[key] = next(sub for sub in found if sub == frozen)
            continue
        root[key] = frozen
        found.add(frozen)
    return frozenset(root.items())


def extract_max(tree_root: MetaTree) -> int:
    return max(key for key, _ in tree_root)


def extract_min(tree_root: MetaTree) -> int:
    return min(key for key, _ in tree_root)


def serialize_branch(
        tree_root: MetaTree, found, base_idx: int=0) -> FlatFmt:
    """convert branch to wire format."""
    small = extract_min(tree_root) - 1
    flat = [None] * (extract_max(tree_root) - small + 1)
    flat[0] = small
    next_idx = base_idx + len(flat)
    for key, value in tree_root:
        key -= small
        if isinstance(value, int):
            flat[key] = value
            continue
        if value in found:
            flat[key] = found[value]
            continue
        seri = serialize_branch(value, found, next_idx)
        flat[key] = next_idx
        found[value] = next_idx
        next_idx += len(seri)
        flat += seri
    return flat


def serialize_meta(tree_root: MetaTree) -> bytes:
    """convert mapping to wire format."""
    frozen = freeze_tree(tree_root, set())
    top = serialize_branch(frozen, dict())
    return pack_format([extract_max(frozen)] + top)


def make_queue(tree_root: MetaTree) -> MetaKeys:
    """make_queue."""
    return set(
        key
        for key in tree_root.keys() if not isinstance(tree_root[key], int))


def mergable_tree(left: MetaTree, right: MetaTree) -> bool:
    """mergable_tree."""
    for key in set(left.keys()).intersection(right.keys()):
        if left[key] != right[key]:
            return False
    return True


def flatten_tree(tree_root: MetaTree) -> MetaTree:
    """flatten_tree."""
    root = {}
    for key, value in tree_root.items():
        if isinstance(value, int):
            root[key] = value
        else:
            root[key] = flatten_tree(value)
    queue = make_queue(root)
    while queue:
        key = queue.pop()
        value = root[key]
        if mergable_tree(root, value):
            root.pop(key)
            root.update(value.items())
            queue.update(make_queue(value))
    return root


def merge_tree(tree_root: MetaTree) -> MetaTree:
    """merge_tree."""
    root = {}
    for key, value in tree_root.items():
        if isinstance(value, int):
            root[key] = value
        else:
            root[key] = merge_tree(value)
    queue = make_queue(root)
    while queue:
        key = queue.pop()
        value = root[key]
        for sibling_key in make_queue(root).difference({key}):
            sibling_value = root[sibling_key]
            if mergable_tree(sibling_value, value):
                value.update(sibling_value)
                root[sibling_key] = value
    return root


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
        followers = Followers.from_data(data)
        condense = condense_unique_map(followers.meta)
        tree_root = flatten_tree(meta_to_tree(condense))
        return serialize_meta(merge_tree(tree_root))


def deserialize_wire(data: bytes) -> bytes:
    """get collection of indexes from wire format."""
    signature = data[:len(SIGNATURE)]
    if signature != SIGNATURE:
        return b''

    tag_idx = data.index(0)
    unpacker = struct.Struct(data[len(SIGNATURE):tag_idx])
    if unpacker.size == 1:
        tag_idx += 1

    states_data = data[tag_idx:]

    indexes = unpacker.iter_unpack(states_data)

    max_key = next(indexes)[0]

    flat = list(itertools.chain.from_iterable(indexes))

    max_idx = len(flat) + 1

    fail_idx = unpacker.unpack(b'\xFF' * unpacker.size)

    output = io.BytesIO()
    index = -1
    flat_offset = 0
    while True:
        if index < -len(output.getvalue()):
            lookback = 0
        else:
            lookback = output.getvalue()[index]
        index -= 1
        flat_offset = flat[flat_offset + lookback - flat[flat_offset]]
        if flat_offset == max_idx:
            break
        elif flat_offset <= max_key:
            output.write(bytes((flat_offset,)))
            index = -1
            flat_offset = 0
        elif flat_offset == fail_idx:
            break
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
        followers = Followers.from_data(orig)
        new = compress(orig)
        trip = decompress(new)
        if orig == trip:
            pprint(len(new) / len(orig))
            for file in pathlib.Path('.').iterdir():
                if file.is_file():
                    file.with_suffix('.states').write_bytes(
                        compress(file.read_bytes()))
        else:
            pprint(trip)
