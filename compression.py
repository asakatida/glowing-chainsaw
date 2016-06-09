"""
toy compression algorithm with an interface matching stdlib modules.

class StatesError

class StatesCompressor
class StatesDecompressor
class StatesFile

def compress
def decompress
def open
"""

import io
import itertools
import os
import pathlib
import struct
import typing

from pprint import pprint

__all__ = [
    'StatesError',
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

D = typing.TypeVar('D')
M = typing.TypeVar('M')


class StatesError(Exception):
    """StatesError."""


class Reversible(typing.Generic[D, M]):
    """Reversible."""

    @classmethod
    def from_data(cls, data: D) -> 'Reversible':
        """from_data."""
        return cls(data=data)

    @classmethod
    def from_meta(cls, meta: M) -> 'Reversible':
        """from_meta."""
        return cls(meta=meta)

    def __init__(self, *, data: D=None, meta: M=None) -> None:
        self._data = data
        self._meta = meta

    @property
    def data(self) -> D:
        """restore."""
        if self._data is not None:
            return self._data
        self._data = self._restore()
        return self._data

    @property
    def meta(self) -> M:
        """analyse."""
        if self._meta is not None:
            return self._meta
        self._meta = self._analyse()
        return self._meta


class Followers(Reversible[bytes, MetaData]):
    """Followers."""

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

    def __analyse(self, count: int) -> OptMetaData:
        """map subsequences to next byte if unique."""
        flat_node = {bytes(reversed(self._data[-count:])): -1}
        for end in range(len(self._data)):
            start = max(end - count, 0)
            key = bytes(reversed(self._data[start:end])).ljust(count, b'\x00')
            value = self._data[end]
            if flat_node.setdefault(key, value) != value:
                return None
        return flat_node

    def _analyse(self) -> MetaData:
        """find shortest size that uniquely maps to next byte."""
        for count in range(len(self._data) + 1):
            flat_node = self.__analyse(count)
            if flat_node:
                return flat_node
        raise StatesError


class KeyTrunc(Reversible[MetaData, MetaData]):
    """KeyTrunc."""

    def _restore(self) -> MetaData:
        """restore."""
        inflate = {}  # type: Dict[bytes, int]
        unique_size = max(len(key) for key in self._meta.keys())
        next_key = b'\x00' * unique_size
        while True:
            for count in range(unique_size + 1, 0, -1):
                value = self._meta.get(next_key[:count], None)
                if value is not None:
                    inflate[next_key] = value
                    if value == -1:
                        return inflate
                    next_key = (bytes([value]) + next_key)[:unique_size]
                    break
                elif count == 1:
                    raise StatesError

    def _analyse(self) -> MetaData:
        """get shortest sequence to match each next."""
        flat_node = dict(self._data.items())
        condense = {}  # type: Dict[bytes, int]
        unique_size = len(next(iter(flat_node.keys()))) + 1
        repeats = iter(range(1, unique_size))
        while flat_node:
            count = next(repeats)
            for start in set(key[:count] for key in flat_node.keys()):
                possible = set(
                    flat_node.get(key)
                    for key in flat_node.keys() if key.startswith(start))
                if len(possible) == 1:
                    condense[start] = possible.pop()
                    for key in tuple(flat_node.keys()):
                        if key.startswith(start):
                            del flat_node[key]
        return condense


class Reshape(Reversible[MetaData, MetaTree]):
    """Reshape."""

    def _restore(self) -> MetaData:
        """restore."""
        flat_node = {}  # type: Dict[bytes, int]
        tree_root = dict(self._meta.items())
        while tree_root:
            for key, value in tuple(tree_root.items()):
                if isinstance(key, int):
                    bytes_key = bytes((key,))
                else:
                    bytes_key = key
                del tree_root[key]
                if isinstance(value, int):
                    flat_node[bytes_key] = value
                    continue
                for child_key, child_value in value.items():
                    if isinstance(child_key, int):
                        child_key = bytes((child_key,))
                    tree_root[bytes_key + child_key] = child_value
        return flat_node

    def _analyse(self) -> MetaTree:
        """convert meta mapping to tree format."""
        tree_root = {}
        for key, value in self._data.items():
            ref = tree_root
            for lookback in key[:-1]:
                ref = ref.setdefault(lookback, {})
            ref[key[-1]] = value
        return tree_root


class Serialize(Reversible[MetaTree, FlatFmt]):
    """Serialize."""

    def _restore(self) -> MetaTree:
        """restore."""
        flat_node = {}  # type: Dict[bytes, int]
        tree_root = dict(self._meta.items())
        while tree_root:
            for key, value in tuple(tree_root.items()):
                if isinstance(key, int):
                    bytes_key = bytes((key,))
                else:
                    bytes_key = key
                del tree_root[key]
                if isinstance(value, int):
                    flat_node[bytes_key] = value
                    continue
                for child_key, child_value in value.items():
                    if isinstance(child_key, int):
                        child_key = bytes((child_key,))
                    tree_root[bytes_key + child_key] = child_value
        return flat_node

    def freeze_tree(tree_root: MetaTree, found) -> MetaTree:
        """freeze_tree."""
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
        """extract_max."""
        return max(key for key, _ in tree_root)


    def extract_min(tree_root: MetaTree) -> int:
        """extract_min."""
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


    def _analyse(self) -> FlatFmt:
        """convert mapping to wire format."""
        frozen = self.freeze_tree(self._data, set())
        top = self.serialize_branch(frozen, dict())
        return [self.extract_max(frozen)] + top


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
    return tree_root
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
    return tree_root
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


class States(Reversible):
    """States."""

    def _restore(self) -> bytes:
        """restore."""
        return deserialize_wire(self._meta)

    def _analyse(self) -> bytes:
        """analyse."""
        node = Followers.from_data(self._data)
        node = KeyTrunc.from_data(node.meta)
        node = Reshape.from_data(node.meta)
        tree_root = merge_tree(flatten_tree(node.meta))
        node = Serialize.from_data(tree_root)
        return pack_format(node.meta)


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
        return States.from_data(self._data.getvalue()).meta


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
        orig = Reshape.from_data(
            KeyTrunc.from_data(
                Followers.from_data(
                    istream.read()).meta).meta).meta
        serialize = Serialize.from_data(orig)
        new = Serialize.from_meta(serialize.meta).data
        for key, value in orig.items():
            if key not in new or new[key] != value:
                pprint((key, value))
