import builtins
import io
import itertools
import os
import typing

from pprint import pprint


__all__ = [
    'StatesCompressor',
    'StatesDecompressor',
    'StatesFile',
    'compress',
    'decompress',
    'open']


def get_next_or_eof(data: bytes, pos: int) -> int:
    try:
        return data[pos]
    except IndexError:
        return -1


def build_followers_size(data: bytes, count: int) -> typing.Optional[typing.Dict[bytes, int]]:
    flat_node = {}  # type: Dict[bytes, int]
    if any(map(
        lambda result, position: flat_node.setdefault(
            bytes(reversed(data[position:position + count])), result) != result,
        map(
            lambda position: get_next_or_eof(data, position + count + 1),
            range(len(data))),
        range(len(data)))):

        return None
    return flat_node


def build_unique_followers(data: bytes) -> typing.Dict[bytes, int]:
    return next(filter(None, map(
        build_followers_size, itertools.repeat(data), range(1, len(data)))))


def filter_keys_values(start: bytes, flat_node: typing.Mapping[bytes, int]) -> typing.MutableSet[int]:
    return set(map(
        flat_node.get,
        filter(lambda key: key.startswith(start), flat_node.keys())))


def condense_unique_map(flat_node: typing.Mapping[bytes, int]) -> typing.Dict[bytes, int]:
    condense = {}  # type: Dict[bytes, int]
    data_set = tuple(map(
        lambda start: bytes((start,)),
        set(itertools.chain.from_iterable(flat_node.keys()))))
    unique_size = len(next(iter(flat_node.keys()))) - 1
    for start in data_set:
        possible = filter_keys_values(start, flat_node)
        if len(possible) == 0:
            continue
        elif len(possible) == 1:
            condense[start] = possible.pop()
            continue
        repeats = iter(range(1, unique_size))
        while possible:
            for long_start in map(
                lambda prod: start + b''.join(prod),
                itertools.product(
                    data_set, repeat=next(repeats))):
                long_possible = filter_keys_values(long_start, flat_node)
                if len(long_possible) == 1:
                    condense[long_start] = long_possible.pop()
                    possible.discard(condense[long_start])
    return condense
    


class StatesCompressor:
    def __init__(self) -> None:
        self._data = b''

    def compress(self, data: bytes) -> bytes:
        self._data += data
        return b''

    def flush(self) -> bytes:
        if len(self._data) < 2:
            return self._data
        flat_node = build_unique_followers(self._data)
        condense = condense_unique_map(flat_node)
        return self._data[0]


class StatesDecompressor:
    def __init__(self) -> None:
        self._data = b''
        self._eof = False
        self._needs_input = True
        self._unused_data = b''

    def decompress(self, data: bytes, *, max_length: int = -1) -> bytes:
        self._data += data
        return b''

    @builtins.property
    def eof(self) -> bool:
        return self._eof

    @builtins.property
    def needs_input(self) -> bool:
        return self._needs_input

    @builtins.property
    def unused_data(self) -> bytes:
        return self._unused_data


class StatesFile(typing.BinaryIO, typing.Iterable, io.BufferedIOBase):
    def __init__(self, filename, *, mode='r') -> None:
        if 'r' in mode:
            self._file = io.FileIO(filename, 'rb')
        else:
            self._file = io.FileIO(filename, 'wb')

    def __enter__(self) -> 'StatesFile':
        return self

    def __exit__(self, *args) -> bool:
        return self._file.__exit__(*args)

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        return self.readline()

    def close(self) -> None:
        self._file.close()

    def detach(self) -> int:
        raise io.UnsupportedOperation

    def fileno(self) -> int:
        return self._file.fileno()

    def flush(self) -> None:
        return self._file.flush()

    def isatty(self) -> bool:
        return self._file.isatty()

    def readable(self) -> bool:
        return self._file.readable()

    def read(self, size: int = -1) -> bytes:
        return self._file.read(size)

    def read1(self, size: int = -1) -> bytes:
        raise io.UnsupportedOperation

    def readinto(self, b: bytes) -> bool:
        return self._file.readinto(b)

    def readinto1(self, b: bytes) -> bool:
        raise io.UnsupportedOperation

    def readline(self, size: int = -1) -> bytes:
        return self._file.readline(size)

    def readlines(self, hint: int = -1) -> typing.List[bytes]:
        return self._file.readlines(hint)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> bool:
        return self._file.seek(offset, whence)

    def seekable(self) -> bool:
        return self._file.seekable()

    def tell(self) -> bool:
        return self._file.tell()

    def truncate(self, size: int = None) -> int:
        return self._file.truncate(size)

    def writable(self) -> bool:
        return self._file.writable()

    def write(self, b: bytes) -> int:
        return self._file.write(b)

    def writelines(self, lines: typing.Iterable[bytes]) -> None:
        return self._file.writelines(lines)

    def peek(self, n: int = None) -> bytes:
        raise io.UnsupportedOperation


def compress(data: bytes) -> bytes:
    states_compressor = StatesCompressor()
    states_compressor.compress(data)
    return states_compressor.flush()


def decompress(data: bytes) -> bytes:
    return StatesDecompressor().decompress(data)


def open(filename: str, **kwargs: typing.Any) -> typing.Union[StatesFile, typing.TextIO]:
    try:
        mode = typing.cast(str, kwargs.pop('mode'))
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
        pprint(compress(istream.read()))
