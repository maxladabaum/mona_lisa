"""Exact GF(2) encoding and honest storage accounting for binary file collections.

Run --demo, --input binary.csv, or --folder PATH (equal-sized files).
The decoder is an affine XOR function, not a file lookup table. This is a
mathematical encoding check, not a simulation of digital synthesis chemistry.
"""
import argparse
import json
from pathlib import Path
import struct
import numpy as np

HEADER = struct.Struct('>4sIIII')  # magic, files, bits/file, affine rank, code width


def binary_matrix(values):
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError('Expected a nonempty matrix containing only 0 and 1; one file per row.')
    return values.astype(np.uint8)


def factor(values):
    """Return RREF basis B and codes C such that X = offset XOR (C @ B mod 2).

    The full reversible transform also retains the nonpivot residual bits.
    Those residuals are zero for this collection and therefore need no storage.
    """
    x = binary_matrix(values)
    offset = x[0].copy()
    rows = x ^ offset
    basis = rows.copy()
    rank = 0
    pivots = []
    for col in range(x.shape[1]):
        candidates = np.flatnonzero(basis[rank:, col])
        if not len(candidates):
            continue
        index = rank + candidates[0]
        basis[[rank, index]] = basis[[index, rank]]
        others = np.flatnonzero(basis[:, col])
        others = others[others != rank]
        basis[others] ^= basis[rank]
        pivots.append(col)
        rank += 1
        if rank == len(basis):
            break
    basis = basis[:rank].copy()
    codes = rows[:, pivots]
    return dict(offset=offset, basis=basis, codes=codes, pivots=np.asarray(pivots, dtype=int))


def decode(codes, offset, basis):
    return (np.asarray(codes, dtype=np.uint8) @ basis % 2) ^ offset


def transform(values, model):
    """Invertible n-bit transform, even on inputs outside the fitted affine space."""
    x = binary_matrix(values)
    delta = x ^ model['offset']
    pivots = model['pivots']
    rest = np.setdiff1d(np.arange(x.shape[1]), pivots)
    codes = delta[:, pivots]
    residual = delta ^ (codes @ model['basis'] % 2)
    return np.column_stack((codes, residual[:, rest]))


def inverse_transform(encoded, model):
    z = binary_matrix(encoded)
    rank = len(model['basis'])
    rest = np.setdiff1d(np.arange(z.shape[1]), model['pivots'])
    x = decode(z[:, :rank], model['offset'], model['basis'])
    x[:, rest] ^= z[:, rank:]
    return x


def pack(model, address_bits):
    """Store the decoder and ordered selection manifest in a real compact format.

    Manifest entries use all address bits when feasible; padding bits are zero.
    Infeasible collections are still encoded with rank bits for comparison.
    """
    rows, rank = model['codes'].shape
    width = max(rank, address_bits)
    codes = np.zeros((rows, width), dtype=np.uint8)
    codes[:, :rank] = model['codes']
    return HEADER.pack(b'GFA1', rows, len(model['offset']), rank, width) + b''.join(
        np.packbits(a.reshape(-1)).tobytes() for a in (model['offset'], model['basis'], codes))


def unpack(blob):
    magic, rows, sites, rank, width = HEADER.unpack_from(blob)
    if magic != b'GFA1' or rank > width:
        raise ValueError('Invalid affine archive.')
    cursor = HEADER.size
    arrays = []
    for shape in ((sites,), (rank, sites), (rows, width)):
        size = int(np.prod(shape)); count = (size+7)//8
        if cursor+count > len(blob):
            raise ValueError('Truncated affine archive.')
        arrays.append(np.unpackbits(np.frombuffer(blob[cursor:cursor+count], dtype=np.uint8))[:size].reshape(shape))
        cursor += count
    if cursor != len(blob):
        raise ValueError('Unexpected trailing bytes.')
    offset, basis, codes = arrays
    return decode(codes[:, :rank], offset, basis)


def analyze(values, address_bits=8):
    if not 1 <= address_bits <= 20:
        raise ValueError('Use 1–20 address bits.')
    x = binary_matrix(values)
    model = factor(x)
    rank = len(model['basis']); count, sites = x.shape
    blob = pack(model, address_bits)
    exact = np.array_equal(unpack(blob), x)
    unique = len(np.unique(x, axis=0))
    original_bytes = (x.size+7)//8
    report = dict(files=count, bits_per_file=sites, unique_files=unique,
        affine_dimension=rank, address_bits=address_bits,
        fits_independent_binary_rounds=rank <= address_bits,
        exact_archive_round_trip=bool(exact),
        original_packed_bytes=original_bytes,
        decoder_logical_bits=(rank+1)*sites,
        ordered_selection_manifest_bits=count*max(rank, address_bits),
        header_bytes=HEADER.size, total_encoded_bytes=len(blob),
        storage_saving_fraction=1-len(blob)/original_bytes,
        physical_addresses=2**address_bits,
        generated_unique_payloads=2**rank if rank <= address_bits else None,
        unused_unique_payloads=2**rank-unique if rank <= address_bits else None,
        physical_payload_is_transformed_code=True,
        note='Counts decoder, ordered selection manifest, header and byte padding. '
             'Physical molecules and experimental redundancy are not byte costs. '
             'File names and per-file lengths are not stored. Digital chemistry is not simulated.')
    return report, blob, model


def demos(seed=17):
    rng = np.random.default_rng(seed)
    for rank in (6, 8):
        basis = np.column_stack((np.eye(rank, dtype=np.uint8), rng.integers(0, 2, (rank, 128-rank), dtype=np.uint8)))
        ids = np.arange(64) if rank == 6 else rng.choice(256, 64, replace=False)
        codes = ((ids[:, None] >> np.arange(rank)) & 1).astype(np.uint8)
        yield f'structured_rank_{rank}', decode(codes, rng.integers(0, 2, 128, dtype=np.uint8), basis)
    yield 'arbitrary_random', rng.integers(0, 2, (64, 128), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true')
    source.add_argument('--input', type=Path, help='Binary CSV or .npy matrix, one file per row')
    source.add_argument('--folder', type=Path, help='Regular nonhidden files, sorted by name, equal byte lengths')
    parser.add_argument('--address-bits', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.demo:
        datasets = list(demos())
    elif args.input:
        matrix = np.load(args.input, allow_pickle=False) if args.input.suffix == '.npy' else np.loadtxt(args.input, delimiter=',', ndmin=2)
        datasets = [('input', binary_matrix(matrix))]
    else:
        paths = sorted(p for p in args.folder.iterdir() if p.is_file() and not p.name.startswith('.'))
        data = [p.read_bytes() for p in paths]
        if not data or not data[0] or len({len(v) for v in data}) != 1:
            parser.error('Folder must contain nonempty files of equal byte length. No implicit padding or truncation is applied.')
        datasets = [('input', np.stack([np.unpackbits(np.frombuffer(v, dtype=np.uint8)) for v in data]))]
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, values in datasets:
        report, blob, _ = analyze(values, args.address_bits)
        (args.output/f'{name}.gfa').write_bytes(blob)
        np.savetxt(args.output/f'{name}_bits.csv', values, fmt='%d', delimiter=',')
        reports[name] = report
    (args.output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
