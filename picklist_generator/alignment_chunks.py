"""Local raster blocks for alignment operators, including cross-tile overlap.

A block owns a disjoint raster patch for one design. Different designs may
contribute to the same pixels; forward accumulation and its adjoint retain that
overlap. Gaussian tails are cropped only beyond eight standard deviations.
"""
from dataclasses import dataclass
import numpy as np

KERNEL_CUTOFF_SIGMA = 8.0
CHUNK_EDGE = 64
MAX_CACHED_ELEMENTS = 64_000_000


def raster_patches(xs, ys, points, sigma, edge=CHUNK_EDGE):
    points = np.asarray(points).reshape(-1, 2)
    lower = points.min(axis=0) - KERNEL_CUTOFF_SIGMA * sigma
    upper = points.max(axis=0) + KERNEL_CUTOFF_SIGMA * sigma
    x0, x1 = np.searchsorted(xs, [lower[0], upper[0]], side='left')
    y0, y1 = np.searchsorted(ys, [lower[1], upper[1]], side='left')
    for y in range(y0, y1, edge):
        for x in range(x0, x1, edge):
            yield slice(y, min(y + edge, y1)), slice(x, min(x + edge, x1))


@dataclass
class MeanBlock:
    y: slice
    x: slice
    columns: slice
    values: object

    @property
    def image_shape(self):
        return (self.y.stop - self.y.start, self.x.stop - self.x.start)


class ChunkedMean:
    def __init__(self, height, width, columns, blocks):
        self.height, self.width = height, width
        self.shape = (height * width, columns)
        self.blocks = blocks
        self._column_blocks = [[] for _ in range(columns)]
        for block in blocks:
            for i in range(block.columns.start, block.columns.stop):
                self._column_blocks[i].append(block)

    @property
    def nbytes(self):
        return sum(block.values.nbytes for block in self.blocks)

    @property
    def T(self):
        return _Adjoint(self)

    def __matmul__(self, weights):
        weights = np.asarray(weights)
        tail = weights.shape[1:]
        result = np.zeros((self.height, self.width) + tail)
        for block in self.blocks:
            value = block.values @ weights[block.columns]
            result[block.y, block.x] += value.reshape(block.image_shape + tail)
        return result.reshape((self.shape[0],) + tail)

    def adjoint(self, pixels):
        pixels = np.asarray(pixels)
        tail = pixels.shape[1:]
        local = pixels.reshape((self.height, self.width) + tail)
        result = np.zeros((self.shape[1],) + tail)
        for block in self.blocks:
            patch = local[block.y, block.x].reshape((-1,) + tail)
            result[block.columns] += block.values.T @ patch
        return result

    def column_squared_norms(self):
        result = np.zeros(self.shape[1])
        for block in self.blocks:
            result[block.columns] += np.einsum('ij,ij->j', block.values, block.values)
        return result

    def column_dot(self, index, pixels):
        local = pixels.reshape(self.height, self.width)
        return sum(block.values[:, index - block.columns.start] @ local[block.y, block.x].reshape(-1)
                   for block in self._column_blocks[index])

    def add_column(self, pixels, index, scale):
        local = pixels.reshape(self.height, self.width)
        for block in self._column_blocks[index]:
            local[block.y, block.x] += scale * block.values[:, index - block.columns.start].reshape(block.image_shape)

    def gram(self):
        # Used only for small variable counts; never materialize the raster ×
        # variables matrix, even when constructing a small dense Hessian.
        result = np.zeros((self.shape[1], self.shape[1]))
        for i, a in enumerate(self.blocks):
            for b in self.blocks[i:]:
                y0, y1 = max(a.y.start, b.y.start), min(a.y.stop, b.y.stop)
                x0, x1 = max(a.x.start, b.x.start), min(a.x.stop, b.x.stop)
                if y0 >= y1 or x0 >= x1:
                    continue
                av = a.values.reshape(a.image_shape + (-1,))[y0-a.y.start:y1-a.y.start, x0-a.x.start:x1-a.x.start].reshape(-1, a.values.shape[1])
                bv = b.values.reshape(b.image_shape + (-1,))[y0-b.y.start:y1-b.y.start, x0-b.x.start:x1-b.x.start].reshape(-1, b.values.shape[1])
                product = av.T @ bv
                result[a.columns, b.columns] += product
                if a is not b:
                    result[b.columns, a.columns] += product.T
        return result


class _Adjoint:
    def __init__(self, mean):
        self.mean = mean

    def __matmul__(self, value):
        return self.mean.gram() if value is self.mean else self.mean.adjoint(value)
