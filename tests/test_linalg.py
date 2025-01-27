from textwrap import dedent

import pytest

import mlir.extras.types as T
from mlir.extras.dialects.ext import linalg, memref, tensor
from mlir.ir import AffineMap, OpResultList, RankedTensorType

# noinspection PyUnresolvedReferences
from mlir.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


def test_np_constructor(ctx: MLIRContext):
    x = memref.alloc((10, 10), T.i32())
    linalg.fill(5, x)
    linalg.fill_rng_2d(0.0, 10.0, 1, x)

    x = tensor.empty(10, 10, T.i32())
    y = linalg.fill_rng_2d(0.0, 10.0, 1, x)
    z = linalg.fill(5, x)

    correct = dedent(
        """\
    module {
      %alloc = memref.alloc() : memref<10x10xi32>
      %c5_i32 = arith.constant 5 : i32
      linalg.fill ins(%c5_i32 : i32) outs(%alloc : memref<10x10xi32>)
      %cst = arith.constant 0.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+01 : f64
      %c1_i32 = arith.constant 1 : i32
      linalg.fill_rng_2d ins(%cst, %cst_0, %c1_i32 : f64, f64, i32) outs(%alloc : memref<10x10xi32>)
      %0 = tensor.empty() : tensor<10x10xi32>
      %cst_1 = arith.constant 0.000000e+00 : f64
      %cst_2 = arith.constant 1.000000e+01 : f64
      %c1_i32_3 = arith.constant 1 : i32
      %1 = linalg.fill_rng_2d ins(%cst_1, %cst_2, %c1_i32_3 : f64, f64, i32) outs(%0 : tensor<10x10xi32>) -> tensor<10x10xi32>
      %c5_i32_4 = arith.constant 5 : i32
      %2 = linalg.fill ins(%c5_i32_4 : i32) outs(%0 : tensor<10x10xi32>) -> tensor<10x10xi32>
    }
    """
    )
    filecheck(correct, ctx.module)


def test_generic(ctx: MLIRContext):
    id_map = AffineMap.get_identity(2)

    x = tensor.empty(16, 16, T.f32())
    y = tensor.empty(16, 16, T.f32())

    @linalg.generic(
        [x],
        [y],
        [id_map, id_map],
        [linalg.IteratorType.parallel, linalg.IteratorType.parallel],
    )
    def f(x, y):
        return x + y

    print(f)

    z = tensor.empty(16, 16, 16, T.f32())

    minor_id = AffineMap.get_minor_identity(3, 2)
    id_map = AffineMap.get_identity(3)

    @linalg.generic(
        [x],
        [z, z],
        [minor_id, id_map, id_map],
        [
            linalg.IteratorType.parallel,
            linalg.IteratorType.parallel,
            linalg.IteratorType.parallel,
        ],
    )
    def g(x, z1, z2):
        return x, z1

    assert isinstance(g, OpResultList)
    assert len(g) == 2
    assert isinstance(g[0].type, RankedTensorType)
    assert isinstance(g[1].type, RankedTensorType)

    correct = dedent(
        """\
    #map = affine_map<(d0, d1) -> (d0, d1)>
    #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
    #map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    module {
      %0 = tensor.empty() : tensor<16x16xf32>
      %1 = tensor.empty() : tensor<16x16xf32>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<16x16xf32>) outs(%1 : tensor<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %in, %out : f32
        linalg.yield %5 : f32
      } -> tensor<16x16xf32>
      %3 = tensor.empty() : tensor<16x16x16xf32>
      %4:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<16x16xf32>) outs(%3, %3 : tensor<16x16x16xf32>, tensor<16x16x16xf32>) {
      ^bb0(%in: f32, %out: f32, %out_0: f32):
        linalg.yield %in, %out : f32, f32
      } -> (tensor<16x16x16xf32>, tensor<16x16x16xf32>)
    }
    """
    )
    filecheck(correct, ctx.module)
