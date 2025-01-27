from . import arith
from ...util import get_user_code_loc
from ....dialects import linalg
from ....dialects._linalg_enum_gen import _iteratortypeenum

# noinspection PyUnresolvedReferences
from ....dialects.linalg import *
from ....extras import types as T
from ...meta import region_op
from ....ir import Context, register_attribute_builder, ArrayAttr, RankedTensorType
from ....dialects._ods_common import get_default_loc_context


def abs(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.abs(I, loc=loc, ip=ip, outs=[O])


def add(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.add(lhs, rhs, loc=loc, ip=ip, outs=[O])


def batch_matmul(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_matmul(A, B, loc=loc, ip=ip, outs=[C])


def batch_matmul_transpose_a(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_matmul_transpose_a(A, B, loc=loc, ip=ip, outs=[C])


def batch_matmul_transpose_b(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_matmul_transpose_b(A, B, loc=loc, ip=ip, outs=[C])


def batch_matvec(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_matvec(A, B, loc=loc, ip=ip, outs=[C])


def batch_mmt4d(lhs, rhs, accum, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_mmt4d(lhs, rhs, loc=loc, ip=ip, outs=[accum])


def batch_reduce_matmul(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_reduce_matmul(A, B, loc=loc, ip=ip, outs=[C])


def batch_vecmat(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.batch_vecmat(A, B, loc=loc, ip=ip, outs=[C])


def ceil(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.ceil(I, loc=loc, ip=ip, outs=[O])


def conv_1d(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_1d(I, K, loc=loc, ip=ip, outs=[O])


def conv_1d_ncw_fcw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_1d_ncw_fcw(I, K, loc=loc, ip=ip, outs=[O])


def conv_1d_nwc_wcf(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_1d_nwc_wcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nchw_fchw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_nchw_fchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_ngchw_fgchw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_ngchw_fgchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_ngchw_gfchw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_ngchw_gfchw(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_fhwc(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_nhwc_fhwc(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_fhwc_q(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_nhwc_fhwc_q(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_hwcf(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_nhwc_hwcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_2d_nhwc_hwcf_q(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_2d_nhwc_hwcf_q(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_3d(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ncdhw_fcdhw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_3d_ncdhw_fcdhw(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ndhwc_dhwcf(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_3d_ndhwc_dhwcf(I, K, loc=loc, ip=ip, outs=[O])


def conv_3d_ndhwc_dhwcf_q(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.conv_3d_ndhwc_dhwcf_q(I, K, loc=loc, ip=ip, outs=[O])


def copy(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.copy(I, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_ncw_cw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_1d_ncw_cw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_nwc_wc(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_1d_nwc_wc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_1d_nwc_wcm(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_1d_nwc_wcm(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nchw_chw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_2d_nchw_chw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwc(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_2d_nhwc_hwc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwc_q(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_2d_nhwc_hwc_q(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwcm(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_2d_nhwc_hwcm(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_2d_nhwc_hwcm_q(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_2d_nhwc_hwcm_q(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ncdhw_cdhw(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_3d_ncdhw_cdhw(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ndhwc_dhwc(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_3d_ndhwc_dhwc(I, K, loc=loc, ip=ip, outs=[O])


def depthwise_conv_3d_ndhwc_dhwcm(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.depthwise_conv_3d_ndhwc_dhwcm(I, K, loc=loc, ip=ip, outs=[O])


def div(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.div(lhs, rhs, loc=loc, ip=ip, outs=[O])


def div_unsigned(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.div_unsigned(lhs, rhs, loc=loc, ip=ip, outs=[O])


def dot(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.dot(A, B, loc=loc, ip=ip, outs=[C])


def elemwise_binary(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.elemwise_binary(lhs, rhs, loc=loc, ip=ip, outs=[O])


def elemwise_unary(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.elemwise_unary(I, loc=loc, ip=ip, outs=[O])


def exp(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.exp(I, loc=loc, ip=ip, outs=[O])


def fill(v, O, *, loc=None, ip=None):
    if isinstance(v, (float, int, bool)):
        v = arith.constant(v)
    if loc is None:
        loc = get_user_code_loc()
    return linalg.fill(v, loc=loc, ip=ip, outs=[O])


def fill_rng_2d(min, max, seed, O, *, loc=None, ip=None):
    params = [min, max]
    for i, m in enumerate(params):
        if isinstance(m, (float, int)):
            params[i] = arith.constant(m, type=T.f64())
    min, max = params
    if isinstance(seed, int):
        seed = arith.constant(seed, T.i32())
    if loc is None:
        loc = get_user_code_loc()
    return linalg.fill_rng_2d(min, max, seed, loc=loc, ip=ip, outs=[O])


def floor(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.floor(I, loc=loc, ip=ip, outs=[O])


def log(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.log(I, loc=loc, ip=ip, outs=[O])


@linalg.linalg_structured_op
def _matmul_generic(
    A=linalg.TensorDef(linalg.T1, linalg.S.M, linalg.S.K),
    B=linalg.TensorDef(linalg.T2, linalg.S.K, linalg.S.N),
    C=linalg.TensorDef(linalg.U, linalg.S.M, linalg.S.N, output=True),
    cast=linalg.TypeFnAttrDef(default=linalg.TypeFn.cast_signed),
):
    linalg.domain(linalg.D.m, linalg.D.n, linalg.D.k)
    linalg.implements(linalg.ContractionOpInterface)
    C[linalg.D.m, linalg.D.n] += cast(linalg.U, A[linalg.D.m, linalg.D.k]) * cast(
        linalg.U, B[linalg.D.k, linalg.D.n]
    )


_matmul_generic.op_name = "matmul"


def matmul(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()

    op_configs = linalg.LinalgOpConfig.from_linalg_op_def(
        _matmul_generic.op_def, context=Context.current
    )
    op_config = op_configs[0]
    (
        _all_arg_defs,
        _in_arg_defs,
        _out_arg_defs,
        _outs,
        result_types,
        _type_mapping,
        indexing_maps_attr,
        _iterator_types_attr,
        _index_attrs,
        _fn_attr_mapping,
        _block_arg_types,
    ) = linalg.opdsl.lang.emitter.prepare_common_structured_op(
        op_config.structured_op, A, B, outs=[C], loc=loc, ip=ip
    )
    named_op = linalg.MatmulOp(
        result_types,
        inputs=[A, B],
        outputs=[C],
        indexing_maps=indexing_maps_attr,
        cast=linalg.TypeFn.cast_signed,
        loc=loc,
        ip=ip,
    )
    linalg.fill_builtin_region(named_op.operation)
    return named_op.results


def matmul_transpose_a(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.matmul_transpose_a(A, B, loc=loc, ip=ip, outs=[C])


def matmul_transpose_b(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.matmul_transpose_b(A, B, loc=loc, ip=ip, outs=[C])


def matmul_unsigned(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.matmul_unsigned(A, B, loc=loc, ip=ip, outs=[C])


def matvec(A, y, x, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.matvec(A, y, loc=loc, ip=ip, outs=[x])


def max(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.max(lhs, rhs, loc=loc, ip=ip, outs=[O])


def mmt4d(lhs, rhs, accum, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.mmt4d(lhs, rhs, loc=loc, ip=ip, outs=[accum])


def mul(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.mul(lhs, rhs, loc=loc, ip=ip, outs=[O])


def negf(I, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.negf(I, loc=loc, ip=ip, outs=[O])


def pooling_nchw_max(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nchw_max(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nchw_sum(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nchw_sum(I, K, loc=loc, ip=ip, outs=[O])


def pooling_ncw_max(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_ncw_max(I, K, loc=loc, ip=ip, outs=[O])


def pooling_ncw_sum(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_ncw_sum(I, K, loc=loc, ip=ip, outs=[O])


def pooling_ndhwc_max(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_ndhwc_max(I, K, loc=loc, ip=ip, outs=[O])


def pooling_ndhwc_min(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_ndhwc_min(I, K, loc=loc, ip=ip, outs=[O])


def pooling_ndhwc_sum(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_ndhwc_sum(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nhwc_max(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nhwc_max(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nhwc_max_unsigned(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nhwc_max_unsigned(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nhwc_min(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nhwc_min(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nhwc_min_unsigned(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nhwc_min_unsigned(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nhwc_sum(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nhwc_sum(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nwc_max(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nwc_max(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nwc_max_unsigned(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nwc_max_unsigned(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nwc_min(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nwc_min(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nwc_min_unsigned(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nwc_min_unsigned(I, K, loc=loc, ip=ip, outs=[O])


def pooling_nwc_sum(I, K, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.pooling_nwc_sum(I, K, loc=loc, ip=ip, outs=[O])


def quantized_batch_matmul(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.quantized_batch_matmul(A, B, loc=loc, ip=ip, outs=[C])


def quantized_matmul(A, B, C, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.quantized_matmul(A, B, loc=loc, ip=ip, outs=[C])


def sub(lhs, rhs, O, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.sub(lhs, rhs, loc=loc, ip=ip, outs=[O])


def vecmat(y, A, x, *, loc=None, ip=None):
    if loc is None:
        loc = get_user_code_loc()
    return linalg.vecmat(y, A, loc=loc, ip=ip, outs=[x])


@register_attribute_builder("IteratorTypeArrayAttr")
def _IteratorTypeArrayAttr(x, context):
    return ArrayAttr.get([_iteratortypeenum(v, context) for v in x])


class GenericOp(GenericOp):
    def __init__(
        self,
        inputs,
        outputs,
        indexing_maps,
        iterator_types,
        *,
        doc=None,
        library_call=None,
        loc=None,
        ip=None,
    ):
        if loc is None:
            loc = get_user_code_loc()
        result_types = []
        if isinstance(outputs[0].type, RankedTensorType):
            result_types = [o.type for o in outputs]

        super().__init__(
            result_types,
            inputs,
            outputs,
            indexing_maps,
            iterator_types,
            doc=doc,
            library_call=library_call,
            loc=loc,
            ip=ip,
        )
        element_types = [i.type.element_type for i in inputs] + [
            o.type.element_type for o in outputs
        ]
        self.regions[0].blocks.append(*element_types)


generic = region_op(GenericOp, terminator=YieldOp)
