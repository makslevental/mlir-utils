import inspect
from typing import Optional, Sequence

import libcst as cst
import libcst.matchers as m
from bytecode import ConcreteBytecode, ConcreteInstr
from mlir.dialects import scf
from mlir.ir import InsertionPoint, Value

from mlir_utils.ast.canonicalize import (
    StrictTransformer,
    Canonicalizer,
    BytecodePatcher,
)
from mlir_utils.ast.util import ast_call
from mlir_utils.dialects.ext.arith import constant
from mlir_utils.dialects.scf import yield_ as yield__
from mlir_utils.dialects.util import (
    region_op,
    maybe_cast,
    _update_caller_vars,
    get_result_or_results,
)


def _for(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    if isinstance(start, int):
        start = constant(start, index=True)
    if isinstance(stop, int):
        stop = constant(stop, index=True)
    if isinstance(step, int):
        step = constant(step, index=True)
    return scf.ForOp(start, stop, step, iter_args, loc=loc, ip=ip)


for_ = region_op(_for, terminator=yield__)


def range_(
    start,
    stop=None,
    step=None,
    iter_args: Optional[Sequence[Value]] = None,
    *,
    loc=None,
    ip=None,
):
    for_op = _for(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = maybe_cast(for_op.induction_variable)
    iter_args = tuple(map(maybe_cast, for_op.inner_iter_args))
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args
        elif len(iter_args) == 1:
            yield iv, iter_args[0]
        else:
            yield iv
    if len(iter_args):
        previous_frame = inspect.currentframe().f_back
        replacements = tuple(map(maybe_cast, for_op.results_))
        _update_caller_vars(previous_frame, iter_args, replacements)


def yield_(*args):
    yield__(args)


def _if(cond, results_=None, *, has_else=False, loc=None, ip=None):
    if results_ is None:
        results_ = []
    return scf.IfOp(cond, results_, hasElse=has_else, loc=loc, ip=ip)


if_ = region_op(_if, terminator=yield__)

_current_if_op: list[scf.IfOp] = []
_if_ip: InsertionPoint = None


def stack_if(cond: Value, results_=None, has_else=False):
    if results_ is None:
        results_ = []
    assert isinstance(cond, Value)
    global _if_ip, _current_if_op
    if_op = _if(cond, results_, has_else=has_else)
    cond.owner.move_before(if_op)
    _current_if_op.append(if_op)
    _if_ip = InsertionPoint(if_op.then_block)
    _if_ip.__enter__()
    if len(results_):
        return maybe_cast(get_result_or_results(if_op))
    else:
        return True


def stack_endif_branch():
    global _if_ip
    _if_ip.__exit__(None, None, None)


def stack_else():
    global _if_ip, _current_if_op
    _if_ip = InsertionPoint(_current_if_op[-1].else_block)
    _if_ip.__enter__()
    return True


def stack_endif():
    global _current_if_op
    _current_if_op.pop()


_for_ip = None


class ReplaceSCFYield(StrictTransformer):
    @m.leave(m.Yield(value=m.Tuple()))
    def tuple_yield(self, original_node: cst.Yield, updated_node: cst.Yield):
        args = [cst.Arg(e.value) for e in original_node.value.elements]
        return ast_call(yield_.__name__, args)

    @m.leave(m.Yield(value=~m.Tuple()))
    def single_yield(self, original_node: cst.Yield, updated_node: cst.Yield):
        args = [cst.Arg(original_node.value)] if original_node.value else []
        return ast_call(yield_.__name__, args)


class InsertSCFYield(StrictTransformer):
    @m.leave(m.If() | m.Else())
    def leave_(
        self, _original_node: cst.If | cst.Else, updated_node: cst.If | cst.Else
    ) -> cst.If | cst.Else:
        indented_block = updated_node.body
        last_statement = indented_block.body[-1]
        if not isinstance(last_statement, cst.SimpleStatementLine):
            return updated_node.deep_replace(
                indented_block,
                indented_block.with_deep_changes(
                    indented_block,
                    body=list(indented_block.body)
                    + [cst.SimpleStatementLine([cst.Expr(ast_call(yield_.__name__))])],
                ),
            )

        last_statement_body = list(last_statement.body)
        if not (
            isinstance(last_statement.body[0], cst.Expr)
            and isinstance(last_statement.body[0].value, cst.Yield)
        ):
            last_statement_body.append(cst.Expr(ast_call(yield_.__name__)))
            return updated_node.deep_replace(
                last_statement,
                last_statement.with_deep_changes(
                    last_statement, body=last_statement_body
                ),
            )
        return updated_node


class ReplaceSCFCond(StrictTransformer):
    @m.leave(m.If(test=m.NamedExpr()))
    def insert_with_results(
        self, original_node: cst.If, updated_node: cst.If
    ) -> cst.If:
        test = original_node.test
        results = cst.Tuple(
            [
                cst.Element(cst.Name(n))
                for n in self.func_sym_table[original_node.test.target.value]
            ]
        )
        compare = test.value
        assert isinstance(
            compare, cst.Comparison
        ), f"expected cst.Compare from {compare=}"
        new_compare = ast_call(
            stack_if.__name__,
            args=[cst.Arg(compare), cst.Arg(results), cst.Arg(cst.Name(str(True)))],
        )
        new_test = test.deep_replace(compare, new_compare)
        return updated_node.deep_replace(updated_node.test, new_test)

    @m.leave(m.If(test=~m.NamedExpr()))
    def insert_no_results(self, original_node: cst.If, updated_node: cst.If) -> cst.If:
        test = original_node.test
        assert isinstance(
            original_node.test, cst.Comparison
        ), f"expected cst.Compare for {test=}"
        args = [cst.Arg(test)]
        if original_node.orelse:
            args += [cst.Arg(cst.Tuple([])), cst.Arg(cst.Name(str(True)))]
        new_test = ast_call(stack_if.__name__, args=args)
        return updated_node.deep_replace(updated_node.test, new_test)


class InsertEndIfs(StrictTransformer):
    @m.leave(m.If(orelse=None))
    def no_else(self, _original_node: cst.If, updated_node: cst.If) -> cst.If:
        # every if branch needs a scf_endif_branch
        last_then_statement = updated_node.body.body[-1]
        assert isinstance(
            last_then_statement, cst.SimpleStatementLine
        ), f"expected SimpleStatementLine; got {last_then_statement=}"
        last_then_statement_body = list(last_then_statement.body) + [
            cst.Expr(ast_call(stack_endif_branch.__name__))
        ]
        # no else, then need to end the whole if in the body of the true branch
        last_then_statement_body.append(cst.Expr(ast_call(stack_endif.__name__)))
        return updated_node.deep_replace(
            last_then_statement,
            last_then_statement.with_deep_changes(
                last_then_statement, body=last_then_statement_body
            ),
        )

    @m.leave(m.If(orelse=m.Else()))
    def has_else(self, _original_node: cst.If, updated_node: cst.If) -> cst.If:
        # every if branch needs a scf_endif_branch
        last_then_statement = updated_node.body.body[-1]
        assert isinstance(
            last_then_statement, cst.SimpleStatementLine
        ), f"expected SimpleStatementLine; got {last_then_statement=}"
        last_then_statement_body = list(last_then_statement.body) + [
            cst.Expr(ast_call(stack_endif_branch.__name__))
        ]
        updated_node = updated_node.deep_replace(
            last_then_statement,
            last_then_statement.with_deep_changes(
                last_then_statement, body=last_then_statement_body
            ),
        )
        orig_orelse = updated_node.orelse

        # otherwise insert the else
        first_else_statement = updated_node.orelse.body.body[0]
        assert isinstance(
            first_else_statement, cst.SimpleStatementLine
        ), f"expected SimpleStatementLine; got {first_else_statement=}"

        first_else_statement_body = [cst.Expr(ast_call(stack_else.__name__))] + list(
            first_else_statement.body
        )
        orelse = updated_node.orelse.deep_replace(
            first_else_statement,
            first_else_statement.with_deep_changes(
                first_else_statement, body=first_else_statement_body
            ),
        )

        # and end the if after the else branch
        last_else_statement = orelse.body.body[-1]
        assert isinstance(
            last_else_statement, cst.SimpleStatementLine
        ), f"expected SimpleStatementLine; got {last_else_statement=}"
        last_else_statement_body = list(last_else_statement.body) + [
            cst.Expr(ast_call(stack_endif_branch.__name__)),
            cst.Expr(ast_call(stack_endif.__name__)),
        ]
        orelse = orelse.deep_replace(
            last_else_statement,
            last_else_statement.with_deep_changes(
                last_else_statement, body=last_else_statement_body
            ),
        )
        return updated_node.deep_replace(orig_orelse, orelse)


class RemoveJumpsAndInsertGlobals(BytecodePatcher):
    def patch_bytecode(self, code: ConcreteBytecode, f):
        src_lines = inspect.getsource(f).splitlines()
        early_returns = []
        for i, c in enumerate(code):
            if c.name == "RETURN_VALUE":
                early_returns.append(i)

            if c.name in {
                # this is the first test condition jump from python <= 3.10
                "POP_JUMP_IF_FALSE",
                # this is the test condition jump from python >= 3.11
                "POP_JUMP_FORWARD_IF_FALSE",
            }:
                code[i] = ConcreteInstr("POP_TOP", lineno=c.lineno, location=c.location)

            if c.name in {
                # this is the jump after each arm in a conditional
                "JUMP_FORWARD",
                # this is the jump at the end of a for loop
                # "JUMP_BACKWARD",
                # in principle this should be no-oped too but for whatever reason it leads to a stack-size
                # miscalculation (inside bytecode). we don't really need it though because
                # affine_range returns an iterator with length 1
            }:
                # only remove the jump if generated by an if stmt (not a with stmt)
                if "with" not in src_lines[c.lineno - code.first_lineno]:
                    code[i] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

        # early returns cause branches in conditionals to not be visited
        for idx in early_returns[:-1]:
            c = code[idx]
            code[idx] = ConcreteInstr("NOP", lineno=c.lineno, location=c.location)

        # TODO(max): this is bad
        f.__globals__["stack_if"] = stack_if
        f.__globals__["stack_endif_branch"] = stack_endif_branch
        f.__globals__["stack_endif"] = stack_endif
        f.__globals__["stack_else"] = stack_else
        return code


class SCFCanonicalizer(Canonicalizer):
    @property
    def cst_transformers(self):
        return [InsertSCFYield, ReplaceSCFYield, ReplaceSCFCond, InsertEndIfs]

    @property
    def bytecode_patchers(self):
        return [RemoveJumpsAndInsertGlobals]


canonicalizer = SCFCanonicalizer()
