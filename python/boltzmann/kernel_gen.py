from __future__ import annotations

"""
This script auto-generates the kernels for the OpenCL updates.
It's not actually needed to run the module since the OpenCL kernels get compiled into the Rust extension module.
For this reason its dependencies do not form part of the package dependencies (e.g. sympy).
"""

import sympy as sp

from enum import Enum
from fractions import Fraction
from typing import Generator, Literal
from textwrap import dedent, indent
from contextlib import contextmanager
from sympy.printing import ccode

from boltzmann.core import CellFlags


class VelocitySet:
    """
    Information about the velocity set, used in kernel generation.
    """

    def __init__(self, ws: list[Fraction], qs: list[list[int]]) -> None:
        assert len(ws) == len(qs)

        def key(q):
            """
            Sort the velocites in the order we would write them manually.
            Not necessary but makes debugging the auto-generated kernels easier.
            """

            s = next(filter(lambda x: x != 0, q), 0)
            q = [s * qq for qq in q]

            k = 0
            for i, qq in enumerate(q):
                k += (qq % 3) * (3**i) * 2

            # extra term to sort by number of components
            n = sum(qq != 0 for qq in q)
            k += n * (3 ** len(q)) * 2

            # extra term to put opposite pairs together
            if s < 0:
                k += 1

            return k

        # sort together
        wq = [(w, q) for w, q in sorted(zip(ws, qs), key=lambda x: key(x[1]))]

        # pull out ws & qs
        ws = [w for w, _ in wq]
        qs = [q for _, q in wq]

        def _neg(x: list[int]) -> list[int]:
            return [-xx for xx in x]

        # index of opposite velocity
        js = [qs.index(_neg(q)) for q in qs]

        self.ws = ws
        self.qs = qs
        self.js = js

        self.Q = len(self.qs)
        self.D = len(self.qs[0])


def tensor_product(m1: VelocitySet, m2: VelocitySet) -> VelocitySet:
    """
    Take the tensor product of two velocity sets.
    """
    qs = []
    ws = []
    for i in range(m1.Q):
        for j in range(m2.Q):
            q = [qq for qq in m1.qs[i]]
            q.extend(m2.qs[j])
            qs.append(q)
            ws.append(m1.ws[i] * m2.ws[j])

    ws = [w / sum(ws) for w in ws]

    return VelocitySet(ws, qs)


D1Q3 = VelocitySet(
    [Fraction("4/6"), Fraction("1/6"), Fraction("1/6")],
    [
        [0],
        [1],
        [-1],
    ],
)

D2Q9 = tensor_product(D1Q3, D1Q3)
D3Q27 = tensor_product(D2Q9, D1Q3)

D2Q5 = VelocitySet(
    [
        Fraction("1/3"),
        Fraction("1/6"),
        Fraction("1/6"),
        Fraction("1/6"),
        Fraction("1/6"),
    ],
    [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ],
)


def make_advdiff_isotropic(d: int) -> VelocitySet:
    """
    Make an isotropic advection-diffusion velocity set.
    See https://journals.aps.org/pre/pdf/10.1103/PhysRevE.105.025308.
    """
    match d:
        case 1:
            f = Fraction("1/3")
        case 2:
            f = Fraction("2/3")
        case 3:
            f = Fraction("19/20")
        case _:
            raise ValueError(f"Invalid dimension {d}")

    ws = [Fraction(1, 1) - f]
    qs = [[0] * d]

    w = f / (d * 2)
    for i in range(d):
        q1 = [0] * d
        q2 = [0] * d
        q1[i] = 1
        q2[i] = -1
        qs.extend([q1, q2])
        ws.extend([w, w])

    return VelocitySet(ws, qs)


D3Q7 = make_advdiff_isotropic(3)


class Axis(Enum):
    """
    :meta private:
    """

    X = 0
    Y = 1
    Z = 2

    @property
    def idx(self) -> int:
        return self.value

    @property
    def name(self) -> str:
        match self:
            case Axis.X:
                return "x"
            case Axis.Y:
                return "y"
            case Axis.Z:
                return "z"


def _prepare(block: str, indent_lvl: int) -> str:
    return indent(dedent(block).strip(), "    " * indent_lvl) + "\n"


class Kernel:
    """
    Simple class to make building the kernel string less unpleasant.
    """

    def __init__(self, indent: int = 0) -> None:
        self.kernel = ""
        self.indent = indent
        self.is_if = False

    @contextmanager
    def no_format(self):
        """
        Disable clang-format for any statements added inside the block.
        """
        self.kernel += "// clang-format off\n"
        yield
        self.kernel += "// clang-format on\n"

    @contextmanager
    def block(self):
        """
        A logical "block" of operations which belong together.
        This puts a newline after the end of the block to space things out.
        """
        yield
        self.kernel += "\n"

    @contextmanager
    def if_(self, condition: str) -> Generator[Kernel, None, None]:
        """
        Contents will be wrapped in an in if-statement.
        """
        self.kernel += f"if ({condition}) {{\n"
        inner = Kernel(self.indent + 1)
        yield inner
        self.kernel += _prepare(inner.kernel, self.indent + 1)
        self.kernel += "}"
        self.is_if = True

    @contextmanager
    def else_(self) -> Generator[Kernel, None, None]:
        """
        Contents will be wrapped in an else-statement.
        """
        assert self.is_if, "'else' block must follow an 'if'."
        self.is_if = False
        self.kernel += " else {\n"
        inner = Kernel(self.indent + 1)
        yield inner
        self.kernel += _prepare(inner.kernel.strip(), self.indent + 1)
        self.kernel += "}\n\n"

    def __iadd__(self, rhs: str) -> "Kernel":
        if self.is_if:
            self.kernel += "\n\n"
            self.is_if = False
        self.kernel += rhs + "\n"
        return self


def _dot(x: list[sp.Expr], f: list[sp.Expr | int]) -> sp.Expr:
    """
    Simplified dot product of sympy expressions.
    """
    return sp.simplify(sp.Add(*[sp.Mul(xx, ff) for xx, ff in zip(x, f)]))


def gen_kernel_AA_v1(
    model: VelocitySet,
    kernel_type: Literal["fluid", "scalar"],
    omega_type: Literal["bgk", "trt"],
) -> str:
    """
    This is a pretty basic transcoding of my hand-written AA-pattern+BKG kernel for D2Q9 which is
    generic over the velocity set.
    """
    # first get the shape of the velocity set
    nvel = model.Q
    ndim = model.D
    axes = [Axis.X, Axis.Y, Axis.Z][:ndim]

    # sympy symbols useful for later
    v_ = sp.symbols([f"v{a.name}" for a in axes])
    f_ = sp.symbols([f"f_[{i}]" for i in range(nvel)])

    kernel = Kernel()

    val_name = "rho" if kernel_type == "fluid" else "val"

    # get the velocity set
    qs = ", ".join(f"{{{', '.join(list(map(str, q)))}}}" for q in model.qs)
    kernel += f"const int qs[{nvel}][{ndim}] = {{{qs}}};\n"

    # get the OpenCL indices
    for axis in axes:
        kernel += f"const size_t i{axis.name} = get_global_id({axis.idx});"
    kernel += "\n"

    # check if we actually need to do any work
    cond = " || ".join([f"i{axis.name} >= s[{axis.idx}]" for axis in axes])
    kernel += f"if ({cond}) return;\n"

    # calculate the 1d index into the data
    stride = [1]
    index = []
    for axis in axes[::-1]:
        s = sp.Mul(*stride)
        i = sp.Symbol(f"i{axis.name}") * s
        stride.append(sp.Symbol(f"s[{axis.idx}]"))
        index.append(i)

    ii = sp.simplify(sp.Add(*index))
    kernel += f"const size_t ii = {ii};"

    # check the cell type
    fixed = CellFlags.FIXED_FLUID if kernel_type == "fluid" else CellFlags.FIXED_SCALAR_VALUE
    kernel += f"""
    const int c = cell[ii];
    const bool wall = (c & {CellFlags.WALL});
    const bool fixed = (c & {fixed});

    if (wall) {{
        // wall => do nothing
        return;
    }}
    """

    # calculate offset indices
    inner1 = ""
    inner2 = 0
    for axis in axes:
        inner1 += f"const int i{axis.name}_ = (i{axis.name} + qs[i][{axis.value}] + s[{axis.idx}]) % s[{axis.idx}];\n"
        inner2 = inner2 * sp.Symbol(f"s[{axis.idx}]") + sp.Symbol(f"i{axis.name}_")

    inner2 = sp.simplify(inner2)

    kernel += f"""
    int off[{nvel}];

    #pragma unroll
    for (int i = 0; i < {nvel}; i++) {{
        {inner1}
        off[i] = ({inner2}) * {nvel};
    }}
    """

    # now the reads
    reads_even = ""
    reads_odd = ""
    for i in range(nvel):
        reads_even += f"f_[{i}] = f[off[{model.js[i]}] + {i}];\n"
        reads_odd += f"f_[{i}] = f[off[0] + {model.js[i]}];\n"

    kernel += f"float f_[{nvel}];"
    with kernel.if_("even") as inner:
        inner += reads_even
    with kernel.else_() as inner:
        inner += reads_odd

    # calculate the moments
    rho = []
    vel = [[] for _ in axes]
    for i in range(nvel):
        rho.append(f"f_[{i}]")
        for axis in axes:
            vel[axis.idx].append(model.qs[i][axis.idx])

    rho = " + ".join(rho)
    vel = [str(_dot(f_, v)) for v in vel]

    with kernel.no_format():
        kernel += f"float r = {rho};"
        for axis in axes:
            if kernel_type == "fluid":
                kernel += f"float v{axis.name} = ({vel[axis.idx]}) / r;"
            else:
                kernel += f"float v{axis.name} = vel[{ndim} * ii + {axis.idx}];"

    # add gravity if needed
    if kernel_type == "fluid":
        with kernel.if_("use_acc") as inner:
            for axis in axes:
                omega = "omega" if omega_type == "bgk" else "omega_pos"
                inner += f"v{axis.name} += acc[ii * {ndim} + {axis.idx}] / {omega};"

    # update values if fixed
    with kernel.if_("fixed") as inner:
        if omega_type == "bgk":
            inner += "omega = 1.0;"
        else:
            inner += "omega_pos = 1.0;"
            inner += "omega_neg = 1.0;"

        inner += f"r = {val_name}[ii];".strip()

        for axis in axes:
            inner += f"v{axis.name} = vel[ii*{ndim} + {axis.idx}];"

    # calculate equilibrium & collide
    vv = " + ".join([f"v{axis.name} * v{axis.name}" for axis in axes])
    kernel += f"const float vv = {vv};\n\n"

    if omega_type == "trt":
        # temporary variables needed to store intermediate calculations
        with kernel.block():
            kernel += "float fpos;"
            kernel += "float fneg;"

    vv = sp.Symbol("vv")
    with kernel.block():
        with kernel.no_format():
            for i in range(nvel):
                if omega_type == "bgk":
                    uq = _dot(v_, model.qs[i])
                    feq = model.ws[i] * (1 + 3 * uq + 0.5 * (9 * (uq * uq) - 3 * vv))
                    feq = ccode(sp.simplify(feq, rational=True))
                    feq = f"r * ({feq})"
                    kernel += f"f_[{i}] += omega * ({feq} - f_[{i}]);"

                if omega_type == "trt":
                    j = model.js[i]
                    if j < i:
                        continue
                    uq_i = _dot(v_, model.qs[i])
                    uq_j = _dot(v_, model.qs[j])
                    feq_i = model.ws[i] * (1 + 3 * uq_i + 0.5 * (9 * (uq_i * uq_i) - 3 * vv))
                    feq_j = model.ws[j] * (1 + 3 * uq_j + 0.5 * (9 * (uq_j * uq_j) - 3 * vv))
                    feq_pos = (feq_i + feq_j) * 0.5
                    feq_neg = (feq_i - feq_j) * 0.5
                    feq_pos = "r * ({})".format(ccode(sp.simplify(feq_pos, rational=True)))
                    feq_neg = "r * ({})".format(ccode(sp.simplify(feq_neg, rational=True)))
                    f_pos = f"(f_[{i}] + f_[{j}]) * 0.5"
                    f_neg = f"(f_[{i}] - f_[{j}]) * 0.5"
                    kernel += f"fpos = omega_pos * ({feq_pos} - {f_pos});"
                    kernel += f"fneg = omega_neg * ({feq_neg} - {f_neg});"
                    kernel += f"f_[{i}] += fpos + fneg;"
                    if j != i:  # i.e. zero velocity
                        kernel += f"f_[{j}] += fpos - fneg;"

    # write back
    writes_even = ""
    writes_odd = ""
    for i in range(nvel):
        writes_even += f"f[off[{model.js[i]}] + {i}] = f_[{model.js[i]}];\n"
        writes_odd += f"f[off[0] + {i}] = f_[{i}];\n"

    with kernel.block():
        with kernel.if_("even") as inner:
            inner += writes_even
        with kernel.else_() as inner:
            inner += writes_odd

    # update macroscopic variables
    kernel += f"{val_name}[ii] = r;"
    if kernel_type == "fluid":
        for axis in axes:
            kernel += f"vel[ii*{ndim} + {axis.idx}] = v{axis.name};"

    # now finally wrap it in the function call
    if omega_type == "bgk":
        omega_args = "float omega"
    elif omega_type == "trt":
        omega_args = "float omega_pos, float omega_neg"

    if kernel_type == "fluid":
        args = f"__constant int *s, int even, {omega_args}, global float *f, global float *rho, global float *vel, global float *acc, int use_acc, global int *cell"
    else:
        args = f"__constant int *s, int even, {omega_args}, global float *f, global float *val, global float *vel, global int *cell"

    kernel.kernel = dedent(
        f"""
    kernel void update_d{ndim}q{nvel}_{omega_type}(
        {args}
    ) {{
        {indent(kernel.kernel, "   ")}
    }}
    """.strip()
    )

    return kernel.kernel


if __name__ == "__main__":
    import sys

    fname = sys.argv[1]

    with open(fname, "w") as fout:
        for model, kernel in [
            (D2Q5, "scalar"),
            (D2Q9, "fluid"),
            (D3Q7, "scalar"),
            (D3Q27, "fluid"),
        ]:
            print(gen_kernel_AA_v1(model, kernel), file=fout)
            print("\n\n", file=fout)
