from enum import Enum
from fractions import Fraction
from typing import Literal
from textwrap import dedent, indent
from contextlib import contextmanager


from boltzmann.core import CellType


class VelocitySet:
    """
    Information about the velocity set, used in kernel generation.
    """

    def __init__(self, ws: list[Fraction], qs: list[list[int]]) -> None:
        assert len(ws) == len(qs)

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
    [Fraction("1/3"), Fraction("1/6"), Fraction("1/6"), Fraction("1/6"), Fraction("1/6")],
    [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ],
)


class Axis(Enum):
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


class Kernel:
    def __init__(self) -> None:
        self.kernel = ""

    @contextmanager
    def no_format(self):
        """
        Disable clang-format for any statements added inside the block.
        """
        self.kernel += "// clang-format off\n"
        yield
        self.kernel += "// clang-format on\n"

    def __iadd__(self, rhs: str) -> "Kernel":
        self.kernel += rhs
        return self


def gen_kernel_AA1(model: VelocitySet, kernel_type: Literal["fluid", "scalar"]) -> str:
    """
    This is a pretty basic transcoding of my hand-written AA-pattern+BKG kernel for D2Q9 which is
    generic over the velocity set.
    """
    # first get the shape of the velocity set
    nvel = model.Q
    ndim = model.D
    axes = [Axis.X, Axis.Y, Axis.Z][:ndim]

    kernel = Kernel()

    # get the velocity set
    qs = ", ".join(f"{{{', '.join(list(map(str, q)))}}}" for q in model.qs)
    kernel += f"const int qs[{nvel}][{ndim}] = {{{qs}}};\n\n"

    # get the OpenCL indices
    for axis in axes:
        kernel += f"const int i{axis.name} = get_global_id({axis.idx});\n"
    kernel += "\n"

    # check if we actually need to do any work
    cond = " || ".join([f"i{axis.name} >= s[{axis.idx}]" for axis in axes])
    kernel += f"if ({cond}) return;\n\n"

    # calculate the 1d index into the data
    stride = ["1"]
    index = []
    for axis in axes[::-1]:
        s = " * ".join(stride)
        i = f"i{axis.name} * {s}"
        stride.append(f"s[{axis.idx}]")
        index.append(i)

    ii = " + ".join(index)
    kernel += f"const int ii = {ii};\n\n"

    # check the cell type
    fixed = (
        CellType.FIXED_FLUID.value if kernel_type == "fluid" else CellType.FIXED_SCALAR_VALUE.value
    )
    kernel += f"""
    const int c = cell[ii];
    const int wall = (c & {CellType.WALL.value});
    const int fixed = (c & {fixed});

    if (wall) {{
        // wall => do nothing
        return;
    }}
    """

    # calculate offset indices
    inner1 = ""
    inner2 = "0"  # relying on the compiler to optimize this away
    for axis in axes:
        inner1 += f"const int i{axis.name}_ = (i{axis.name} + qs[i][{axis.value}] + s[{axis.idx}]) % s[{axis.idx}];\n"
        inner2 = f"{inner2} * s[{axis.idx}] + i{axis.name}_"
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

    kernel += f"""
    float f_[{nvel}];
    if (even) {{
        {reads_even}
    }} else {{
        {reads_odd}
    }}

    """

    # calculate the moments
    rho = []
    vel = [[] for _ in axes]
    for i in range(nvel):
        rho.append(f"f_[{i}]")
        for axis in axes:
            vel[axis.idx].append(f"(f_[{i}] * {model.qs[i][axis.idx]})")

    rho = " + ".join(rho)
    vel = [f"({'+'.join(v)}) / r" for v in vel]

    with kernel.no_format():
        kernel += f"float r = {rho};\n"
        for axis in axes:
            kernel += f"float v{axis.name} = {vel[axis.idx]};\n"

    vv = " + ".join([f"v{axis.name} * v{axis.name}" for axis in axes])
    kernel += f"const float vv = {vv};\n\n"

    # add gravity if needed
    if kernel_type == "fluid":
        for axis in axes:
            kernel += f"v{axis.name} += (g[{axis.idx}] / omega);\n"

    # update values if fixed
    kernel += """
    if (fixed) {
        omega = 1.0;
        r = rho[ii];
    """
    for axis in axes:
        kernel += f"v{axis.name} = vel[ii*{ndim} + {axis.idx}];\n"
    kernel += "}\n\n"

    # calculate equilibrium & collide
    with kernel.no_format():
        for i in range(nvel):
            uq = " + ".join([f"{model.qs[i][axis.idx]} * v{axis.name}" for axis in axes])
            feq = f"r * ({model.ws[i]}) * (1 + {uq} + 0.5 * (({uq}) * ({uq}) - vv))"
            kernel += f"f_[{i}] += omega * ({feq} - f_[{i}]);\n"

    # write back
    writes_even = ""
    writes_odd = ""
    for i in range(nvel):
        writes_even += f"f[{model.js[i]} + {i}] = f_[{i}];\n"
        writes_odd += f"f[off[0] + {model.js[i]}] = f_[{i}];\n"

    kernel += f"""
    if (even) {{
        {writes_even}
    }} else {{
        {writes_odd}
    }}

    """

    # update macroscopic variables
    kernel += "rho[ii] = r;\n"
    for axis in axes:
        kernel += f"vel[ii*{ndim} + {axis.idx}] = v{axis.name};\n"

    # now finally wrap it in the function call
    args_g = ", ".join(f"float g{a.name}" for a in axes)
    args_s = ", ".join(f"int s{a.name}" for a in axes)

    if kernel_type == "fluid":
        args = f"__constant int *s, int even, float omega, __constant float *g, global float *f, global float *rho, global float *vel, __constant int *cell"
    else:
        args = f"__constant int *s, int even, float omega, global float *f, global float *val, global float *vel, __constant int *cell"

    kernel.kernel = dedent(
        f"""
    kernel void update_d{ndim}q{nvel}_bgk(
        {args}
    ) {{
        {indent(kernel.kernel, "   ")}
    }}
    """.strip()
    )

    return kernel.kernel
