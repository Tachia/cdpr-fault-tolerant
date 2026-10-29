"""Compile CasADi functions to shared libraries once and call them with minimal overhead.

The estimators and the MPC run on nominal parameters, which do not change between runs, so
their functions are generated as C code, compiled with -O2 and cached under build/, keyed by a
hash of the generated code; a campaign pays the compilation once instead of once per run.
The compiled functions are called through ctypes on NumPy buffers (column-major, dense), which
avoids the conversion of large CasADi matrices to NumPy on every call.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import casadi as ca
import numpy as np

BUILD = Path(__file__).resolve().parents[2] / "build"


class CFunction:
    """A compiled CasADi function called on NumPy arrays. Inputs and outputs are dense."""

    def __init__(self, fun: ca.Function, lib_path: Path):
        self.name = fun.name()
        self.lib = ctypes.CDLL(str(lib_path))
        self._f = getattr(self.lib, self.name)
        self._f.restype = ctypes.c_int
        work = getattr(self.lib, self.name + "_work")
        sizes = [ctypes.c_longlong() for _ in range(4)]
        work(*[ctypes.byref(s) for s in sizes])
        n_arg, n_res, n_iw, n_w = (s.value for s in sizes)
        self.iw = np.zeros(max(n_iw, 1), np.int64)
        self.w = np.zeros(max(n_w, 1))
        self.arg = (ctypes.c_void_p * max(n_arg, 1))()
        self.res = (ctypes.c_void_p * max(n_res, 1))()
        self.in_shapes = [fun.size_in(i) for i in range(fun.n_in())]
        self.out = [np.zeros(fun.size_out(i), order="F") for i in range(fun.n_out())]
        for i, o in enumerate(self.out):
            self.res[i] = o.ctypes.data
        self._keep = [None] * fun.n_in()
        self._iw_p = self.iw.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong))
        self._w_p = self.w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def __call__(self, *args):
        for i, (a, shape) in enumerate(zip(args, self.in_shapes)):
            arr = np.asfortranarray(np.asarray(a, float).reshape(shape, order="F"))
            self._keep[i] = arr
            self.arg[i] = arr.ctypes.data
        flag = self._f(self.arg, self.res, self._iw_p, self._w_p, 0)
        if flag:
            raise RuntimeError(f"{self.name} returned {flag}")
        return [o.copy(order="F") for o in self.out]


def compiled(fun: ca.Function) -> CFunction:
    """Return a compiled version of fun (outputs densified), compiling it if not cached."""
    BUILD.mkdir(exist_ok=True)
    name = fun.name()
    sym_in = [ca.SX.sym(f"i{k}", *fun.size_in(k)) for k in range(fun.n_in())]
    dense = ca.Function(name, sym_in, [ca.densify(o) for o in fun.call(sym_in)])
    with tempfile.TemporaryDirectory(dir=BUILD) as tmp:
        gen = ca.CodeGenerator(f"{name}.c", {"with_header": False})
        gen.add(dense)
        gen.generate(tmp + os.sep)
        code = (Path(tmp) / f"{name}.c").read_text()
    key = hashlib.sha256(code.encode()).hexdigest()[:12]
    lib = BUILD / f"{name}_{key}.so"
    if not lib.exists():
        c_file = BUILD / f"{name}_{key}.c"
        c_file.write_text(code)
        part = BUILD / f"{name}_{key}.{os.getpid()}.tmp"
        subprocess.run(["gcc", "-O2", "-fPIC", "-shared", str(c_file), "-o", str(part), "-lm"], check=True)
        part.replace(lib)
    return CFunction(dense, lib)


def compiled_nlpsol(name: str, nlp: dict, plugin: str, opts: dict) -> ca.Function:
    """nlpsol whose callback functions (objective, constraints, derivatives) are generated as C,
    compiled with -O2 and cached; the solver itself (e.g. IPOPT) is unchanged."""
    BUILD.mkdir(exist_ok=True)
    base_opts = {k: v for k, v in opts.items() if k not in ("expand",)}
    symbolic = ca.nlpsol(name, plugin, nlp, {**base_opts, "expand": opts.get("expand", False)})
    with tempfile.TemporaryDirectory(dir=BUILD) as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            symbolic.generate_dependencies(f"{name}.c")
            code = Path(f"{name}.c").read_text()
        finally:
            os.chdir(cwd)
    key = hashlib.sha256(code.encode()).hexdigest()[:12]
    lib = BUILD / f"{name}_{key}.so"
    if not lib.exists():
        c_file = BUILD / f"{name}_{key}.c"
        c_file.write_text(code)
        part = BUILD / f"{name}_{key}.{os.getpid()}.tmp"
        subprocess.run(["gcc", "-O2", "-fPIC", "-shared", str(c_file), "-o", str(part), "-lm"], check=True)
        part.replace(lib)
    return ca.nlpsol(name, plugin, str(lib), base_opts)
