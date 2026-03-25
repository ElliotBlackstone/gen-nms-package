# This file is derived from torchvision's setup.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

from pathlib import Path
import os
import shlex
import sys
import sysconfig
import glob
import shutil
import distutils.command.clean

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    ROCM_HOME,
)



FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS")

ROOT_DIR = Path(__file__).resolve().parent
CSRC_DIR = ROOT_DIR / "gen_nms" / "csrc"
IS_ROCM = (torch.version.hip is not None) and (ROCM_HOME is not None)
BUILD_CUDA_SOURCES = ((CUDA_HOME is not None) or IS_ROCM) or FORCE_CUDA


print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("CUDA_HOME =", CUDA_HOME)
print("IS_ROCM =", IS_ROCM)
print("FORCE_CUDA =", FORCE_CUDA)
print("BUILD_CUDA_SOURCES =", BUILD_CUDA_SOURCES)


def get_macros_and_flags():
    define_macros: list[tuple[str, str | None]] = []
    extra_compile_args: dict[str, list[str]] = {"cxx": []}

    is_windows = (sys.platform == "win32")

    if is_windows:
        extra_compile_args["cxx"] += [
            "/MP",
            "/std:c++17",
            "/Zc:__cplusplus",
        ]
    else:
        extra_compile_args["cxx"] += ["-std=c++17"]

    if BUILD_CUDA_SOURCES:
        if IS_ROCM:
            define_macros += [("WITH_HIP", None)]
            nvcc_flags: list[str] = []
        else:
            define_macros += [("WITH_CUDA", None)]
            nvcc_flags = [] if NVCC_FLAGS is None else shlex.split(NVCC_FLAGS)

            if "-std=c++17" not in nvcc_flags:
                nvcc_flags.append("-std=c++17")

            if is_windows:
                nvcc_flags += [
                    "-Xcompiler", "/std:c++17",
                    "-Xcompiler", "/Zc:__cplusplus",
                ]

        extra_compile_args["nvcc"] = nvcc_flags

    if DEBUG:
        if is_windows:
            extra_compile_args["cxx"] += ["/Od"]
        else:
            extra_compile_args["cxx"] += ["-g", "-O0"]

        if "nvcc" in extra_compile_args:
            filtered = []
            for f in extra_compile_args["nvcc"]:
                if f in {"-O0", "-O1", "-O2", "-O3", "-g"}:
                    continue
                filtered.append(f)
            extra_compile_args["nvcc"] = filtered + ["-O0", "-g"]
    else:
        if not is_windows:
            extra_compile_args["cxx"].append("-g0")

    return define_macros, extra_compile_args


def make_C_extension():
    sources = (
        list(CSRC_DIR.glob("*.cpp"))
        + list(CSRC_DIR.glob("ops/*.cpp"))
        + list(CSRC_DIR.glob("ops/cpu/*.cpp"))
    )

    if BUILD_CUDA_SOURCES:
        sources += list(CSRC_DIR.glob("ops/cuda/*.cu"))
        Extension = CUDAExtension
    else:
        Extension = CppExtension

    define_macros, extra_compile_args = get_macros_and_flags()

    return Extension(
        name="gen_nms._C",
        sources=[s.relative_to(ROOT_DIR).as_posix() for s in sorted(sources)],
        include_dirs=[str(CSRC_DIR.relative_to(ROOT_DIR).as_posix())],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore") as f:
            for wildcard in filter(None, f.read().splitlines()):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)
        super().run()


setup(
    packages=["gen_nms", "gen_nms.ops"],
    ext_modules=[make_C_extension()],
    zip_safe=False,
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
        "clean": clean,
    },
)