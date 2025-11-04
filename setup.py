"""
================================================================================
# Copyright (c) 2025 Gilad Leef
#
# This software is provided for educational, research, and personal use only.
# Commercial use, resale, or distribution for profit is strictly prohibited.
# All modifications and derivative works must be distributed under the same license terms.
#
# Any disputes arising from the use of this software shall be governed by and construed in accordance with the laws of the State of Israel.
# Exclusive jurisdiction for any such disputes shall lie with the competent courts located in Israel.
================================================================================
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

extra_compile_args = ['-O3', '-march=native']
extra_link_args = []

if os.name == 'posix':
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
elif os.name == 'nt':
    extra_compile_args.append('/openmp')

setup(
    name='rosa_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            name='rosa_cpp',
            sources=['csrc/rosa_cpu.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
