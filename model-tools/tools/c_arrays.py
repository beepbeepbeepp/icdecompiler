# Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
# Author: Muhammad Faudzan Abdullah
# SPDX-License-Identifier: Apache-2.0

"""
Writing numpy arrays as C array definitions, for the weight files.
"""


def dims_of(shape):
    return "".join(f"[{d}]" for d in shape)


def initializer(arr, per_line, indent=""):
    """Lines of the initializer of `arr`, with one brace level per dimension.

    C also takes a flat list for an array of arrays, but compilers warn about
    it (-Wmissing-braces), and the nesting shows which values belong together.
    """
    if arr.ndim == 1:
        values = [str(int(v)) for v in arr]
        rows = [", ".join(values[i:i + per_line])
                for i in range(0, len(values), per_line)]
        if len(rows) == 1:
            return [f"{indent}{{{rows[0]}}}"]
        return ([indent + "{"]
                + [f"{indent}    {row}," for row in rows[:-1]]
                + [f"{indent}    {rows[-1]}", indent + "}"])

    lines = [indent + "{"]
    for k, part in enumerate(arr):
        inner = initializer(part, per_line, indent + "    ")
        if k + 1 < len(arr):
            inner[-1] += ","
        lines += inner
    lines.append(indent + "}")
    return lines


def c_array(ctype, name, arr, dims, per_line):
    """The definition of a const C array holding `arr`, at most `per_line`
    values on a line."""
    lines = initializer(arr, per_line)
    lines[0] = f"const {ctype} {name}{dims} = {lines[0]}"
    lines[-1] += ";"
    return "\n".join(lines)
