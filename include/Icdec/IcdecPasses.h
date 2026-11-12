//===- IcdecPasses.h - ICDeCompiler passes ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ICDEC_ICDECPASSES_H
#define ICDEC_ICDECPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace icdec {
#define GEN_PASS_DECL
#include "Icdec/IcdecPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Icdec/IcdecPasses.h.inc"
} // namespace icdec
} // namespace mlir

#endif
