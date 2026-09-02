//===- StandalonePasses.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_PASS
#define STANDALONE_PASS

include "mlir/Pass/PassBase.td"

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def MatmulToSystolic : Pass<"icdec-matmul-to-systolic", "::mlir::ModuleOp"> {
  let summary = "Rewrite tiled 8x8 matmul into systolic array driver call";
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
}

def IcdecCompile : Pass<"icdec-compile", "::mlir::ModuleOp"> {
  let summary = "Full ICDeCompiler pipeline: linalg to C with systolic offload";
  let description = [{Runs the complete lowering chain: matmul ops are offloaded to the systolic array driver, remaining ops are lowered to C loops via emitC.}];
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::bufferization::BufferizationDialect", "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect", "mlir::affine::AffineDialect"];
}

def IcdecCompileCpu : Pass<"icdec-compile-cpu", "::mlir::ModuleOp"> {
  let summary = "Lower to plain C without systolic array offload";
  let description = [{
    Same pipeline as icdec-compile but skips the matmul offload pass, so every
    operation is lowered to C loops executed on the CPU. Used for targets
    without the systolic array.
  }];
  let dependentDialects = [
    "mlir::emitc::EmitCDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::affine::AffineDialect"
  ];
}

def ElideCopyAlloc : Pass<"icdec-elide-copy-alloc", "::mlir::func::FuncOp"> {
  let summary = "Remove temp buffers that are only written then copied out";
  let description = [{
    Rewrites the pattern where a memref.alloc is filled by loops and then
    copied wholesale into another buffer: the loops are redirected to write
    directly into the destination, and both the alloc and the copy are erased.
    This keeps generated C free of heap allocations for such buffers.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def OutParam : Pass<"icdec-out-param", "::mlir::ModuleOp"> {
  let summary = "Turn memref return values into output parameters";
  let description = [{
    Rewrites functions that return memrefs so the results become trailing
    parameters instead. Emitted C cannot return arrays, and bare-metal targets
    expect the caller to supply the output buffer.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

#endif // STANDALONE_PASS
