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

#endif // STANDALONE_PASS
