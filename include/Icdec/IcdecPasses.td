//===- IcdecPasses.td - ICDeCompiler passes ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ICDEC_PASS
#define ICDEC_PASS

include "mlir/Pass/PassBase.td"

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

def AddIncludes : Pass<"icdec-add-includes", "::mlir::ModuleOp"> {
  let summary = "Insert the C headers the generated code needs";
  let description = [{
    Emitted code uses size_t, bool and heap allocation but does not declare the
    headers that provide them. This pass prepends the required includes so the
    output compiles as-is.
  }];
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
}

def ExpandIntMinMax : Pass<"icdec-expand-int-minmax", "::mlir::ModuleOp"> {
  let summary = "Rewrite arith.maxsi and arith.minsi into compare and select";
  let description = [{
    EmitC has no lowering for integer min and max, but quantized models produce
    them from clamp operations. This pass expands them into a comparison
    followed by a select, both of which EmitC supports.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def GlobalToEmitC : Pass<"icdec-global-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert constant memref globals into emitc globals";
  let description = [{
    The bundled emitc conversion rewrites memref.get_global into
    emitc.get_global but leaves the memref.global declaration behind, so the
    symbol no longer resolves. Quantized models hit this because rescale
    parameters become constant globals. This pass converts the declarations.
  }];
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::memref::MemRefDialect"];
}

def AllocToArena : Pass<"icdec-alloc-to-arena", "::mlir::func::FuncOp"> {
  let summary = "Place static allocations in one shared arena";
  let description = [{
    Bare-metal targets have no heap worth using, and giving every temporary its
    own global wastes memory: most of them are dead by the time the next layer
    runs. This pass works out when each buffer is live, then packs the ones
    whose lifetimes do not overlap into the same bytes of a single arena.

    Only allocations with a fully static shape are placed. Anything else is
    left alone.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def ConvToIm2Col : Pass<"icdec-conv-to-im2col", "::mlir::func::FuncOp"> {
  let summary = "Rewrite convolutions as im2col plus matmul";
  let description = [{
    A convolution is a matrix multiply once the input windows are laid out as
    rows. Rewriting it that way lets the matmul offload pass send the bulk of
    the work to the accelerator instead of leaving it on the CPU.
  }];
  let dependentDialects = ["mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

def ViewToEmitC : Pass<"icdec-view-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower memref.view into emitc pointer arithmetic";
  let description = [{
    The bundled emitc conversion has no pattern for memref.view, which the
    arena allocator produces. A view is just a typed pointer into the arena, so
    it becomes an address-of plus a cast.
  }];
  let dependentDialects = ["mlir::emitc::EmitCDialect"];
}

def PrintSummary : Pass<"icdec-print-summary", "::mlir::ModuleOp"> {
  let summary = "Report what the compiler did with the model";
  let description = [{
    Prints operator counts, how much went to the accelerator, and how much
    memory the generated code needs. Written to stderr so it does not end up
    in the generated source.
  }];
}

def ReinterpretCastToIndex : Pass<"icdec-reinterpret-cast-to-index", "::mlir::func::FuncOp"> {
  let summary = "Rewrite accesses through reinterpret_cast as accesses to the source";
  let description = [{
    A reinterpret_cast over contiguous memory only relabels the shape. EmitC
    cannot express that relabelling, so every load and store through the cast
    is rewritten to address the source directly: the index is linearised with
    the cast's strides, then split back into the source's shape.

    Copies through a cast are expanded into loops first so they reduce to the
    same load/store case. Once nothing uses the cast it is erased.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect", "mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
}

#endif // ICDEC_PASS
