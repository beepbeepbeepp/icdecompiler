//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"

#include "Standalone/StandalonePasses.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#define GEN_PASS_DEF_MATMULTOSYSTOLIC
#define GEN_PASS_DEF_ICDECCOMPILE
#include "Standalone/StandalonePasses.h.inc"

namespace {
class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class StandaloneSwitchBarFoo
    : public impl::StandaloneSwitchBarFooBase<StandaloneSwitchBarFoo> {
public:
  using impl::StandaloneSwitchBarFooBase<
      StandaloneSwitchBarFoo>::StandaloneSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<StandaloneSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

// Shared rewrite: replace a matmul-like op with a systolic driver call.
// Returns failure if shapes are not static 8x8.
static LogicalResult rewriteToSystolicCall(Operation *op, ValueRange inputs, ValueRange outputs, PatternRewriter &rewriter) {
  
  auto aType = dyn_cast<MemRefType>(inputs[0].getType());
  auto bType = dyn_cast<MemRefType>(inputs[1].getType());
  if (!aType || !bType)
    return failure();
  if (!aType.hasStaticShape() || !bType.hasStaticShape())
    return failure();

// matmul dims live in the last two axes; leading axes are batch
  if (aType.getRank() < 2 || bType.getRank() < 2)
    return failure();

  auto aShape = aType.getShape();
  auto bShape = bType.getShape();
  int64_t M = aShape[aType.getRank() - 2];
  int64_t K = aShape[aType.getRank() - 1];
  int64_t N = bShape[bType.getRank() - 1];
  
  // the driver tiles internally and pads partial tiles,
  // so any static M/K/N is acceptable

  // materialize M, K, N as i32 constants
  auto i32Type = rewriter.getI32Type();
  SmallVector<Value> args;
  for (int64_t dim : {M, K, N}) {
    auto attr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));
    args.push_back(rewriter.create<emitc::ConstantOp>(
        op->getLoc(), i32Type, attr));
  }

  // walk back through casts and shape-only ops to reach the underlying buffer
  auto unwrap = [](Value v) -> Value {
    while (true) {
      if (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() != 1)
          break;
        v = cast.getInputs()[0];
        continue;
      }
      if (auto collapse = v.getDefiningOp<memref::CollapseShapeOp>()) {
        v = collapse.getSrc();
        continue;
      }
      if (auto expand = v.getDefiningOp<memref::ExpandShapeOp>()) {
        v = expand.getSrc();
        continue;
      }
      break;
    }
    return v;
  };

// convert each buffer to a flat C pointer: &arr[0]...[0]
  auto toPointer = [&](Value v) -> Value {
    Value base = unwrap(v);

    // if still a memref, cast it to an equivalent emitc.array first
    if (auto memrefType = dyn_cast<MemRefType>(base.getType())) {
      if (!memrefType.hasStaticShape())
        return Value();
      auto arrTy = emitc::ArrayType::get(memrefType.getShape(), memrefType.getElementType());
      base = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), arrTy, base).getResult(0);
    }
    
    auto arrType = dyn_cast<emitc::ArrayType>(base.getType());
    if (!arrType)
      return Value();

    // build one zero index per array dimension
    auto indexType = rewriter.getIndexType();
    auto zeroAttr = rewriter.getIndexAttr(0);
    Value zero = rewriter.create<emitc::ConstantOp>(
        op->getLoc(), indexType, zeroAttr);
    SmallVector<Value> indices(arrType.getRank(), zero);

  auto typedBase = cast<TypedValue<emitc::ArrayType>>(base);
    Value elem = rewriter.create<emitc::SubscriptOp>(
        op->getLoc(), typedBase, ValueRange(indices));

    auto ptrType = emitc::PointerType::get(arrType.getElementType());
    return rewriter.create<emitc::AddressOfOp>(op->getLoc(), ptrType, elem);
  };

  for (Value v : inputs) {
    Value p = toPointer(v);
    if (!p)
      return failure();
    args.push_back(p);
  }
  for (Value v : outputs) {
    Value p = toPointer(v);
    if (!p)
      return failure();
    args.push_back(p);
  }

  rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(),
      TypeRange{},
      rewriter.getStringAttr("systolic_multiply_large"),
      args);

  // block until the array finishes before anyone reads the result
  auto timeoutAttr = rewriter.getI32IntegerAttr(10000);
  Value timeout = rewriter.create<emitc::ConstantOp>(op->getLoc(), i32Type, timeoutAttr);
  SmallVector<Value> waitArgs{timeout};
  rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), TypeRange{},rewriter.getStringAttr("systolic_wait_done"),waitArgs);

  rewriter.eraseOp(op);
  return success();
}

class MatmulToSystolicRewriter: public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const final{

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

class GenericMatmulToSystolicRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // shape check: exactly two inputs and one output
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    // iterator check: must contain a reduction dimension
    auto iterTypes = op.getIteratorTypesArray();
    bool hasReduction = llvm::any_of(iterTypes, [](utils::IteratorType t) {
      return t == utils::IteratorType::reduction;
    });
    if (!hasReduction)
      return failure();

    // body check: must be exactly mulf -> addf -> yield
    Block &body = op.getRegion().front();
    auto ops = body.without_terminator();
    if (std::distance(ops.begin(), ops.end()) != 2)
      return failure();

    auto it = ops.begin();
    auto *mulOp = &*it++;
    auto *addOp = &*it;
    if (!isa<arith::MulFOp>(mulOp) || !isa<arith::AddFOp>(addOp))
      return failure();

    // the add must consume the mul result
    if (addOp->getOperand(0) != mulOp->getResult(0))
      return failure();

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};
class MatmulToSystolic
    : public impl::MatmulToSystolicBase<MatmulToSystolic> {
public:
  using impl::MatmulToSystolicBase<MatmulToSystolic>::MatmulToSystolicBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToSystolicRewriter>(&getContext());
    patterns.add<GenericMatmulToSystolicRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

class IcdecCompile : public impl::IcdecCompileBase<IcdecCompile> {
public:
  using impl::IcdecCompileBase<IcdecCompile>::IcdecCompileBase;

  void runOnOperation() final {
    OpPassManager pm("builtin.module");

    // offload matmul to the systolic array driver
    pm.addPass(createMatmulToSystolic());

    // lower everything else to loops
    pm.addPass(createConvertLinalgToLoopsPass());
    pm.addPass(createLowerAffinePass());

    // convert to emitc and clean up
    pm.addPass(createConvertToEmitC());
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::standalone
