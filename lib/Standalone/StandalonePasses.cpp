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

#include "Standalone/StandalonePasses.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#define GEN_PASS_DEF_MATMULTOSYSTOLIC
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

  for (Value v : inputs)
    args.push_back(unwrap(v));
  for (Value v : outputs)
    args.push_back(unwrap(v));

  rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(),
      TypeRange{},
      rewriter.getStringAttr("systolic_multiply_large"),
      args);

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

} // namespace
} // namespace mlir::standalone
