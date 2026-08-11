//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

class MatmulToSystolicRewriter: public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const final{
    // temporary: just marking that the match was found; haven't rewritten it yet
    
    // collect operands: A, B (ins) and C (outs)

    // step 1: prove the rewrite mechanism works, no operands yet
    // memref is not a valid emitC operand type; conversion comes next
    rewriter.create<emitc::CallOpaqueOp>(
      op.getLoc(),
      TypeRange{},
      rewriter.getStringAttr("systolic_multiply_large"),
      ValueRange{});
    
    // erase the origina linalg.matmul
    rewriter.eraseOp(op);
    return success();
  }
};

class MatmulToSystolic
    : public impl::MatmulToSystolicBase<MatmulToSystolic> {
public:
  using impl::MatmulToSystolicBase<MatmulToSystolic>::MatmulToSystolicBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToSystolicRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::standalone
