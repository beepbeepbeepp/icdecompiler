//===- IcdecPasses.cpp - ICDeCompiler passes --------------------*- C++ -*-===//
//
// Copyright 2026 Indonesia Chip Design Collaborative Center (ICDeC)
// Author: Muhammad Faudzan Abdullah
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"

#include "Icdec/IcdecPasses.h"

namespace mlir::icdec {
#define GEN_PASS_DEF_MATMULTOSYSTOLIC
#define GEN_PASS_DEF_ICDECCOMPILE
#define GEN_PASS_DEF_ICDECCOMPILECPU
#define GEN_PASS_DEF_ELIDECOPYALLOC
#define GEN_PASS_DEF_OUTPARAM
#define GEN_PASS_DEF_ADDINCLUDES
#define GEN_PASS_DEF_EXPANDINTMINMAX
#define GEN_PASS_DEF_GLOBALTOEMITC
#define GEN_PASS_DEF_ALLOCTOARENA
#define GEN_PASS_DEF_CONVTOIM2COL
#define GEN_PASS_DEF_VIEWTOEMITC
#define GEN_PASS_DEF_PRINTSUMMARY
#define GEN_PASS_DEF_REINTERPRETCASTTOINDEX
#include "Icdec/IcdecPasses.h.inc"

namespace {
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
    // a reinterpret_cast only relabels the shape; the underlying pointer is
    // the same, so take the source and skip the cast entirely
    while (auto rc = base.getDefiningOp<memref::ReinterpretCastOp>())
      base = rc.getSource();

    // already a pointer (e.g from memref.alloc): use it directly
    if (isa<emitc::PointerType>(base.getType()))
      return base;

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
  // The driver overwrites its result buffer, but linalg matmul accumulates into
  // its output. Anything already there, a bias for instance, would be lost. So
  // the driver writes into scratch and we add that onto the real output.
  // Write the accumulation into whatever the output collapses from, so the
  // collapse has no remaining user and folds away.
  Value cOut = unwrap(outputs[0]);
  auto cType = dyn_cast<MemRefType>(cOut.getType());
  if (!cType || !cType.hasStaticShape())
    return failure();

  Value scratch = rewriter.create<memref::AllocOp>(
      op->getLoc(), MemRefType::get(cType.getShape(), cType.getElementType()));

  Value scratchPtr = toPointer(scratch);
  if (!scratchPtr)
    return failure();
  args.push_back(scratchPtr);

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

  // out += scratch
  SmallVector<AffineMap> accMaps(2, rewriter.getMultiDimIdentityMap(cType.getRank()));
  SmallVector<utils::IteratorType> accIters(cType.getRank(),
                                            utils::IteratorType::parallel);
  rewriter.create<linalg::GenericOp>(
      op->getLoc(), ValueRange{scratch}, ValueRange{cOut}, accMaps, accIters,
      [](OpBuilder &b, Location loc, ValueRange elems) {
        Value sum = b.create<arith::AddIOp>(loc, elems[0], elems[1]);
        b.create<linalg::YieldOp>(loc, sum);
      });

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

    // work out which loop dimension is the reduction, then read the operand
    // maps to see how M, K and N are laid out
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3)
      return failure();

    unsigned redDim = 0;
    bool foundRed = false;
    for (unsigned i = 0; i < iterTypes.size(); i++)
      if (iterTypes[i] == utils::IteratorType::reduction) {
        if (foundRed)
          return failure();             // more than one reduction
        redDim = i;
        foundRed = true;
      }
    if (!foundRed)
      return failure();

    // the output map tells us which dimension is N: its last result
    AffineMap cMap = maps[2];
    if (cMap.getNumResults() < 1)
      return failure();
    auto nExpr = dyn_cast<AffineDimExpr>(cMap.getResult(cMap.getNumResults() - 1));
    if (!nExpr)
      return failure();
    unsigned nDim = nExpr.getPosition();

    // B must be indexed by exactly N and K; their order tells us whether it is
    // stored [K, N] as the driver expects, or [N, K] transposed
    AffineMap bMap = maps[1];
    if (bMap.getNumResults() != 2)
      return failure();
    auto b0 = dyn_cast<AffineDimExpr>(bMap.getResult(0));
    auto b1 = dyn_cast<AffineDimExpr>(bMap.getResult(1));
    if (!b0 || !b1)
      return failure();

    if (b0.getPosition() == nDim && b1.getPosition() == redDim)
      return failure();                 // transposed B, not handled yet
    if (!(b0.getPosition() == redDim && b1.getPosition() == nDim))
      return failure();                 // some other layout entirely

    // body check: a multiply feeding an add, ignoring any widening casts that
    // integer types need before the multiply
    Block &body = op.getRegion().front();
    Operation *mulOp = nullptr, *addOp = nullptr;
    for (Operation &o : body.without_terminator()) {
      if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(o))
        continue;                       // widening, not the computation
      if (isa<arith::MulFOp, arith::MulIOp>(o)) {
        if (mulOp)
          return failure();             // more than one multiply
        mulOp = &o;
        continue;
      }
      if (isa<arith::AddFOp, arith::AddIOp>(o)) {
        if (addOp)
          return failure();
        addOp = &o;
        continue;
      }
      return failure();                 // anything else is not a plain matmul
    }
    if (!mulOp || !addOp)
      return failure();

    // the add must consume the mul result
    if (addOp->getOperand(0) != mulOp->getResult(0) &&
        addOp->getOperand(1) != mulOp->getResult(0))
      return failure();

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

class BatchMatmulToSystolicRewriter
    : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rewriter) const final {
    // only single-batch is supported: the driver has no batch concept
    auto aType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    if (!aType || aType.getRank() != 3 || aType.getShape()[0] != 1)
      return failure();

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

class MatmulToSystolic
    : public impl::MatmulToSystolicBase<MatmulToSystolic> {
public:
  using impl::MatmulToSystolicBase<MatmulToSystolic>::MatmulToSystolicBase;

  void runOnOperation() final {
    Operation *root = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToSystolicRewriter>(&getContext());
    patterns.add<GenericMatmulToSystolicRewriter>(&getContext());
    patterns.add<BatchMatmulToSystolicRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(root, patternSet)))
      signalPassFailure();

    // Record how much went to the accelerator, for the summary at the end.
    int64_t offloaded = 0;
    root->walk([&](emitc::CallOpaqueOp call) {
      if (call.getCallee() == "systolic_multiply_large")
        offloaded++;
    });
    root->setAttr("icdec.offloaded_ops", OpBuilder(&getContext()).getI64IntegerAttr(offloaded));
  }
};

// Prepends the C headers the emitted code relies on. Existing includes are
// left alone so a model can still pull in its own driver header.
class AddIncludes : public impl::AddIncludesBase<AddIncludes> {
public:
  using impl::AddIncludesBase<AddIncludes>::AddIncludesBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();

    llvm::SmallDenseSet<StringRef> present;
    mod.walk([&](emitc::IncludeOp inc) { present.insert(inc.getInclude()); });

    static constexpr StringRef needed[] = {
        "stdint.h", "stddef.h", "stdbool.h", "stdlib.h"};

    OpBuilder builder(mod.getBodyRegion());

    // The array path calls into the driver, so pull its header in as well.
    // Quoted, not angle-bracketed: it sits next to the generated source.
    // Inserted before the standard headers so they end up above it.
    // libm calls come from convert-math-to-emitc; the driver call from the
    // matmul pass. Each needs its header, and only when actually used.
    static constexpr StringRef mathFns[] = {
        "expf", "exp", "logf", "log", "sqrtf", "sqrt", "tanhf", "tanh",
        "floorf", "floor", "ceilf", "ceil", "roundf", "round", "fabsf", "fabs",
        "cosf", "cos", "sinf", "sin", "powf", "pow"};

    bool usesDriver = false, usesMath = false;
    mod.walk([&](emitc::CallOpaqueOp call) {
      StringRef callee = call.getCallee();
      if (callee == "systolic_multiply_large")
        usesDriver = true;
      if (llvm::is_contained(mathFns, callee))
        usesMath = true;
    });
    if (usesMath && !present.contains("math.h")) {
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<emitc::IncludeOp>(mod.getLoc(),
                                       builder.getStringAttr("math.h"),
                                       /*is_standard_include=*/builder.getUnitAttr());
    }
    if (usesDriver && !present.contains("systolic_drv.h")) {
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<emitc::IncludeOp>(mod.getLoc(),
                                       builder.getStringAttr("systolic_drv.h"),
                                       /*is_standard_include=*/UnitAttr());
    }

    builder.setInsertionPointToStart(mod.getBody());
    for (StringRef hdr : llvm::reverse(ArrayRef<StringRef>(needed))) {
      if (present.contains(hdr))
        continue;
      builder.create<emitc::IncludeOp>(mod.getLoc(),
                                       builder.getStringAttr(hdr),
                                       /*is_standard_include=*/builder.getUnitAttr());
    }
  }
};

// Turns memref return values into trailing output parameters, since C cannot
// return arrays and bare-metal callers supply the output buffer themselves.
class OutParam : public impl::OutParamBase<OutParam> {
public:
  using impl::OutParamBase<OutParam>::OutParamBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();

    mod.walk([&](func::FuncOp fn) {
      if (fn.isExternal())
        return;
      auto resultTypes = fn.getFunctionType().getResults();
      if (resultTypes.empty())
        return;
      // only handle the all-memref case
      for (Type t : resultTypes)
        if (!isa<MemRefType>(t))
          return;

      Block &entry = fn.getBody().front();
      SmallVector<Value> outArgs;
      for (Type t : resultTypes)
        outArgs.push_back(entry.addArgument(t, fn.getLoc()));

      SmallVector<func::ReturnOp> returns;
      fn.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

      OpBuilder builder(fn.getContext());
      for (func::ReturnOp ret : returns) {
        builder.setInsertionPoint(ret);
        for (auto [val, out] : llvm::zip(ret.getOperands(), outArgs))
          if (val != out)
            builder.create<memref::CopyOp>(ret.getLoc(), val, out);
        builder.create<func::ReturnOp>(ret.getLoc());
        ret.erase();
      }

      SmallVector<Type> argTypes(entry.getArgumentTypes());
      fn.setType(builder.getFunctionType(argTypes, {}));
    });
  }
};

// Removes temp buffers that are only written and then copied out wholesale.
// Redirects the writes straight into the copy destination.
class ElideCopyAlloc : public impl::ElideCopyAllocBase<ElideCopyAlloc> {
public:
  using impl::ElideCopyAllocBase<ElideCopyAlloc>::ElideCopyAllocBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    SmallVector<std::pair<memref::AllocOp, memref::CopyOp>> work;

    fn.walk([&](memref::AllocOp alloc) {
      memref::CopyOp theCopy;
      for (Operation *user : alloc->getUsers()) {
        auto copy = dyn_cast<memref::CopyOp>(user);
        if (!copy || copy.getSource() != alloc.getResult())
          continue;
        if (theCopy)
          return;              // more than one copy: leave it alone
        theCopy = copy;
      }
      if (!theCopy)
        return;
      // types must match exactly so the redirect is safe
      if (theCopy.getTarget().getType() != alloc.getType())
        return;
      work.push_back({alloc, theCopy});
    });

    for (auto [alloc, copy] : work) {
      Value dst = copy.getTarget();
      // the destination must not be produced after the alloc
      if (Operation *def = dst.getDefiningOp())
        if (!def->isBeforeInBlock(alloc))
          continue;
      alloc.getResult().replaceAllUsesWith(dst);
      copy.erase();
      alloc.erase();
    }
  }
};

// EmitC cannot lower integer min and max, which quantized models produce from
// clamp. Rewrite them into a comparison plus select, both of which it supports.
class ExpandIntMinMax : public impl::ExpandIntMinMaxBase<ExpandIntMinMax> {
public:
  using impl::ExpandIntMinMaxBase<ExpandIntMinMax>::ExpandIntMinMaxBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<Operation *> work;
    mod.walk([&](Operation *op) {
      if (isa<arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp>(op))
        work.push_back(op);
    });

    for (Operation *op : work) {
      builder.setInsertionPoint(op);
      Value lhs = op->getOperand(0), rhs = op->getOperand(1);

      arith::CmpIPredicate pred;
      if (isa<arith::MaxSIOp>(op))      pred = arith::CmpIPredicate::sgt;
      else if (isa<arith::MinSIOp>(op)) pred = arith::CmpIPredicate::slt;
      else if (isa<arith::MaxUIOp>(op)) pred = arith::CmpIPredicate::ugt;
      else                              pred = arith::CmpIPredicate::ult;

      Value cond = builder.create<arith::CmpIOp>(op->getLoc(), pred, lhs, rhs);
      Value sel = builder.create<arith::SelectOp>(op->getLoc(), cond, lhs, rhs);
      op->getResult(0).replaceAllUsesWith(sel);
      op->erase();
    }
  }
};

// A reinterpret_cast over contiguous memory only relabels the shape, and emitc
// has no way to write that down. Rewrite every access through the cast so it
// addresses the source directly: linearise the index with the cast's strides,
// then split it back into the source's shape.
//
// Copies through a cast are expanded into loops first, so they reduce to the
// same load/store case. Once nothing uses the cast it is erased.
class ReinterpretCastToIndex
    : public impl::ReinterpretCastToIndexBase<ReinterpretCastToIndex> {
public:
    using impl::ReinterpretCastToIndexBase<ReinterpretCastToIndex>::ReinterpretCastToIndexBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();

    SmallVector<memref::ReinterpretCastOp> casts;
    fn.walk([&](memref::ReinterpretCastOp c) { casts.push_back(c); });

    for (memref::ReinterpretCastOp rc : casts) {
      auto srcTy = dyn_cast<MemRefType>(rc.getSource().getType());
      MemRefType dstTy = rc.getType();
      if (!srcTy || !srcTy.hasStaticShape() || !srcTy.getLayout().isIdentity())
        continue;
      if (!dstTy.hasStaticShape())
        continue;

      // only fully static offset and strides can be folded into an index
      ArrayRef<int64_t> offsets = rc.getStaticOffsets();
      ArrayRef<int64_t> strides = rc.getStaticStrides();
      if (offsets.size() != 1 || ShapedType::isDynamic(offsets[0]))
        continue;
      if (llvm::any_of(strides, ShapedType::isDynamic))
        continue;
      int64_t offset = offsets[0];

      // turn copies into explicit loops so only loads and stores remain
      SmallVector<memref::CopyOp> copies;
      for (Operation *user : rc->getUsers())
        if (auto cp = dyn_cast<memref::CopyOp>(user))
          copies.push_back(cp);
      for (memref::CopyOp cp : copies)
        expandCopy(cp);

      bool onlyAccesses = llvm::all_of(rc->getUsers(), [](Operation *u) {
        return isa<memref::LoadOp, memref::StoreOp>(u);
      });
      if (!onlyAccesses)
        continue;                     // something we cannot follow; leave it

      SmallVector<Operation *> users(rc->getUsers().begin(),
                                     rc->getUsers().end());
      for (Operation *user : users) {
        OpBuilder b(user);
        Location loc = user->getLoc();

        if (auto load = dyn_cast<memref::LoadOp>(user)) {
          SmallVector<Value> idx = relabel(b, loc, load.getIndices(), offset,
                                           strides, srcTy);
          Value v = b.create<memref::LoadOp>(loc, rc.getSource(), idx);
          load.getResult().replaceAllUsesWith(v);
          load.erase();
        } else {
          auto store = cast<memref::StoreOp>(user);
          SmallVector<Value> idx = relabel(b, loc, store.getIndices(), offset,
                                           strides, srcTy);
          b.create<memref::StoreOp>(loc, store.getValueToStore(),
                                    rc.getSource(), idx);
          store.erase();
        }
      }

      if (rc->use_empty())
        rc.erase();
    }
  }

private:
  // memref.copy -> nested loops of load and store over the copied shape
  static void expandCopy(memref::CopyOp cp) {
    OpBuilder b(cp);
    Location loc = cp.getLoc();
    auto ty = cast<MemRefType>(cp.getSource().getType());

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> lbs(ty.getRank(), zero), steps(ty.getRank(), one), ubs;
    for (int64_t d : ty.getShape())
      ubs.push_back(b.create<arith::ConstantIndexOp>(loc, d));

    Value src = cp.getSource(), dst = cp.getTarget();
    scf::buildLoopNest(b, loc, lbs, ubs, steps,
                       [&](OpBuilder &nb, Location l, ValueRange iv) {
                         Value v = nb.create<memref::LoadOp>(l, src, iv);
                         nb.create<memref::StoreOp>(l, v, dst, iv);
                       });
    cp.erase();
  }

  // Indices on the cast's shape -> indices on the source's shape.
  static SmallVector<Value> relabel(OpBuilder &b, Location loc,
                                    ValueRange dstIdx, int64_t offset,
                                    ArrayRef<int64_t> dstStrides,
                                    MemRefType srcTy) {
    Value linear = b.create<arith::ConstantIndexOp>(loc, offset);
    for (auto [idx, st] : llvm::zip(dstIdx, dstStrides)) {
      Value s = b.create<arith::ConstantIndexOp>(loc, st);
      linear = b.create<arith::AddIOp>(loc, linear,
                                       b.create<arith::MulIOp>(loc, idx, s));
    }

    ArrayRef<int64_t> shape = srcTy.getShape();
    SmallVector<int64_t> srcStrides(shape.size());
    int64_t acc = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      srcStrides[i] = acc;
      acc *= shape[i];
    }

    SmallVector<Value> out;
    for (size_t j = 0; j < shape.size(); ++j) {
      if (shape[j] == 1) {            // unit dims are always index zero
        out.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
        continue;
      }
      Value q = linear;
      if (srcStrides[j] != 1)
        q = b.create<arith::DivSIOp>(
            loc, q, b.create<arith::ConstantIndexOp>(loc, srcStrides[j]));
      if (j != 0)                     // the outermost index cannot overflow
        q = b.create<arith::RemSIOp>(
            loc, q, b.create<arith::ConstantIndexOp>(loc, shape[j]));
      out.push_back(q);
    }
    return out;
  }
};

// Bare-metal targets have no heap worth using, and the interrupt vector sits at
// a fixed address, so bss has a hard ceiling. Giving every temporary its own
// global does not fit.
//
// Work out when each buffer is live, pack the ones whose lifetimes do not
// overlap into the same bytes of one arena, then rewrite every access into an
// index into that arena. Rewriting the accesses rather than handing out
// pointers keeps everything in terms emitc can already lower.
//
// Buffers handed to the accelerator driver are reached through a pointer to
// their first element. Those join the arena too: the pointer becomes the
// address of their slot, and their lifetime runs until the driver call is done.
//
// Buffers used in ways the rewrite cannot follow still become globals, just
// without sharing, so nothing falls back to malloc.
class AllocToArena : public impl::AllocToArenaBase<AllocToArena> {
public:
  using impl::AllocToArenaBase<AllocToArena>::AllocToArenaBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    Block *entry = &fn.getBody().front();

    struct Slot {
      memref::AllocOp alloc;
      MemRefType type;
      int64_t elems;
      unsigned first;
      unsigned last;
      int64_t offset = -1;      // in elements, not bytes
    };

    DenseMap<Operation *, unsigned> order;
    unsigned n = 0;
    for (Operation &op : *entry)
      order[&op] = n++;

    auto indexOf = [&](Operation *op) -> unsigned {
      while (op && op->getBlock() != entry)
        op = op->getParentOp();
      auto it = order.find(op);
      return it == order.end() ? 0 : it->second;
    };

    // One arena per element type: mixing types would need byte addressing,
    // which puts us back into pointer casts emitc cannot express.
    DenseMap<Type, SmallVector<Slot>> byType;
    for (Operation &op : *entry) {
      auto alloc = dyn_cast<memref::AllocOp>(&op);
      if (!alloc)
        continue;
      auto ty = alloc.getType();
      if (!ty.hasStaticShape() || !alloc.getDynamicSizes().empty())
        continue;
      if (!ty.getLayout().isIdentity())
        continue;

      unsigned first = order[&op];
      unsigned last = first;
      bool followable = true;
      for (Operation *user : alloc.getResult().getUsers()) {
        if (isa<memref::LoadOp, memref::StoreOp, memref::DeallocOp>(user)) {
          last = std::max(last, indexOf(user));
          continue;
        }
        auto rc = dyn_cast<UnrealizedConversionCastOp>(user);
        std::optional<unsigned> end =
            rc ? pointerUseEnd(rc, indexOf) : std::nullopt;
        if (!end) {
          followable = false;         // something the rewrite cannot follow
          break;
        }
        last = std::max(last, *end);
      }
      if (!followable)
        continue;

      byType[ty.getElementType()].push_back(
          {alloc, ty, ty.getNumElements(), first, last});
    }

    OpBuilder builder(fn.getContext());
    auto mod = fn->getParentOfType<ModuleOp>();
    unsigned arenaIdx = 0;
    int64_t totalArenaBytes = 0;

    for (auto &[elemType, slots] : byType) {
      // Place the largest first: it constrains the layout most.
      SmallVector<Slot *> byLargest;
      for (Slot &s : slots)
        byLargest.push_back(&s);
      llvm::stable_sort(byLargest, [](const Slot *a, const Slot *b) {
        return a->elems > b->elems;
      });

      int64_t arenaElems = 0;
      for (Slot *s : byLargest) {
        int64_t candidate = 0;
        bool placed = false;
        while (!placed) {
          placed = true;
          for (Slot *other : byLargest) {
            if (other == s || other->offset < 0)
              continue;
            bool inTime = s->first <= other->last && other->first <= s->last;
            bool inSpace = candidate < other->offset + other->elems &&
                           other->offset < candidate + s->elems;
            if (inTime && inSpace) {
              candidate = other->offset + other->elems;
              placed = false;
              break;
            }
          }
        }
        s->offset = candidate;
        arenaElems = std::max(arenaElems, candidate + s->elems);
      }

      auto arenaTy = MemRefType::get({arenaElems}, elemType);
      std::string name =
          ("__icdec_arena" + Twine(arenaIdx++) + "_" + fn.getName()).str();

      unsigned width = elemType.getIntOrFloatBitWidth();
      totalArenaBytes += arenaElems * (width / 8);

      builder.setInsertionPointToStart(mod.getBody());
      builder.create<memref::GlobalOp>(
          fn.getLoc(), name, builder.getStringAttr("private"), arenaTy,
          /*initial_value=*/Attribute(), /*constant=*/false,
          /*alignment=*/builder.getI64IntegerAttr(8));

      builder.setInsertionPointToStart(entry);
      Value arena =
          builder.create<memref::GetGlobalOp>(fn.getLoc(), arenaTy, name);

      for (Slot &s : slots) {
        // row-major strides for this buffer's shape
        SmallVector<int64_t> strides(s.type.getRank());
        int64_t acc = 1;
        for (int i = s.type.getRank() - 1; i >= 0; --i) {
          strides[i] = acc;
          acc *= s.type.getShape()[i];
        }

        SmallVector<Operation *> users(s.alloc.getResult().getUsers().begin(),
                                       s.alloc.getResult().getUsers().end());
        for (Operation *user : users) {
          if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
            dealloc.erase();
            continue;
          }
          if (auto rc = dyn_cast<UnrealizedConversionCastOp>(user)) {
            rewritePointer(builder, rc, arena, arenaElems, elemType, s.offset,
                           strides);
            continue;
          }

          builder.setInsertionPoint(user);
          Location loc = user->getLoc();

          ValueRange indices = isa<memref::LoadOp>(user)
                                   ? cast<memref::LoadOp>(user).getIndices()
                                   : cast<memref::StoreOp>(user).getIndices();

          // linear index = offset + sum(index[i] * stride[i])
          Value linear = builder.create<arith::ConstantIndexOp>(loc, s.offset);
          for (auto [i, idx] : llvm::enumerate(indices)) {
            Value st = builder.create<arith::ConstantIndexOp>(loc, strides[i]);
            Value term = builder.create<arith::MulIOp>(loc, idx, st);
            linear = builder.create<arith::AddIOp>(loc, linear, term);
          }

          if (auto load = dyn_cast<memref::LoadOp>(user)) {
            Value v = builder.create<memref::LoadOp>(loc, arena,
                                                     ValueRange{linear});
            load.getResult().replaceAllUsesWith(v);
            load.erase();
          } else {
            auto store = cast<memref::StoreOp>(user);
            builder.create<memref::StoreOp>(loc, store.getValueToStore(), arena,
                                            ValueRange{linear});
            store.erase();
          }
        }

        s.alloc.erase();
      }
    }

    fn->setAttr("icdec.arena_bytes",
                builder.getI64IntegerAttr(totalArenaBytes));

    // Anything the arena could not take still needs to stay off the heap.
    SmallVector<memref::AllocOp> leftovers;
    fn.walk([&](memref::AllocOp a) {
      if (a.getType().hasStaticShape() && a.getDynamicSizes().empty())
        leftovers.push_back(a);
    });

    unsigned spare = 0;
    for (memref::AllocOp alloc : leftovers) {
      auto ty = alloc.getType();
      std::string name =
          ("__icdec_buf" + Twine(spare++) + "_" + fn.getName()).str();

      builder.setInsertionPointToStart(mod.getBody());
      builder.create<memref::GlobalOp>(
          alloc.getLoc(), name, builder.getStringAttr("private"), ty,
          /*initial_value=*/Attribute(), /*constant=*/false,
          /*alignment=*/alloc.getAlignmentAttr());

      builder.setInsertionPoint(alloc);
      Value repl = builder.create<memref::GetGlobalOp>(alloc.getLoc(), ty, name);
      alloc.getResult().replaceAllUsesWith(repl);
      alloc.erase();
    }

    SmallVector<memref::DeallocOp> deadDeallocs;
    fn.walk([&](memref::DeallocOp d) {
      if (d.getMemref().getDefiningOp<memref::GetGlobalOp>())
        deadDeallocs.push_back(d);
    });
    for (memref::DeallocOp d : deadDeallocs)
      d.erase();
  }

private:
  static std::optional<int64_t> constantIndex(Value v) {
    if (auto c = v.getDefiningOp<emitc::ConstantOp>())
      if (auto a = dyn_cast<IntegerAttr>(c.getValue()))
        return a.getInt();
    if (auto c = v.getDefiningOp<arith::ConstantOp>())
      if (auto a = dyn_cast<IntegerAttr>(c.getValue()))
        return a.getInt();
    return std::nullopt;
  }

  // A buffer handed to the driver is reached as
  //   cast(memref -> emitc.array) -> subscript[constants] -> address_of
  // and the pointer is passed to the call. Returns where the last use of that
  // pointer is, or nothing if the chain has any other shape.
  static std::optional<unsigned>
  pointerUseEnd(UnrealizedConversionCastOp rc,
                llvm::function_ref<unsigned(Operation *)> indexOf) {
    if (rc->getNumResults() != 1 ||
        !isa<emitc::ArrayType>(rc->getResult(0).getType()))
      return std::nullopt;

    unsigned end = indexOf(rc);
    for (Operation *s : rc->getResult(0).getUsers()) {
      auto sub = dyn_cast<emitc::SubscriptOp>(s);
      if (!sub)
        return std::nullopt;
      for (Value idx : sub.getIndices())
        if (!constantIndex(idx))
          return std::nullopt;
      for (Operation *a : sub->getResult(0).getUsers()) {
        auto addr = dyn_cast<emitc::AddressOfOp>(a);
        if (!addr)
          return std::nullopt;
        // +1 keeps the buffer live through the wait that follows the call
        for (Operation *u : addr->getResult(0).getUsers())
          end = std::max(end, indexOf(u) + 1);
      }
    }
    return end;
  }

  // Point the driver at the buffer's slot in the arena instead.
  static void rewritePointer(OpBuilder &b, UnrealizedConversionCastOp rc,
                             Value arena, int64_t arenaElems, Type elemTy,
                             int64_t offset, ArrayRef<int64_t> strides) {
    auto arrTy = emitc::ArrayType::get({arenaElems}, elemTy);
    auto ptrTy = emitc::PointerType::get(elemTy);

    SmallVector<emitc::SubscriptOp> subs;
    for (Operation *u : rc->getResult(0).getUsers())
      subs.push_back(llvm::cast<emitc::SubscriptOp>(u));

    for (emitc::SubscriptOp sub : subs) {
      int64_t linear = offset;
      for (auto [i, idx] : llvm::enumerate(sub.getIndices()))
        linear += *constantIndex(idx) * strides[i];

      SmallVector<emitc::AddressOfOp> addrs;
      for (Operation *u : sub->getResult(0).getUsers())
        addrs.push_back(llvm::cast<emitc::AddressOfOp>(u));

      for (emitc::AddressOfOp addr : addrs) {
        b.setInsertionPoint(addr);
        Location loc = addr.getLoc();
        Value arr =
            b.create<UnrealizedConversionCastOp>(loc, arrTy, arena).getResult(0);
        Value idx = b.create<emitc::ConstantOp>(loc, b.getIndexType(),
                                                b.getIndexAttr(linear));
        Value elem = b.create<emitc::SubscriptOp>(
            loc, llvm::cast<TypedValue<emitc::ArrayType>>(arr), ValueRange{idx});
        Value ptr = b.create<emitc::AddressOfOp>(loc, ptrTy, elem);
        addr->getResult(0).replaceAllUsesWith(ptr);
        addr.erase();
      }
      sub.erase();
    }
    rc.erase();
  }
};

// The bundled emitc conversion has no pattern for memref.view, which the arena
// allocator produces. A view is a typed pointer into the arena.
//
// The conversion recognises a memref backed by an emitc.ptr through an
// unrealized cast, and computes element addresses linearly from it. So taking
// the address of the byte at the offset and casting that pointer to the buffer
// type is enough; no array type is involved.
class ViewToEmitC : public impl::ViewToEmitCBase<ViewToEmitC> {
public:
  using impl::ViewToEmitCBase<ViewToEmitC>::ViewToEmitCBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<memref::ViewOp> views;
    mod.walk([&](memref::ViewOp v) { views.push_back(v); });

    for (memref::ViewOp view : views) {
      auto dstTy = dyn_cast<MemRefType>(view.getType());
      auto srcTy = dyn_cast<MemRefType>(view.getSource().getType());
      if (!dstTy || !srcTy || !dstTy.hasStaticShape())
        continue;

      auto offsetOp = view.getByteShift().getDefiningOp<arith::ConstantIndexOp>();
      if (!offsetOp)
        continue;                     // only constant offsets are handled

      builder.setInsertionPoint(view);
      Location loc = view.getLoc();

      // the arena, as an emitc array
      auto srcArrTy = emitc::ArrayType::get(srcTy.getShape(),
                                            srcTy.getElementType());
      Value arena = builder.create<UnrealizedConversionCastOp>(
                        loc, srcArrTy, view.getSource()).getResult(0);

      // &arena[offset], a pointer to the arena's element type
      Value idx = builder.create<emitc::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(offsetOp.value()));
      Value elem = builder.create<emitc::SubscriptOp>(
          loc, cast<TypedValue<emitc::ArrayType>>(arena), ValueRange{idx});
      auto srcPtrTy = emitc::PointerType::get(srcTy.getElementType());
      Value bytePtr = builder.create<emitc::AddressOfOp>(loc, srcPtrTy, elem);

      // reinterpret those bytes as the buffer's element type
      auto dstPtrTy = emitc::PointerType::get(dstTy.getElementType());
      Value typedPtr = bytePtr;
      if (srcTy.getElementType() != dstTy.getElementType())
        typedPtr = builder.create<emitc::CastOp>(loc, dstPtrTy, bytePtr);

      // hand it back as a memref; the conversion will index it linearly
      Value repl = builder.create<UnrealizedConversionCastOp>(
                       loc, dstTy, typedPtr).getResult(0);

      view.getResult().replaceAllUsesWith(repl);
      view.erase();
    }
  }
};

// The bundled emitc conversion rewrites memref.get_global but leaves the
// memref.global declaration behind. Convert those declarations so the symbols
// resolve again.
class GlobalToEmitC : public impl::GlobalToEmitCBase<GlobalToEmitC> {
public:
  using impl::GlobalToEmitCBase<GlobalToEmitC>::GlobalToEmitCBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    // rewrite the readers first, so the declarations are unused when replaced
    SmallVector<memref::GetGlobalOp> readers;
    mod.walk([&](memref::GetGlobalOp g) { readers.push_back(g); });

    for (memref::GetGlobalOp g : readers) {
      auto memrefTy = dyn_cast<MemRefType>(g.getType());
      if (!memrefTy || !memrefTy.hasStaticShape())
        continue;
      auto arrayTy = emitc::ArrayType::get(memrefTy.getShape(),
                                           memrefTy.getElementType());
      builder.setInsertionPoint(g);
      Value repl = builder.create<emitc::GetGlobalOp>(g.getLoc(), arrayTy,
                                                      g.getNameAttr());
      Value cast = builder.create<UnrealizedConversionCastOp>(
                       g.getLoc(), memrefTy, repl).getResult(0);
      g.getResult().replaceAllUsesWith(cast);
      g.erase();
    }

    SmallVector<memref::GlobalOp> decls;
    mod.walk([&](memref::GlobalOp g) { decls.push_back(g); });

    for (memref::GlobalOp g : decls) {
      auto memrefTy = g.getType();
      if (!memrefTy.hasStaticShape())
        continue;
      auto arrayTy = emitc::ArrayType::get(memrefTy.getShape(),
                                           memrefTy.getElementType());
      builder.setInsertionPoint(g);
      builder.create<emitc::GlobalOp>(
          g.getLoc(), g.getSymName(), arrayTy, g.getInitialValueAttr(),
          /*extern_specifier=*/false,
          /*static_specifier=*/true,
          /*const_specifier=*/g.getConstant());
      g.erase();
    }
  }
};

// A convolution is a matrix multiply once the input windows are laid out as
// rows. Rewriting it that way lets the matmul offload pass send the bulk of
// the work to the accelerator.
class ConvToIm2Col : public impl::ConvToIm2ColBase<ConvToIm2Col> {
public:
  using impl::ConvToIm2ColBase<ConvToIm2Col>::ConvToIm2ColBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    IRRewriter rewriter(fn.getContext());

    SmallVector<linalg::Conv2DNhwcHwcfOp> convs;
    fn.walk([&](linalg::Conv2DNhwcHwcfOp op) { convs.push_back(op); });

    for (linalg::Conv2DNhwcHwcfOp op : convs) {
      rewriter.setInsertionPoint(op);
      FailureOr<std::pair<Operation *, Operation *>> result =
          linalg::rewriteInIm2Col(rewriter, op);
      if (failed(result))
        continue;                 // leave it for the CPU path
      // rewriteInIm2Col already replaced the uses; nothing else to do
    }
  }
};

// Report what the compiler did, in the spirit of what Vela prints: how much of
// the multiply-accumulate work went to the accelerator, and how much memory
// the result needs.
//
// Only MACs are counted. Pooling, clamping and requantisation run on the CPU
// too, but they are element-wise and cheap next to the matrix work, so folding
// them into the same number would misrepresent where the time goes.
class PrintSummary : public impl::PrintSummaryBase<PrintSummary> {
public:
  using impl::PrintSummaryBase<PrintSummary>::PrintSummaryBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();

    int64_t accelCalls = 0, accelMacs = 0;
    int64_t cpuMacs = 0;
    int64_t arenaBytes = 0, weightBytes = 0;

    // Accelerator calls carry M, K and N as constants, so the MAC count is
    // just their product.
    mod.walk([&](emitc::CallOpaqueOp call) {
      if (call.getCallee() != "systolic_multiply_large")
        return;
      accelCalls++;

      int64_t dims[3];
      for (unsigned i = 0; i < 3 && i < call.getNumOperands(); i++) {
        auto c = call.getOperand(i).getDefiningOp<emitc::ConstantOp>();
        if (!c)
          return;
        auto attr = dyn_cast<IntegerAttr>(c.getValue());
        if (!attr)
          return;
        dims[i] = attr.getInt();
      }
      accelMacs += dims[0] * dims[1] * dims[2];
    });

    // A multiply inside a loop nest runs once per iteration of every loop
    // around it. Loops with non-constant bounds are skipped rather than
    // guessed at.
    mod.walk([&](Operation *op) {
      if (!isa<emitc::MulOp>(op))
        return;

      // Only multiply-accumulates on model data count: a multiply of at most
      // 32 bits, or a float one, whose result feeds an add. Index arithmetic
      // is size_t, and requantisation multiplies in 64 bits, so neither is
      // model work.
      Type t = op->getResult(0).getType();
      bool onData = isa<FloatType>(t) ||
                    (isa<IntegerType>(t) && cast<IntegerType>(t).getWidth() <= 32);
      // Signed arithmetic is done in unsigned types to keep overflow defined in
      // C, so the product may pass through a cast or two before the add.
      std::function<bool(Value, int)> feedsAdd = [&](Value v, int depth) {
        for (Operation *u : v.getUsers()) {
          if (isa<emitc::AddOp>(u))
            return true;
          if (depth > 0 && isa<emitc::CastOp>(u) &&
              feedsAdd(u->getResult(0), depth - 1))
            return true;
        }
        return false;
      };
      bool accumulates = feedsAdd(op->getResult(0), 3);
      if (!onData || !accumulates)
        return;

      int64_t trips = 1;
      for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
        auto loop = dyn_cast<emitc::ForOp>(p);
        if (!loop)
          continue;
        auto lo = loop.getLowerBound().getDefiningOp<emitc::ConstantOp>();
        auto hi = loop.getUpperBound().getDefiningOp<emitc::ConstantOp>();
        auto st = loop.getStep().getDefiningOp<emitc::ConstantOp>();
        if (!lo || !hi || !st)
          return;                       // bound we cannot evaluate
        auto loA = dyn_cast<IntegerAttr>(lo.getValue());
        auto hiA = dyn_cast<IntegerAttr>(hi.getValue());
        auto stA = dyn_cast<IntegerAttr>(st.getValue());
        if (!loA || !hiA || !stA || stA.getInt() == 0)
          return;
        int64_t count = (hiA.getInt() - loA.getInt() + stA.getInt() - 1) /
                        stA.getInt();
        trips *= (count > 0 ? count : 0);
      }
      cpuMacs += trips;
    });

    mod.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<IntegerAttr>("icdec.arena_bytes"))
        arenaBytes += attr.getInt();
    });

    mod.walk([&](emitc::GlobalOp g) {
      if (!g.getConstSpecifier())
        return;
      auto arrTy = dyn_cast<emitc::ArrayType>(g.getType());
      if (!arrTy)
        return;
      int64_t elems = 1;
      for (int64_t d : arrTy.getShape())
        elems *= d;
      weightBytes += elems * (arrTy.getElementType().getIntOrFloatBitWidth() / 8);
    });

    int64_t totalMacs = accelMacs + cpuMacs;

    // Only the first and last parameters are interesting: the input tensor and
    // the output buffer. Everything between them is weights.
    std::string inShape, outShape;
    mod.walk([&](emitc::FuncOp fn) {
      if (!inShape.empty())
        return;
      auto inputs = fn.getFunctionType().getInputs();
      if (inputs.empty())
        return;

      auto describe = [](Type t) {
        std::string s;
        llvm::raw_string_ostream os(s);
        if (auto arr = dyn_cast<emitc::ArrayType>(t)) {
          llvm::interleave(arr.getShape(), os, "x");
          os << " " << arr.getElementType();
        } else {
          os << t;
        }
        return s;
      };

      inShape = describe(inputs.front());
      outShape = describe(inputs.back());
    });

    llvm::errs() << "\nICDeCompiler summary\n";
    if (!inShape.empty()) {
      llvm::errs() << "  input             : " << inShape << "\n";
      llvm::errs() << "  output            : " << outShape << "\n";
    }
    llvm::errs() << "  MACs total        : " << totalMacs << "\n";
    if (totalMacs > 0) {
      llvm::errs() << "  on accelerator    : " << accelMacs
                   << "  (" << (accelMacs * 100 / totalMacs) << "%)"
                   << " in " << accelCalls << " calls\n";
      llvm::errs() << "  on CPU            : " << cpuMacs
                   << "  (" << (cpuMacs * 100 / totalMacs) << "%)\n";
    }
    llvm::errs() << "  scratch memory    : " << arenaBytes << " bytes\n";
    if (weightBytes)
      llvm::errs() << "  constants in code : " << weightBytes << " bytes\n";
    llvm::errs() << "\n";
  }
};

// Shared pipeline. When `offloadMatmul` is false, everything runs on the CPU.
static void buildIcdecPipeline(OpPassManager &pm, bool offloadMatmul) {
  {
    // HWCF kernel layout turns the im2col result into a plain matmul instead
    // of a row-wise dot product, which is the orientation the driver expects.
    TosaToLinalgNamedOptions namedOpts;
    namedOpts.preferConv2DKernelLayoutHWCF = true;
    pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalgNamed(namedOpts));
  }
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  if (offloadMatmul)
    pm.addNestedPass<func::FuncOp>(createConvToIm2Col());
  pm.addNestedPass<func::FuncOp>(createTosaToTensorPass());
  {
    TosaToArithPassOptions opts;
    opts.includeApplyRescale = true;
    pm.addNestedPass<func::FuncOp>(createTosaToArithPass(opts));
  }
  pm.addPass(createExpandIntMinMax());
  pm.addPass(createCanonicalizerPass());

  // tensor -> memref
  pm.addPass(bufferization::createEmptyTensorEliminationPass());
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createOutParam());

  if (offloadMatmul) {
    pm.addPass(createMatmulToSystolic());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(memref::createElideReinterpretCastPass());
    pm.addNestedPass<func::FuncOp>(createElideCopyAlloc());
    pm.addNestedPass<func::FuncOp>(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(memref::createElideReinterpretCastPass());
  }

  // turn subviews from padding into reinterpret_casts, then drop them
  pm.addNestedPass<func::FuncOp>(memref::createExpandStridedMetadataPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(memref::createElideReinterpretCastPass());

  // lower everything else to loops
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  pm.addPass(createExpandIntMinMax());

  // second cleanup round: casts that only appear after loop lowering
  pm.addPass(memref::createElideReinterpretCastPass());
  pm.addNestedPass<func::FuncOp>(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createElideCopyAlloc());

  // Casts left over from reshapes become plain indexed accesses, so the
  // buffers behind them can join the arena like any other.
  pm.addNestedPass<func::FuncOp>(createReinterpretCastToIndex());
  pm.addPass(createCanonicalizerPass());

  // Buffers become arena slices only once the loops exist, so every access is
  // a plain load or store the rewrite can follow.
  pm.addNestedPass<func::FuncOp>(createAllocToArena());
  pm.addPass(createCanonicalizerPass());

  // convert to emitc and clean up
  pm.addPass(createGlobalToEmitC());
  pm.addPass(createConvertMathToEmitC());
  pm.addPass(createConvertToEmitC());
  pm.addPass(createExpandIntMinMax());
  pm.addNestedPass<emitc::FuncOp>(createConvertArithToEmitC());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createAddIncludes());
  pm.addPass(createPrintSummary());
}

class IcdecCompile : public impl::IcdecCompileBase<IcdecCompile> {
public:
  using impl::IcdecCompileBase<IcdecCompile>::IcdecCompileBase;

  void runOnOperation() final {
    OpPassManager pm("builtin.module");
    buildIcdecPipeline(pm, /*offloadMatmul=*/true);
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

class IcdecCompileCpu : public impl::IcdecCompileCpuBase<IcdecCompileCpu> {
public:
  using impl::IcdecCompileCpuBase<IcdecCompileCpu>::IcdecCompileCpuBase;

  void runOnOperation() final {
    OpPassManager pm("builtin.module");
    buildIcdecPipeline(pm, /*offloadMatmul=*/false);
    if (failed(runPipeline(pm, getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::icdec
