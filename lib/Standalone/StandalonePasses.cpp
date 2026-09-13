//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
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

#include "Standalone/StandalonePasses.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#define GEN_PASS_DEF_MATMULTOSYSTOLIC
#define GEN_PASS_DEF_ICDECCOMPILE
#define GEN_PASS_DEF_ICDECCOMPILECPU
#define GEN_PASS_DEF_ELIDECOPYALLOC
#define GEN_PASS_DEF_OUTPARAM
#define GEN_PASS_DEF_ADDINCLUDES
#define GEN_PASS_DEF_EXPANDINTMINMAX
#define GEN_PASS_DEF_GLOBALTOEMITC
#define GEN_PASS_DEF_ALLOCTOGLOBAL
#define GEN_PASS_DEF_CONVTOIM2COL
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
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToSystolicRewriter>(&getContext());
    patterns.add<GenericMatmulToSystolicRewriter>(&getContext());
    patterns.add<BatchMatmulToSystolicRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
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
    bool usesDriver = false;
    mod.walk([&](emitc::CallOpaqueOp call) {
      if (call.getCallee() == "systolic_multiply_large")
        usesDriver = true;
    });
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

// Bare-metal targets have no usable heap, and the bundled emitc conversion
// cannot express the pointer-to-array cast that memref.alloc produces.
// Replacing each statically shaped allocation with a module-level global
// sidesteps both: the buffer lands in .bss and no allocator is needed.
class AllocToGlobal : public impl::AllocToGlobalBase<AllocToGlobal> {
public:
  using impl::AllocToGlobalBase<AllocToGlobal>::AllocToGlobalBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    OpBuilder builder(mod.getContext());

    SmallVector<memref::AllocOp> work;
    mod.walk([&](memref::AllocOp a) {
      if (a.getType().hasStaticShape() && a.getDynamicSizes().empty())
        work.push_back(a);
    });

    unsigned counter = 0;
    for (memref::AllocOp alloc : work) {
      auto ty = alloc.getType();

      std::string name = ("__icdec_buf" + Twine(counter++)).str();
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<memref::GlobalOp>(
          alloc.getLoc(),
          name,
          builder.getStringAttr("private"),
          ty,
          /*initial_value=*/Attribute(),
          /*constant=*/false,
          /*alignment=*/alloc.getAlignmentAttr());

      builder.setInsertionPoint(alloc);
      Value repl = builder.create<memref::GetGlobalOp>(alloc.getLoc(), ty, name);
      alloc.getResult().replaceAllUsesWith(repl);
      alloc.erase();
    }

    // any dealloc of those buffers is now meaningless
    SmallVector<memref::DeallocOp> deallocs;
    mod.walk([&](memref::DeallocOp d) {
      if (d.getMemref().getDefiningOp<memref::GetGlobalOp>())
        deallocs.push_back(d);
    });
    for (memref::DeallocOp d : deallocs)
      d.erase();
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
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::IdentityLayoutMap;
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

  // convert to emitc and clean up
  pm.addPass(createAllocToGlobal());
  pm.addPass(createGlobalToEmitC());
  pm.addPass(createConvertToEmitC());
  pm.addPass(createExpandIntMinMax());
  pm.addNestedPass<emitc::FuncOp>(createConvertArithToEmitC());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createAddIncludes());
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
} // namespace mlir::standalone
