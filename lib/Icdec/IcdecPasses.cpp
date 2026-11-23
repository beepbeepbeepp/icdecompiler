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
#define GEN_PASS_DEF_RETURNSTATUS
#define GEN_PASS_DEF_ADDINCLUDES
#define GEN_PASS_DEF_EXPANDINTMINMAX
#define GEN_PASS_DEF_GLOBALTOEMITC
#define GEN_PASS_DEF_ALLOCTOARENA
#define GEN_PASS_DEF_CONVTOIM2COL
#define GEN_PASS_DEF_VIEWTOEMITC
#define GEN_PASS_DEF_PRINTSUMMARY
#define GEN_PASS_DEF_REINTERPRETCASTTOINDEX
#define GEN_PASS_DEF_VALIDATEINPUT
#include "Icdec/IcdecPasses.h.inc"

namespace {
// Largest inner dimension K the current bitstream takes in one call. Anything
// wider stays on the CPU.
static constexpr int64_t kSystolicMaxK = 128;

// Driver entry point. It returns 0 once the result is in place, or a negative
// code when the job failed.
static constexpr StringLiteral kSystolicCallee = "systolic_multiply_large_hwpe";

// Set on functions that return a status code, see ReturnStatus.
static constexpr StringLiteral kReturnsStatusAttr = "icdec.returns_status";

// The driver is (int16_t *A, int16_t *B, int32_t *Res). Any other element
// type would be read or written at the wrong width.
static bool hasSystolicElementTypes(Value a, Value b, Value res) {
  auto isInt = [](Value v, unsigned width) {
    auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(v.getType()));
    return intTy && intTy.getWidth() == width;
  };
  return isInt(a, 16) && isInt(b, 16) && isInt(res, 32);
}

// The driver multiplies one M x K matrix by one K x N matrix. Leading axes are
// batch, and only a batch of one is the same thing as a plain matmul.
static bool hasUnitBatch(Value v) {
  auto type = dyn_cast<ShapedType>(v.getType());
  if (!type || !type.hasRank())
    return false;
  if (type.getRank() < 2)
    return true; // a vector has no batch axis
  return llvm::all_of(type.getShape().drop_back(2),
                      [](int64_t d) { return d == 1; });
}

// True if the cast starts at the first element of its source and keeps the
// elements next to each other in row-major order, so that it changes the shape
// and nothing else. A cast with an offset or with wider strides is a slice.
static bool relabelsShapeOnly(memref::ReinterpretCastOp rc) {
  ArrayRef<int64_t> offsets = rc.getStaticOffsets();
  ArrayRef<int64_t> sizes = rc.getStaticSizes();
  ArrayRef<int64_t> strides = rc.getStaticStrides();
  if (offsets.size() != 1 || offsets[0] != 0 || sizes.size() != strides.size())
    return false;
  int64_t contiguous = 1;
  for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(sizes[i]) || strides[i] != contiguous)
      return false;
    contiguous *= sizes[i];
  }
  return true;
}

// True if the body multiplies and adds the product onto something, inside a
// reduction loop. That is the multiply-accumulate work the accelerator is
// there for, and the work the summary counts.
static bool multipliesAndAccumulates(linalg::LinalgOp op) {
  if (op.getNumReductionLoops() == 0)
    return false;
  for (Operation &inner : *op.getBlock()) {
    if (!isa<arith::MulIOp, arith::MulFOp>(inner))
      continue;
    for (Operation *user : inner.getResult(0).getUsers())
      if (isa<arith::AddIOp, arith::AddFOp>(user))
        return true;
  }
  return false;
}

// Walks back through casts and shape-only ops to the value they start from.
static Value throughShapeOps(Value v) {
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
}

// The buffer behind an operand, or null if it cannot become a C pointer.
//
// The driver reads M x K elements in a row from the pointer it is given. An
// operand that is a slice of a larger buffer, with an offset or with rows
// further apart than K, does not look like that in memory, so it gets no
// pointer and the op stays on the CPU, where the loops index it correctly.
static Value bufferBehind(Value v) {
  Value base = throughShapeOps(v);
  // a reinterpret_cast that only relabels the shape shares its source's
  // pointer, so take the source and skip the cast
  while (auto rc = base.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (!relabelsShapeOnly(rc))
      return Value();
    base = rc.getSource();
  }

  // already a pointer (e.g from memref.alloc) or a C array
  if (isa<emitc::PointerType, emitc::ArrayType>(base.getType()))
    return base;
  // a memref becomes an equivalent emitc.array, which needs a static shape
  // and the plain row-major layout; a subview has a strided layout
  if (auto memrefType = dyn_cast<MemRefType>(base.getType()))
    if (memrefType.hasStaticShape() && memrefType.getLayout().isIdentity())
      return base;
  return Value();
}

// What the driver call needs, collected while checking that an op fits.
struct SystolicOperands {
  int64_t M = 0, K = 0, N = 0;
  Value a, b; // the buffers behind the two inputs
  Value out;  // the buffer the result is added onto
};

static constexpr StringLiteral kNotSystolicTypes =
    "matmul is not i16 x i16 -> i32, the only element types the accelerator "
    "takes";

// Decides whether a matmul with these operands can go to the accelerator.
// Returns an empty string and fills in `found` if it can. Otherwise returns
// the reason, worded so that "; it stays on the CPU" can follow it.
//
// This only looks at the IR and creates nothing. The rewrite calls it before
// it builds anything, and the pass calls it again on whatever is left, to tell
// the user why. One function for both keeps the warning and the decision from
// drifting apart.
static std::string whyNotOnSystolic(ValueRange inputs, ValueRange outputs,
                                    SystolicOperands &found) {
  auto aType = dyn_cast<MemRefType>(inputs[0].getType());
  auto bType = dyn_cast<MemRefType>(inputs[1].getType());
  if (!aType || !bType || !aType.hasStaticShape() || !bType.hasStaticShape())
    return "matmul has an operand that is not a buffer with a fixed shape";

  if (!hasSystolicElementTypes(inputs[0], inputs[1], outputs[0]))
    return kNotSystolicTypes.str();

  // A batch larger than one would need one driver call per batch. That is not
  // done yet.
  if (!hasUnitBatch(inputs[0]) || !hasUnitBatch(inputs[1]) ||
      !hasUnitBatch(outputs[0]))
    return "matmul has a batch larger than one, which the accelerator does "
           "not take";

  // matmul dims live in the last two axes; leading axes are batch
  if (aType.getRank() < 2 || bType.getRank() < 2)
    return "matmul has an operand that is not a matrix";
  auto aShape = aType.getShape();
  auto bShape = bType.getShape();
  found.M = aShape[aType.getRank() - 2];
  found.K = aShape[aType.getRank() - 1];
  found.N = bShape[bType.getRank() - 1];

  // The driver tiles internally and pads partial tiles, so any static M and N
  // work. K is bounded by the bitstream.
  if (found.K > kSystolicMaxK)
    return ("matmul with K = " + Twine(found.K) +
            " exceeds the accelerator limit of " + Twine(kSystolicMaxK))
        .str();

  found.a = bufferBehind(inputs[0]);
  found.b = bufferBehind(inputs[1]);
  if (!found.a || !found.b)
    return "matmul has an operand that is a slice of a larger buffer, which "
           "the driver cannot read";

  // The driver overwrites its result buffer, but linalg matmul accumulates into
  // its output. Anything already there, a bias for instance, would be lost. So
  // the driver writes into scratch and we add that onto the real output.
  // Write the accumulation into whatever the output collapses from, so the
  // collapse has no remaining user and folds away.
  found.out = throughShapeOps(outputs[0]);
  auto cType = dyn_cast<MemRefType>(found.out.getType());
  if (!cType || !cType.hasStaticShape())
    return "matmul has an output that is not a buffer with a fixed shape";

  return "";
}

// Shared rewrite: replace a matmul-like op with a systolic driver call, or
// fail if whyNotOnSystolic has a reason. Every check runs before the first op
// is created: a pattern that fails must leave the IR as it found it.
static LogicalResult rewriteToSystolicCall(Operation *op, ValueRange inputs, ValueRange outputs, PatternRewriter &rewriter) {
  SystolicOperands found;
  std::string why = whyNotOnSystolic(inputs, outputs, found);
  if (!why.empty())
    return rewriter.notifyMatchFailure(op, why);
  int64_t M = found.M, K = found.K, N = found.N;
  Value aBuffer = found.a, bBuffer = found.b, cOut = found.out;
  auto cType = cast<MemRefType>(cOut.getType());

  // Every check has passed. Nothing below this line can fail.

  // materialize M, K, N as i32 constants
  auto i32Type = rewriter.getI32Type();
  SmallVector<Value> args;
  for (int64_t dim : {M, K, N}) {
    auto attr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));
    args.push_back(rewriter.create<emitc::ConstantOp>(
        op->getLoc(), i32Type, attr));
  }

  // convert a buffer from bufferOf to a flat C pointer: &arr[0]...[0]
  auto toPointer = [&](Value base) -> Value {
    // already a pointer: use it directly
    if (isa<emitc::PointerType>(base.getType()))
      return base;

    // if still a memref, cast it to an equivalent emitc.array first
    if (auto memrefType = dyn_cast<MemRefType>(base.getType())) {
      auto arrTy = emitc::ArrayType::get(memrefType.getShape(), memrefType.getElementType());
      base = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), arrTy, base).getResult(0);
    }

    auto arrType = cast<emitc::ArrayType>(base.getType());

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

  args.push_back(toPointer(aBuffer));
  args.push_back(toPointer(bBuffer));

  Value scratch = rewriter.create<memref::AllocOp>(
      op->getLoc(), MemRefType::get(cType.getShape(), cType.getElementType()));
  args.push_back(toPointer(scratch));

  // The driver waits for the array itself, so the result is ready on return.
  Value status = rewriter
                     .create<emitc::CallOpaqueOp>(
                         op->getLoc(), TypeRange{i32Type},
                         rewriter.getStringAttr(kSystolicCallee), args)
                     .getResult(0);

  // A failed job leaves garbage in the result, so the model function returns
  // the driver's code at once. EmitC has no return inside a branch, hence the
  // verbatim line. Only functions that return a status can do this; in any
  // other the status is dropped.
  auto fn = op->getParentOfType<func::FuncOp>();
  if (fn && fn->hasAttr(kReturnsStatusAttr))
    rewriter.create<emitc::VerbatimOp>(op->getLoc(), "if ({}) return {};",
                                       ValueRange{status, status});

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

// linalg.matmul and linalg.batch_matmul can carry their own indexing maps,
// which transpose or broadcast an operand, and a cast attribute, which can
// widen the inputs as unsigned. The driver reads A as M x K and B as K x N
// and multiplies signed values, so only the plain form may go to it. Returns
// the reason this one may not, or an empty string.
template <typename MatmulLikeOp>
static std::string whyNotPlainNamedMatmul(MatmulLikeOp op) {
  if (op.hasUserDefinedMaps())
    return "matmul has an operand stored transposed or broadcast, which the "
           "driver cannot read";
  if (op.getCast() != linalg::TypeFn::cast_signed)
    return "matmul multiplies unsigned values, and the accelerator multiplies "
           "signed ones";
  return "";
}

class MatmulToSystolicRewriter: public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter &rewriter) const final{
    std::string why = whyNotPlainNamedMatmul(op);
    if (!why.empty())
      return rewriter.notifyMatchFailure(op, why);

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

// Why a linalg.generic that multiplies and accumulates is not a matmul the
// driver can take, or an empty string if it is one.
static std::string whyNotPlainGenericMatmul(linalg::GenericOp op) {
  static constexpr StringLiteral notMatmulLoops =
      "linalg.generic multiplies and accumulates, but its loops are not those "
      "of a matrix multiply";

  // The loops must be those of one matmul: a row loop M, a column loop N,
  // one reduction K, and nothing else but batch loops. MLIR counts a batch
  // loop that only A and C carry, which is what im2col produces, among the
  // M loops, so M itself is read from the output map below.
  auto dims = linalg::inferContractionDims(op);
  if (failed(dims) || dims->n.size() != 1 || dims->k.size() != 1)
    return notMatmulLoops.str();
  if (op.getNumLoops() != dims->batch.size() + dims->m.size() + 2)
    return notMatmulLoops.str();
  unsigned n = dims->n[0], k = dims->k[0];

  // Report the wrong element types before anything about the layout: a float
  // matmul is first of all a float matmul.
  if (!hasSystolicElementTypes(op.getInputs()[0], op.getInputs()[1],
                               op.getOutputs()[0]))
    return kNotSystolicTypes.str();

  auto maps = op.getIndexingMapsArray();
  unsigned cRank = maps[2].getNumResults();
  if (cRank < 2)
    return notMatmulLoops.str();
  auto mExpr = dyn_cast<AffineDimExpr>(maps[2].getResult(cRank - 2));
  if (!mExpr)
    return notMatmulLoops.str();
  unsigned m = mExpr.getPosition();

  // Each operand must be stored the way the driver reads it: A as M x K,
  // B as K x N and C as M x N, in the last two axes. Leading axes are batch.
  // A transposed operand is still a matmul, but not one the driver can take
  // as it is.
  auto endsWith = [](AffineMap map, unsigned row, unsigned col) {
    unsigned rank = map.getNumResults();
    if (rank < 2 || !map.isProjectedPermutation())
      return false;
    auto r = dyn_cast<AffineDimExpr>(map.getResult(rank - 2));
    auto c = dyn_cast<AffineDimExpr>(map.getResult(rank - 1));
    return r && c && r.getPosition() == row && c.getPosition() == col;
  };
  if (!endsWith(maps[0], m, k) || !endsWith(maps[1], k, n) ||
      !endsWith(maps[2], m, n))
    return "matmul has an operand stored transposed, which the driver cannot "
           "read";

  // The body must compute acc + a * b on its three arguments. Counting a
  // multiply and an add is not enough: a * a + acc has both. The driver
  // multiplies signed integers, so a and b may be widened with a signed
  // extension but not with an unsigned one.
  static constexpr StringLiteral notSignedProduct =
      "linalg.generic multiplies and accumulates, but not as acc + a * b on "
      "signed values";
  Block &body = op.getRegion().front();
  Value a = body.getArgument(0);
  Value b = body.getArgument(1);
  Value acc = body.getArgument(2);

  auto add = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return notSignedProduct.str();
  Value product;
  if (add.getLhs() == acc)
    product = add.getRhs();
  else if (add.getRhs() == acc)
    product = add.getLhs();
  else
    return notSignedProduct.str();

  auto mul = product.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return notSignedProduct.str();
  auto beforeSignExtension = [](Value v) -> Value {
    if (auto ext = v.getDefiningOp<arith::ExtSIOp>())
      return ext.getIn();
    return v;
  };
  Value x = beforeSignExtension(mul.getLhs());
  Value y = beforeSignExtension(mul.getRhs());
  if (!((x == a && y == b) || (x == b && y == a)))
    return notSignedProduct.str();

  return "";
}

class GenericMatmulToSystolicRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    std::string why = whyNotPlainGenericMatmul(op);
    if (!why.empty())
      return rewriter.notifyMatchFailure(op, why);

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

class BatchMatmulToSystolicRewriter
    : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rewriter) const final {
    std::string why = whyNotPlainNamedMatmul(op);
    if (!why.empty())
      return rewriter.notifyMatchFailure(op, why);

    // only single-batch is supported: the driver has no batch concept
    auto aType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    if (!aType || aType.getRank() != 3 || aType.getShape()[0] != 1)
      return failure();

    return rewriteToSystolicCall(op, op.getInputs(), op.getOutputs(), rewriter);
  }
};

// Why a multiply-accumulate that is still in the IR after the rewrite was left
// for the CPU. Never empty.
static std::string whyLeftOnCpu(linalg::LinalgOp op) {
  Operation *raw = op.getOperation();
  SystolicOperands unused;
  std::string why;
  if (auto generic = dyn_cast<linalg::GenericOp>(raw)) {
    why = whyNotPlainGenericMatmul(generic);
    if (why.empty())
      why = whyNotOnSystolic(generic.getInputs(), generic.getOutputs(), unused);
  } else if (auto matmul = dyn_cast<linalg::MatmulOp>(raw)) {
    why = whyNotPlainNamedMatmul(matmul);
    if (why.empty())
      why = whyNotOnSystolic(matmul.getInputs(), matmul.getOutputs(), unused);
  } else if (auto batch = dyn_cast<linalg::BatchMatmulOp>(raw)) {
    why = whyNotPlainNamedMatmul(batch);
    if (why.empty())
      why = whyNotOnSystolic(batch.getInputs(), batch.getOutputs(), unused);
  } else if (isa<linalg::QuantizedMatmulOp, linalg::QuantizedBatchMatmulOp>(
                 raw)) {
    why = "matmul has a zero point, which the accelerator does not subtract";
  } else {
    why = (raw->getName().getStringRef() +
           " is not a matrix multiply the accelerator takes")
              .str();
  }
  if (why.empty())
    why = "matmul was not sent to the accelerator";
  return why;
}

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

    // Whatever still multiplies and accumulates here was left for the CPU.
    // Every such op gets one warning that says why. The patterns cannot report
    // it themselves, because the greedy driver may visit an op more than once.
    root->walk([&](linalg::LinalgOp op) {
      if (!multipliesAndAccumulates(op))
        return;
      // ConvToIm2Col has already said why a convolution was left as it is
      if (linalg::isaConvolutionOpInterface(op))
        return;
      op->emitWarning() << whyLeftOnCpu(op) << "; it stays on the CPU";
    });

    // Record how much went to the accelerator, for the summary at the end.
    int64_t offloaded = 0;
    root->walk([&](emitc::CallOpaqueOp call) {
      if (call.getCallee() == kSystolicCallee)
        offloaded++;
    });
    root->setAttr("icdec.offloaded_ops", OpBuilder(&getContext()).getI64IntegerAttr(offloaded));
  }
};

// Rejects input the rest of the pipeline cannot compile, and says what is
// wrong with it. Without this the same input fails many passes later, with an
// error about an op the user never wrote.
class ValidateInput : public impl::ValidateInputBase<ValidateInput> {
public:
  using impl::ValidateInputBase<ValidateInput>::ValidateInputBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    bool ok = true;
    bool anyFunction = false;

    for (func::FuncOp fn : mod.getOps<func::FuncOp>()) {
      if (fn.isExternal())
        continue;
      anyFunction = true;
      if (!checkFunction(fn))
        ok = false;
    }
    // input that is already EmitC has its functions in that dialect
    if (!mod.getOps<emitc::FuncOp>().empty())
      anyFunction = true;
    if (!anyFunction) {
      mod->emitError() << "no function to compile";
      ok = false;
    }
    if (!ok)
      signalPassFailure();
  }

private:
  // What is wrong with a type, or an empty string if it can be compiled.
  // Scalars are fine; arrays need an element type that the generated C can
  // hold. The arguments and results of a function also need a fixed shape,
  // because they become C arrays. Values inside the function do not: a
  // dynamic size there can still fold to a constant on the way down.
  static std::string problemWith(Type type, bool needsFixedShape) {
    auto shaped = dyn_cast<ShapedType>(type);
    if (!shaped)
      return "";
    if (needsFixedShape && !shaped.hasRank())
      return "has no rank; every shape must be fixed";
    if (needsFixedShape && !shaped.hasStaticShape())
      return "has a dynamic shape; every shape must be fixed";

    Type elem = shaped.getElementType();
    if (elem.isIndex() || elem.isF32() || elem.isF64())
      return "";
    if (auto intTy = dyn_cast<IntegerType>(elem))
      if (intTy.isSignless() &&
          llvm::is_contained({1u, 8u, 16u, 32u, 64u}, intTy.getWidth()))
        return "";
    return "has an element type that is not supported; use i8, i16, i32, "
           "i64, f32 or f64";
  }

  // Reports the first problem in a function, so one mistake gives one error
  // instead of one per op that touches the same value.
  static bool checkFunction(func::FuncOp fn) {
    FunctionType type = fn.getFunctionType();
    for (auto [i, t] : llvm::enumerate(type.getInputs())) {
      std::string problem = problemWith(t, /*needsFixedShape=*/true);
      if (problem.empty())
        continue;
      fn.emitError() << "argument " << i << " of '" << fn.getSymName()
                     << "' has type " << t << ", which " << problem;
      return false;
    }
    for (auto [i, t] : llvm::enumerate(type.getResults())) {
      std::string problem = problemWith(t, /*needsFixedShape=*/true);
      if (problem.empty())
        continue;
      fn.emitError() << "result " << i << " of '" << fn.getSymName()
                     << "' has type " << t << ", which " << problem;
      return false;
    }

    bool ok = true;
    fn.walk([&](Operation *op) {
      if (isa<func::CallOp, func::CallIndirectOp>(op)) {
        op->emitError() << "calls between functions are not supported; the "
                           "model must be a single function";
        ok = false;
        return WalkResult::interrupt();
      }
      for (Type t : op->getResultTypes()) {
        std::string problem = problemWith(t, /*needsFixedShape=*/false);
        if (problem.empty())
          continue;
        op->emitError() << "produces type " << t << ", which " << problem;
        ok = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return ok;
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
    // matmul pass; memcpy from a memref.copy that could not be removed. Each
    // needs its header, and only when actually used.
    static constexpr StringRef mathFns[] = {
        "expf", "exp", "logf", "log", "sqrtf", "sqrt", "tanhf", "tanh",
        "floorf", "floor", "ceilf", "ceil", "roundf", "round", "fabsf", "fabs",
        "cosf", "cos", "sinf", "sin", "powf", "pow"};

    bool usesDriver = false, usesMath = false, usesMemcpy = false;
    mod.walk([&](emitc::CallOpaqueOp call) {
      StringRef callee = call.getCallee();
      if (callee == kSystolicCallee)
        usesDriver = true;
      if (llvm::is_contained(mathFns, callee))
        usesMath = true;
      if (callee == "memcpy")
        usesMemcpy = true;
    });
    if (usesMemcpy && !present.contains("string.h")) {
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<emitc::IncludeOp>(mod.getLoc(),
                                       builder.getStringAttr("string.h"),
                                       /*is_standard_include=*/builder.getUnitAttr());
    }
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

// Every model function returns a status: 0 when it ran to the end, a negative
// driver code when an accelerator call failed. The early exits are added where
// the driver calls are created; this pass supplies the return type and the
// final return.
class ReturnStatus : public impl::ReturnStatusBase<ReturnStatus> {
public:
  using impl::ReturnStatusBase<ReturnStatus>::ReturnStatusBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    OpBuilder builder(&getContext());
    Type statusType = builder.getI32Type();

    // A function called from inside the module keeps its signature, since the
    // call sites are not rewritten here.
    llvm::SmallDenseSet<StringRef> called;
    mod.walk([&](func::CallOp call) { called.insert(call.getCallee()); });

    mod.walk([&](func::FuncOp fn) {
      if (fn.isExternal() || fn.getFunctionType().getNumResults() != 0 ||
          called.contains(fn.getSymName()))
        return;

      SmallVector<func::ReturnOp> returns;
      fn.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
      for (func::ReturnOp ret : returns) {
        builder.setInsertionPoint(ret);
        Value ok = builder.create<emitc::ConstantOp>(
            ret.getLoc(), statusType, builder.getI32IntegerAttr(0));
        builder.create<func::ReturnOp>(ret.getLoc(), ok);
        ret.erase();
      }

      fn.setType(builder.getFunctionType(fn.getFunctionType().getInputs(),
                                         {statusType}));
      fn->setAttr(kReturnsStatusAttr, builder.getUnitAttr());
    });
  }
};

// The op directly in `block` that contains `op`, or null if `op` is not
// inside that block at all.
static Operation *ancestorIn(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

// True if `value` has a user after `from` and before `to`, which are in the
// same block. A user nested in a loop counts by where the loop stands. A user
// outside the block cannot be placed, so it counts as well.
static bool usedBetween(Value value, Operation *from, Operation *to) {
  for (Operation *user : value.getUsers()) {
    if (user == to)
      continue;
    Operation *at = ancestorIn(user, to->getBlock());
    if (!at || (from->isBeforeInBlock(at) && at->isBeforeInBlock(to)))
      return true;
  }
  return false;
}

// True if `value` has a user after `point`, with the same rules.
static bool usedAfter(Value value, Operation *point) {
  for (Operation *user : value.getUsers()) {
    Operation *at = ancestorIn(user, point->getBlock());
    if (!at || point->isBeforeInBlock(at))
      return true;
  }
  return false;
}

// Removes temp buffers that are only written and then copied out wholesale.
// Redirects the writes straight into the copy destination.
//
// The writes then reach the destination earlier than the copy did. That is
// the same program only if nothing looks at the destination between the
// alloc and the copy, and nothing reads the temp after the copy. Reversing a
// buffer through a temp is the case that breaks otherwise: the loop reads
// the destination while it fills the temp.
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
      // "in between" only has a meaning when both are in one block
      if (alloc->getBlock() != copy->getBlock() ||
          !alloc->isBeforeInBlock(copy))
        continue;
      // the destination must not be produced after the alloc
      if (Operation *def = dst.getDefiningOp())
        if (def->getBlock() != alloc->getBlock() ||
            !def->isBeforeInBlock(alloc))
          continue;

      // Nothing may read or write the destination between the alloc and the
      // copy. The destination can be a view of a larger buffer, and a use of
      // that buffer is a use of the destination too, so follow the views down.
      bool destinationInUse = false;
      for (Value v = dst; v && !destinationInUse;) {
        destinationInUse = usedBetween(v, alloc, copy);
        auto view = v.getDefiningOp<ViewLikeOpInterface>();
        v = view ? view.getViewSource() : Value();
      }
      if (destinationInUse)
        continue;
      // After the rewrite the temp and the destination are one buffer, so the
      // temp must not be read once the copy is past.
      if (usedAfter(alloc.getResult(), copy))
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

    // A C array needs at least one dimension. A rank-0 constant would have to
    // become a scalar, with every reader rewritten to match; until that is
    // done, say so instead of failing an assertion.
    bool rankZero = false;
    mod.walk([&](memref::GlobalOp g) {
      if (g.getType().getRank() != 0)
        return;
      g.emitError() << "rank-0 constant is not supported; give it the shape "
                       "[1]";
      rankZero = true;
    });
    if (rankZero) {
      signalPassFailure();
      return;
    }

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
//
// A convolution that is not rewritten stays on the CPU. Each one gets a
// warning that says why, so that the share of work on the CPU in the summary
// is never a surprise.
class ConvToIm2Col : public impl::ConvToIm2ColBase<ConvToIm2Col> {
public:
  using impl::ConvToIm2ColBase<ConvToIm2Col>::ConvToIm2ColBase;

  void runOnOperation() final {
    func::FuncOp fn = getOperation();
    IRRewriter rewriter(fn.getContext());

    // Every kind of convolution, not only the one that can be rewritten.
    // Pooling has the same loop structure but does not multiply, so it is
    // not work the accelerator could take.
    SmallVector<linalg::LinalgOp> convs;
    fn.walk([&](linalg::LinalgOp op) {
      if (linalg::isaConvolutionOpInterface(op) && multipliesAndAccumulates(op))
        convs.push_back(op);
    });

    for (linalg::LinalgOp anyConv : convs) {
      // Only the NHWC convolution with an HWCF kernel turns into one matmul in
      // the orientation the driver expects. The op name tells the others
      // apart: "depthwise" in it, or the "_q" suffix of the variants that
      // subtract a zero point.
      auto op = dyn_cast<linalg::Conv2DNhwcHwcfOp>(anyConv.getOperation());
      if (!op) {
        StringRef name = anyConv->getName().getStringRef();
        if (name.contains("depthwise"))
          anyConv->emitWarning() << "depthwise convolution is one small matmul "
                                    "per channel, which is not offloaded; it "
                                    "stays on the CPU";
        else if (name.ends_with("_q"))
          anyConv->emitWarning() << "convolution has a zero point, which the "
                                    "accelerator does not subtract; it stays "
                                    "on the CPU";
        else
          anyConv->emitWarning() << name << " is not a convolution the "
                                            "accelerator takes; it stays on "
                                            "the CPU";
        continue;
      }

      // The matmul that im2col produces only goes to the accelerator as
      // i16 x i16 -> i32. With other element types the rewrite would just
      // cost an im2col buffer, so leave the convolution as it is.
      if (!hasSystolicElementTypes(op.getInputs()[0], op.getInputs()[1],
                                   op.getOutputs()[0])) {
        op->emitWarning() << "convolution is not i16 x i16 -> i32, the only "
                             "element types the accelerator takes; it stays "
                             "on the CPU";
        continue;
      }

      // Likewise for a batch larger than one: the accelerator takes a single
      // matmul, and im2col would produce one per batch.
      auto inputType = dyn_cast<ShapedType>(op.getInputs()[0].getType());
      if (inputType && inputType.hasRank() && inputType.getDimSize(0) != 1) {
        op->emitWarning() << "convolution has a batch larger than one, which "
                             "the accelerator does not take; it stays on the "
                             "CPU";
        continue;
      }

      // im2col turns the convolution into a matmul with K = KH * KW * C. If
      // that is more than the accelerator takes, the matmul would stay on the
      // CPU anyway, so skip the rewrite and save the im2col buffer.
      auto kernelType = dyn_cast<ShapedType>(op.getInputs()[1].getType());
      if (kernelType && kernelType.hasStaticShape()) {
        ArrayRef<int64_t> k = kernelType.getShape(); // KH, KW, C, F
        int64_t K = k[0] * k[1] * k[2];
        if (K > kSystolicMaxK) {
          op->emitWarning() << "convolution needs K = KH*KW*C = " << K
                            << ", which exceeds the accelerator limit of "
                            << kSystolicMaxK << "; it stays on the CPU";
          continue;
        }
      }

      // The im2col rewrite does not handle a dilated kernel.
      if (llvm::any_of(op.getDilations().getValues<int64_t>(),
                       [](int64_t d) { return d != 1; })) {
        op->emitWarning() << "convolution has a dilation other than one, "
                             "which is not offloaded; it stays on the CPU";
        continue;
      }

      rewriter.setInsertionPoint(op);
      FailureOr<std::pair<Operation *, Operation *>> result =
          linalg::rewriteInIm2Col(rewriter, op);
      if (failed(result))
        op->emitWarning() << "convolution could not be turned into a matmul; "
                             "it stays on the CPU";
      // on success rewriteInIm2Col has already replaced the uses
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
      if (call.getCallee() != kSystolicCallee)
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
  // Say what is wrong with unsupported input before anything touches it.
  pm.addPass(createValidateInput());

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
  pm.addPass(createReturnStatus());

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
