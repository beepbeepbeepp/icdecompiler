set(LLVM_TARGET_DEFINITIONS StandalonePasses.td)
mlir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRStandalonePassesIncGen)
