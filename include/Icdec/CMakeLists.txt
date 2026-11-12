set(LLVM_TARGET_DEFINITIONS IcdecPasses.td)
mlir_tablegen(IcdecPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRIcdecPassesIncGen)
