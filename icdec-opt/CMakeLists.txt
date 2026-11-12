set(LIBS
    MLIRArithDialect
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions
    MLIRRegisterAllPasses
    MLIRIcdec
  )
add_llvm_executable(icdec-opt icdec-opt.cpp)

llvm_update_compile_flags(icdec-opt)
target_link_libraries(icdec-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(icdec-opt)
