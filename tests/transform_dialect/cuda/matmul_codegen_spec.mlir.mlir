// RUN: iree-opt %s 

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %fused_fill = transform.structured.match ops{["linalg.fill"]} in %arg1

    transform.iree.bufferize { target_gpu } %arg1

    transform.print { name = "after codegen"}
  }
}
