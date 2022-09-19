transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
  %region_op = transform.iree.wrap_in_dispatch_region %0
  transform.iree.region_to_workgroups %region_op
}
