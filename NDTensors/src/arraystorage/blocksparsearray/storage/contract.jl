function contract(
  tensor1::BlockSparseArray,
  labels1,
  tensor2::BlockSparseArray,
  labels2,
)
  return error("Not implemented")
end

function contract(
  tensor1::BlockSparseArray,
  labels1,
  tensor2::Array,
  labels2,
)
  return error("Not implemented")
end

function contract(
  tensor1::Array,
  labels1,
  tensor2::BlockSparseArray,
  labels2,
)
  return error("Not implemented")
end
