
## This function is the inverse of dense. Because its not currently possible
## map a blocksparse tensor into a cutensor I am converting BSTs to 
## dense tensors. Then, after contracting, I am mapping the dense tensor
## into the output blocksparse tensor.
# function to_sparse(sparseT::Type{<:BlockSparseTensor}, T::Tensor, blockinds) 
  # R = zeros(dense(TensorT), inds(T))
  # ## Here this failed with scalar indexing (R[blockindices] = blockview)
  # ## We can fix this by using copyto the arrays
  # r = array(R)
  # for block in keys(blockoffsets(T))
  #   # TODO: make sure this assignment is efficient
  #   rview = @view r[blockindices(T, block)]
  #   copyto!(expose(rview), expose(array(blockview(T, block))))
  # end
  # return tensor(Dense(r), inds(T))
# end
