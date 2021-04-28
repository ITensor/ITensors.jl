
#
# Definitions to move to NDTensors
#

storage(T::Tensor) = store(T)

#function NDTensors.blockdims(inds::IndexSet, block::Tuple)
#  return ntuple(i -> blockdim(inds,block,i), ValLength(block))
#end
