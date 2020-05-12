
"""
inspectQNITensor is a developer-level debugging tool 
to look at internals or properties of QNITensors
"""
function inspectQNITensor(T::ITensor,is::QNIndexSet)
  #@show T.store.blockoffsets
  #@show T.store.data
  println("Block fluxes:")
  for b in nzblocks(T)
    @show flux(T,b)
  end
end
inspectQNITensor(T::ITensor,is::IndexSet) = nothing
inspectQNITensor(T::ITensor) = inspectQNITensor(T,inds(T))
