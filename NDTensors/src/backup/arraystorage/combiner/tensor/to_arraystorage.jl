# TODO: Delete when we directly use `CombinerArray` as storage.
function to_arraystorage(t::CombinerTensor)
  return tensor(CombinerArray(storage(t), to_axes(inds(t))), inds(t))
end
