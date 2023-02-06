# NDTensors.similar
function similar(storagetype::Type{<:Diag}, dims::Dims)
  return setdata(storagetype, similar(datatype(storagetype), mindim(dims)))
end
