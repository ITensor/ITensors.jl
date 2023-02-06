# NDTensors.similar
function similar(storagetype::Type{<:Diag}, dims::Dims)
  return setdata(storagetype, similar(datatype(storagetype), mindim(dims)))
end

# TODO: Redesign UniformDiag to make it handled better
# by generic code.
function similartype(storagetype::Type{<:UniformDiag}, eltype::Type)
  # This will also set the `datatype`.
  return set_eltype(storagetype, eltype)
end
