# This function actually allocates the data.
# NDTensors.similar
function similar(array::Zeros)
  return similartype(array)(dims(array))
end

function similar(array::Zeros, elt::Type)
  return similartype(typeof(array), elt)(dims(array))
end

function similar(arraytype::Type{<:Zeros}, inds::Tuple)
  shape = Tuple(dim(NDTensors.to_shape(arraytype, inds)))
  return arraytype(shape)
end

function similartype(arraytype::Type{<:Zeros})
  return Zeros{eltype(arraytype),ndims(arraytype),axes(arraytype),alloctype(arraytype)}
end

function similartype(arraytype::Type{<:Zeros}, elt::Type)
  return Zeros{elt,ndims(arraytype),NDTensors.axes(arraytype),set_eltype(alloctype(arraytype), elt)}
end
