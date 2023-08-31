# This function actually allocates the data.
# NDTensors.similar
function similar(array::UnallocatedZeros)
  return similartype(array)(dims(array))
end

function similar(array::UnallocatedZeros, elt::Type)
  return similartype(typeof(array), elt)(dims(array))
end

function similar(arraytype::Type{<:UnallocatedZeros}, inds::Tuple)
  shape = Tuple(dim(NDTensors.to_shape(arraytype, inds)))
  return arraytype(shape)
end

function similartype(arraytype::Type{<:UnallocatedZeros})
  return UnallocatedZeros{
    eltype(arraytype),ndims(arraytype),axes(arraytype),alloctype(arraytype)
  }
end

function similartype(arraytype::Type{<:UnallocatedZeros}, elt::Type)
  return UnallocatedZeros{
    elt,ndims(arraytype),NDTensors.axes(arraytype),set_eltype(alloctype(arraytype), elt)
  }
end
