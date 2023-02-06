## Overloads needed for `Array`
function similartype(arraytype::Type{<:Array}, eltype::Type)
  return Array{eltype,ndims(arraytype)}
end

function similartype(arraytype::Type{<:Array}, dims::Tuple)
  return Array{eltype(arraytype),length(dims)}
end
