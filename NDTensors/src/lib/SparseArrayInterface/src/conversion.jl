function sparse_convert(arraytype::Type{<:AbstractArray}, a::AbstractArray)
  a_dest = sparse_zero(arraytype, size(a))
  sparse_copyto!(a_dest, a)
  return a_dest
end
