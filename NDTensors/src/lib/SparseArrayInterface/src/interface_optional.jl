# Optional interface.
# Access a zero value.
function getindex_zero(a::AbstractArray, I)
  return zero(eltype(a))
end

# Optional interface.
# Insert a new zero value.
# Some types (like `Diagonal`) may not support this.
function setindex_zero!(a::AbstractArray, value, I)
  return error("Not implemented")
end
