using Adapt: Adapt, adapt_storage

## TODO for some reason this doesn't work in general need the specialized second function because\
## I recieve an ambiguous function error. Looking more into this
function Adapt.adapt_storage(to::Type{<:AbstractArray}, x::AbstractDiagonalArray)
  dims = size(x)
  diag = adapt_storage(to, x.diag)
  zero = getindex_zero_function(x)
  return DiagonalArray{eltype(x),length(dims),typeof(diag),typeof(zero)}(diag, dims, zero)
end

function Adapt.adapt_storage(to::Type{Array}, x::AbstractDiagonalArray)
  dims = size(x)
  diag = adapt_storage(to, x.diag)
  zero = getindex_zero_function(x)
  return DiagonalArray{eltype(x),length(dims),typeof(diag),typeof(zero)}(diag, dims, zero)
end
