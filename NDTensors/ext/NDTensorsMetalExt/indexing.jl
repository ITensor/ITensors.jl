function Base.getindex(::Type{<:MtlArray}, T::DenseTensor{<:Number})
  return Metal.@allowscalar data(T)[]
end

function Base.setindex!(::Type{<:MtlArray}, T::DenseTensor{<:Number}, x::Number)
  Metal.@allowscalar data(T)[] = x
  return T
end
