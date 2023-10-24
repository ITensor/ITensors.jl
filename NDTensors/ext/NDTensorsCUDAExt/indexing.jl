function Base.getindex(::Type{<:CuArray}, T::DenseTensor{<:Number})
  return CUDA.@allowscalar data(T)[]
end

function Base.setindex!(::Type{<:CuArray}, T::DenseTensor{<:Number}, x::Number)
  CUDA.@allowscalar data(T)[] = x
  return T
end
