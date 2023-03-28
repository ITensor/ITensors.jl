using NDTensors
using LinearAlgebra

function LinearAlgebra.norm(x::Metal.MtlArray)
  v = real(eltype(x))(0.0)
  for i in x
    v += real(conj(i) * i)
  end
  return sqrt(v)
end

function transform(old::Metal.MtlArray, new::Metal.MtlArray, func::Function)
  @assert length(old) == length(new)
  for i in 1:length(old)
    new[i] = func(old[i])
  end
  return new
end

function Base.real(x::Metal.MtlArray)
  new_vec = NDTensors.similartype(x, real(eltype(x)))(undef, length(x))
  return transform(x, new_vec, real)
end

function Base.imag(x::Metal.MtlArray)
  new_vec = NDTensors.similartype(x, complex(eltype(x)))(undef, length(x))
  return transform(x, new_vec, imag)
end

function Base.conj(x::Metal.MtlArray)
  new_vec = NDTensors.similartype(x, eltype(x))(undef, length(x))
  return transform(x, new_vec, conj)
end

function Base.:*(x::Number, S::Metal.MtlArray)
  newV = similar(S)
  for i in 1:length(S)
     newV[i] = S[i] * (x)
  end
  return newV
end