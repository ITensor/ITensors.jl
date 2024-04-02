module ITensorsVectorInterfaceExt
using ITensors: ITensors, ITensor
using VectorInterface: VectorInterface

function VectorInterface.add(a::ITensor, b::ITensor, α::Number, β::Number)
  a′ = copy(a)
  VectorInterface.add!(a′, b, α, β)
  return a′
end
function VectorInterface.add!(a::ITensor, b::ITensor, α::Number, β::Number)
  a .= a * α + b * β
  return a
end
function VectorInterface.add!!(a::ITensor, b::ITensor, α::Number, β::Number)
  VectorInterface.add!(a, b, α, β)
  return a
end

function VectorInterface.inner(a::ITensor, b::ITensor)
  return ITensors.inner(a, b)
end

function VectorInterface.scalartype(a::ITensor)
  return ITensors.scalartype(a)
end

function VectorInterface.scale(a::ITensor, α::Number)
  a′ = copy(a)
  VectorInterface.scale!(a′, α)
  return a′
end
function VectorInterface.scale!(a::ITensor, α::Number)
  a .= a .* α
  return a
end
function VectorInterface.scale!!(a::ITensor, α::Number)
  VectorInterface.scale!(a, α)
  return a
end

function VectorInterface.zerovector(a::ITensor, type::Type{<:Number})
  a′ = similar(a, type)
  VectorInterface.zerovector!(a′)
  return a′
end
function VectorInterface.zerovector!(a::ITensor)
  fill!(a, zero(eltype(a)))
  return a
end
function VectorInterface.zerovector!!(a::ITensor)
  VectorInterface.zerovector!(a)
  return a
end
end
