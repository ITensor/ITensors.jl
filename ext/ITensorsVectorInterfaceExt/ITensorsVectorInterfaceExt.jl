module ITensorsVectorInterfaceExt
using ITensors: ITensors, ITensor
using VectorInterface: VectorInterface

function VectorInterface.add(a::ITensor, b::ITensor)
    return a + b
end
function VectorInterface.add!(a::ITensor, b::ITensor)
    a .= a .+ b
    return a
end
function VectorInterface.add!!(a::ITensor, b::ITensor)
    if promote_type(eltype(a), eltype(b)) <: eltype(a)
        VectorInterface.add!(a, b)
    else
        a = VectorInterface.add(a, b)
    end
    return a
end

function VectorInterface.add(a::ITensor, b::ITensor, α::Number)
    return a + b * α
end
function VectorInterface.add!(a::ITensor, b::ITensor, α::Number)
    a .= a .+ b .* α
    return a
end
function VectorInterface.add!!(a::ITensor, b::ITensor, α::Number)
    if promote_type(eltype(a), eltype(b), typeof(α)) <: eltype(a)
        VectorInterface.add!(a, b, α)
    else
        a = VectorInterface.add(a, b, α)
    end
    return a
end

function VectorInterface.add(a::ITensor, b::ITensor, α::Number, β::Number)
    return a * β + b * α
end
function VectorInterface.add!(a::ITensor, b::ITensor, α::Number, β::Number)
    a .= a .* β .+ b .* α
    return a
end
function VectorInterface.add!!(a::ITensor, b::ITensor, α::Number, β::Number)
    if promote_type(eltype(a), eltype(b), typeof(α), typeof(β)) <: eltype(a)
        VectorInterface.add!(a, b, α, β)
    else
        a = VectorInterface.add(a, b, α, β)
    end
    return a
end

function VectorInterface.inner(a::ITensor, b::ITensor)
    return ITensors.inner(a, b)
end

function VectorInterface.scalartype(a::ITensor)
    return ITensors.scalartype(a)
end

# Circumvent issue that `VectorInterface.jl` computes
# the scalartype in the type domain, which isn't known
# for ITensors.
function VectorInterface.scalartype(a::AbstractArray{ITensor})
    # Like the implementation of `LinearAlgebra.promote_leaf_eltypes`:
    # https://github.com/JuliaLang/LinearAlgebra.jl/blob/e7da19f2764ba36bd0a9eb8ec67dddce19d87114/src/generic.jl#L1933
    return mapreduce(VectorInterface.scalartype, promote_type, a; init = Bool)
end

function VectorInterface.scale(a::ITensor, α::Number)
    return a * α
end
function VectorInterface.scale!(a::ITensor, α::Number)
    a .= a .* α
    return a
end
function VectorInterface.scale!!(a::ITensor, α::Number)
    if promote_type(eltype(a), typeof(α)) <: eltype(a)
        VectorInterface.scale!(a, α)
    else
        a = VectorInterface.scale(a, α)
    end
    return a
end

function VectorInterface.scale!(a_dest::ITensor, a_src::ITensor, α::Number)
    a_dest .= a_src .* α
    return a_dest
end
function VectorInterface.scale!!(a_dest::ITensor, a_src::ITensor, α::Number)
    if promote_type(eltype(a_dest), eltype(a_src), typeof(α)) <: eltype(a_dest)
        VectorInterface.scale!(a_dest, a_src, α)
    else
        a_dest = VectorInterface.scale(a_src, α)
    end
    return a_dest
end

function VectorInterface.zerovector(a::ITensor, type::Type{<:Number})
    a′ = similar(a, type)
    VectorInterface.zerovector!(a′)
    return a′
end
function VectorInterface.zerovector!(a::ITensor)
    a .= zero(eltype(a))
    return a
end
function VectorInterface.zerovector!!(a::ITensor, type::Type{<:Number})
    if type === eltype(a)
        VectorInterface.zerovector!(a)
    else
        a = VectorInterface.zerovector(a, type)
    end
    return a
end
end
