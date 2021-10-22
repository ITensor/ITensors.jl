using ITensors
using ChainRulesCore: ZeroTangent

# Write a custom `rrule` for this.
function inner_circuit(ϕ::ITensor, U::Vector{ITensor}, ψ::ITensor)
  Uψ = ψ
  for u in U
    s = commoninds(u, Uψ)
    s′ = s'
    Uψ = replaceinds(u * Uψ, s′ => s)
  end
  return (ϕ * Uψ)[]
end

name(g::Tuple{String,Vararg}) = g[1]
gate_sites(g::Tuple{<:Any,Tuple{Vararg{Int}},Vararg}) = g[2]
params(g::Tuple{<:Any,<:Any,<:NamedTuple}) = g[3]
params(g::Tuple{<:Any,<:Any}) = NamedTuple()

function gate(g::Tuple, s::Vector{<:Index})
  return gate(name(g), params(g), s[collect(gate_sites(g))])
end

function gate(gn::String, params::NamedTuple, s::Vector{<:Index})
  return gate(gate(gn, params), s)
end

function gate(gn, params::NamedTuple)
  return gate(gn; params...)
end

function gate(gn::String, params::NamedTuple)
  return gate(Val{Symbol(gn)}(), params)
end

function gate(g::Matrix, s::Vector{<:Index})
  s = reverse(s)
  return itensor(g, s'..., dag(s)...)
end

function buildcircuit(gates::Vector, s::Vector{<:Index})
  return [gate(g, s) for g in gates]
end

# Gate definitions
function gate(gn::Val; params...)
  return error("Gate $gn not defined")
end

function gate(::Val{:Ry}; θ)
  return [
    cos(θ / 2) -sin(θ / 2)
    sin(θ / 2) cos(θ / 2)
  ]
end

function gate(::Val{:CX})
  return [
    1 0 0 0
    0 1 0 0
    0 0 0 1
    0 0 1 0
  ]
end

# XXX: For some reason Zygote needs these definitions?
Base.reverse(z::ZeroTangent) = z
Base.adjoint(::Tuple{Nothing}) = nothing
Base.adjoint(::Tuple{Nothing,Nothing}) = nothing
