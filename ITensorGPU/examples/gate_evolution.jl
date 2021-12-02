using ITensors
using ITensorGPU

import ITensors: space, state, op

space(::SiteType"Qubit") = 2
state(::SiteType"Qubit", ::StateName"0") = 1
state(::SiteType"Qubit", ::StateName"1") = 2

op_matrix(s::String) = op_matrix(OpName(s))

op_matrix(::OpName"Id") = [
  1 0
  0 1
]

op_matrix(::OpName"I") = op_matrix("Id")

op_matrix(::OpName"X") = [
  0 1
  1 0
]

function op_matrix(on::OpName, s::Index...; kwargs...)
  rs = reverse(s)
  return itensor(op_matrix(on; kwargs...), prime.(rs)..., dag.(rs)...)
end

op_matrix(gn::String, s::Index...; kwargs...) = op_matrix(OpName(gn), s...; kwargs...)

function op_matrix(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...)
  return op_matrix(OpName(gn), s[[ns...]]...; kwargs...)
end

op(gn::OpName, ::SiteType"Qubit", s::Index...; kwargs...) = op_matrix(gn, s...; kwargs...)

N = 10
s = siteinds("Qubit", N)
X = cu.(ops(s, [("X", n) for n in 1:N]))

initstate = fill("0", N)

ψ0 = productCuMPS(s, initstate)

gates = [X[n] for n in 1:2:N]
ψ = apply(gates, ψ0)
