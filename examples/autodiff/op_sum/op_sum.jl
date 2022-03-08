using ITensors
using Zygote

include("exp_opsum.jl")
include("opsum_chainrules.jl")

function f(x)
  os = OpSum()
  os += (cos(x), "Sz", 1)
  os += (sin(x), "Sz", 3)
  return real(coef(os[1]) + coef(os[2]))
end

x = 2.0
@show x
@show f(x)
@show f'(x)

s = siteinds("S=1/2", 3)
function g(x)
  os = OpSum()
  os += x, "Sz", 1, "Sz", 3
  os += x, "Sx", 1, "Sx", 2
  u = exp(os)
  U = ops(u, s)
  return real(U[1][1])
end

@show g(x)
@show g'(x)
