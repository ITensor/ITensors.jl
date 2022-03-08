using ITensors
using ITensors.Ops
using Zygote

Base.size(o::Sum{Op}) = size(o.args[1])

include("itensors_chainrules.jl")

s = siteinds("S=1/2", 4)

function f1(x)
  y = ITensor(Op("Ry", 1; θ=x), s)
  return y[1, 1]
end

@show x = 2.0

@show f1(x)
@show f1'(x)

function f2(x)
  y = exp(ITensor(Op("Ry", 1; θ=x), s))
  return y[1, 1]
end

@show x = 2.0

@show f2(x)
@show f2'(x)

function f3(x)
  y = exp(ITensor(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x), s))
  return y[1, 1]
end

@show x = 2.0

@show f3(x)
@show f3'(x)
