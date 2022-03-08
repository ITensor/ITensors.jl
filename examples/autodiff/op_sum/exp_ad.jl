using ITensors
using Zygote

include("exp_opsum.jl")
include("opsum_chainrules.jl")

i = Index(2)
A = ITensor([1 0; 0 -1], i', i)

function f(x)
  return exp(x^2 * A)[1, 1]
end

x = 3
@show f(x)
@show f'(x)
