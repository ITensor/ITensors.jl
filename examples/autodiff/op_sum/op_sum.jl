using ChainRulesCore
using ITensors
using Zygote

function f(x)
  os = OpSum()
  os += (cos(x), "Sz", 1)
  return real(ITensors.coef(os[1]))
end

x = 1.2
@show x
@show f(x)
@show f'(x)

## function g(x)
##   os = OpSum()
##   os += (cos(x), "Sz", 1)
##   o = ITensor(os, s)
##   return o[1, 1]
## end
## 
## s = siteinds("S=1/2", 1)
## @show x
## @show g(x)
## @show g'(x)
