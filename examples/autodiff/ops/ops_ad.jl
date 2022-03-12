using ITensors
using ITensors.Ops
using Zygote

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
  y = Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)
  return y[1].params.θ
end

@show x = 2.0

@show f3(x)
@show f3'(x)

function f4(x)
  y = ITensor(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x), s)
  return y[1, 1]
end

@show x = 2.0

@show f4(x)
@show f4'(x)

function f5(x)
  y = exp(ITensor(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x), s))
  return y[1, 1]
end

@show x = 2.0

@show f5(x)
@show f5'(x)

function f6(x)
  y = ITensor(exp(Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)), s)
  return y[1, 1]
end

@show x = 2.0

@show f6(x)
@show f6'(x)

function f7(x)
  y = ITensor(2 * Op("Ry", 1; θ=x), s)
  return y[1, 1]
end

@show x = 2.0

@show f7(x)
@show f7'(x)

function f8(x)
  y = ITensor(2 * (Op("Ry", 1; θ=x) + Op("Ry", 1; θ=x)), s)
  return y[1, 1]
end

@show x = 2.0

@show f8(x)
@show f8'(x)

function f9(x)
  y = ITensor(Op("Ry", 1; θ=x) * Op("Ry", 2; θ=x), s)
  return y[1, 1]
end

@show x = 2.0

@show f9(x)
@show f9'(x)

function f10(x)
  y = ITensor(exp(-x * Op("X", 1) * Op("X", 2)), s)
  return norm(y)
end

@show x = 2.0

@show f10(x)
@show f10'(x)

V = randomITensor(s[1], s[2])

function f11(x)
  y = exp(-x * Op("X", 1) * Op("X", 2))
  y *= exp(-x * Op("X", 1) * Op("X", 2))
  U = Prod{ITensor}(y, s)
  return norm(U(V))
end

@show x = 2.0

@show f11(x)
@show f11'(x)

function f12(x)
  y = exp(-x * (Op("X", 1) + Op("Z", 1) + Op("Z", 1)); alg=Trotter{1}(1))
  U = Prod{ITensor}(y, s)
  return norm(U(V))
end

@show x = 2.0

@show f12(x)
@show f12'(x)

## ## XXX: Error in vcat!
## function f13(x)
##   y = -x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2))
##   U = ITensor(y, s)
##   return norm(U * V)
## end
## 
## @show x = 2.0
## 
## @show f13(x)
## @show f13'(x)
## 
## ## XXX: Error in vcat!
## function f14(x)
##   y = exp(-x * (Op("X", 1) * Op("X", 2) + Op("Z", 1) * Op("Z", 2)); alg=Trotter{1}(1))
##   U = ITensor(y, s)
##   return norm(U * V)
## end
## 
## @show x = 2.0
## 
## @show f14(x)
## @show f14'(x)
