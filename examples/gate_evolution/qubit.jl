using ITensors

#
# Qubit site type
#

ITensors.space(::SiteType"Qubit") = 2

ITensors.state(::SiteType"Qubit", ::StateName"0") = 1

ITensors.state(::SiteType"Qubit", ::StateName"1") = 2

#
# 1-Qubit gates
#

op_matrix(s::String) = op_matrix(OpName(s))

op_matrix(::OpName"Id") =
  [1 0
   0 1]

op_matrix(::OpName"I") =
  op_matrix("Id")

op_matrix(::OpName"X") =
  [0 1
   1 0]

op_matrix(::OpName"σx") =
  op_matrix("X") 

op_matrix(::OpName"σ1") =
  op_matrix("X") 

op_matrix(::OpName"√NOT") =
  [(1+im)/2 (1-im)/2
   (1-im)/2 (1+im)/2]

op_matrix(::OpName"√X") =
  op_matrix("√NOT")

op_matrix(::OpName"Y") =
  [ 0 -im
   im   0]

op_matrix(::OpName"σy") =
  op_matrix("Y") 

op_matrix(::OpName"σ2") =
  op_matrix("Y") 

op_matrix(::OpName"iY") =
  [ 0 1
   -1 0]

op_matrix(::OpName"iσy") =
  op_matrix("iY")

op_matrix(::OpName"iσ2") =
  op_matrix("iY")

# Rϕ with ϕ = π
op_matrix(::OpName"Z") =
  [1  0
   0 -1]

op_matrix(::OpName"σz") =
  op_matrix("Z")

op_matrix(::OpName"σ3") =
  op_matrix("Z")

op_matrix(::OpName"H") =
  [1/sqrt(2) 1/sqrt(2)
   1/sqrt(2) -1/sqrt(2)]

# Rϕ with ϕ = π/2
op_matrix(::OpName"Phase") =
  [1  0
   0 im]

op_matrix(::OpName"P") =
  op_matrix("Phase")

op_matrix(::OpName"S") =
  op_matrix("Phase")

# Rϕ with ϕ = π/4
op_matrix(::OpName"π/8") =
  [1  0
   0  1/sqrt(2) + im/sqrt(2)]

op_matrix(::OpName"T") =
  op_matrix("π/8")

# Rotation around X-axis
op_matrix(::OpName"Rx"; θ::Number) =
  [    cos(θ/2)  -im*sin(θ/2)
   -im*sin(θ/2)      cos(θ/2)]

# Rotation around Y-axis
op_matrix(::OpName"Ry"; θ::Number) =
  [cos(θ/2) -sin(θ/2)
   sin(θ/2)  cos(θ/2)]

# Rotation around Z-axis
op_matrix(::OpName"Rz"; ϕ::Number) =
  [1         0
   0 exp(im*ϕ)]

#  [exp(-im*ϕ/2)           0
#   0            exp(im*ϕ/2)]

# Rotation around generic axis n̂
op_matrix(::OpName"Rn";
    θ::Real, ϕ::Real, λ::Real) =
  [          cos(θ/2)    -exp(im*λ)*sin(θ/2)
   exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

op_matrix(::OpName"Rn̂"; kwargs...) =
  op_matrix("Rn"; kwargs...)

#
# 2-Qubit gates
#

op_matrix(::OpName"CNOT") =
  [1 0 0 0
   0 1 0 0
   0 0 0 1
   0 0 1 0]

op_matrix(::OpName"CX") =
  op_matrix("CNOT")

op_matrix(::OpName"CY") =
  [1 0  0   0
   0 1  0   0
   0 0  0 -im
   0 0 im   0]

op_matrix(::OpName"CZ") =
  [1 0 0  0
   0 1 0  0
   0 0 1  0
   0 0 0 -1]

op_matrix(::OpName"SWAP") =
  [1 0 0 0
   0 0 1 0
   0 1 0 0
   0 0 0 1]

op_matrix(::OpName"√SWAP") =
  [1        0        0 0
   0 (1+im)/2 (1-im)/2 0
   0 (1-im)/2 (1+im)/2 0
   0        0        0 1]

# Ising (XX) coupling gate
op_matrix(::OpName"XX"; ϕ::Number) =
  [    cos(ϕ)          0          0 -im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
   -im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (YY) coupling gate
op_matrix(::OpName"YY"; ϕ::Number) =
  [    cos(ϕ)          0          0  im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
    im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (ZZ) coupling gate
op_matrix(::OpName"ZZ"; ϕ::Number) =
  [    exp(-im*ϕ)         0         0          0
            0     exp(im*ϕ)         0          0
            0             0 exp(im*ϕ)          0
            0             0         0 exp(-im*ϕ)]

#
# 3-Qubit gates
#

op_matrix(::OpName"Toffoli") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1
   0 0 0 0 0 0 1 0]

op_matrix(::OpName"CCNOT") =
  op_matrix("Toffoli")

op_matrix(::OpName"CCX") =
  op_matrix("Toffoli")

op_matrix(::OpName"TOFF") =
  op_matrix("Toffoli")

op_matrix(::OpName"Fredkin") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 0 1 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1]

op_matrix(::OpName"CSWAP") =
  op_matrix("Fredkin")

op_matrix(::OpName"CS") =
  op_matrix("Fredkin")

#
# 4-Qubit gates
#

op_matrix(::OpName"CCCNOT") =
  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]
  
op_matrix(::OpName"randn"; dim) =
    randn(dim)

op_matrix(::OpName"noise"; dim) =
  randn(dim)

op_matrix(gn::OpName; kwargs...) = error("Gate $gn not implemented.")

function op_matrix(on::OpName, s::Index...; kwargs...)
  rs = reverse(s)
  return itensor(op_matrix(on; kwargs...), prime.(rs)..., dag.(rs)...)
end

op_matrix(gn::OpName"randn", s::Index...) =
  itensor(op_matrix(gn; dim = dim(s) * dim(s)), prime.(s)..., dag.(s)...)

op_matrix(gn::OpName"noise", s::Index...; krausind = Index(2, "kraus")) =
  itensor(op_matrix(gn; dim = dim(s) * dim(s) * dim(krausind)), prime.(s)..., dag.(s)..., krausind)

op_matrix(gn::String, s::Index...; kwargs...) =
  op_matrix(OpName(gn), s...; kwargs...)

op_matrix(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
  op_matrix(OpName(gn), s[[ns...]]...; kwargs...)

ITensors.op(gn::OpName, ::SiteType"Qubit", s::Index...; kwargs...) =
  op_matrix(gn, s...; kwargs...)
