using ITensors

#
# qubit
#

ITensors.space(::SiteType"qubit") = 2

ITensors.state(::SiteType"qubit", ::StateName"0") = 1

ITensors.state(::SiteType"qubit", ::StateName"1") = 2

const GateName = OpName

macro GateName_str(s)
  GateName{ITensors.SmallString(s)}
end


#
# 1-qubit gates
#

gate(s::String) = gate(GateName(s))

gate(::GateName"Id") =
  [1 0
   0 1]

gate(::GateName"I") =
  gate("Id")

gate(::GateName"X") =
  [0 1
   1 0]

gate(::GateName"σx") =
  gate("X") 

gate(::GateName"σ1") =
  gate("X") 

gate(::GateName"√NOT") =
  [(1+im)/2 (1-im)/2
   (1-im)/2 (1+im)/2]

gate(::GateName"√X") =
  gate("√NOT")

gate(::GateName"Y") =
  [ 0 -im
   im   0]

gate(::GateName"σy") =
  gate("Y") 

gate(::GateName"σ2") =
  gate("Y") 

gate(::GateName"iY") =
  [ 0 1
   -1 0]

gate(::GateName"iσy") =
  gate("iY")

gate(::GateName"iσ2") =
  gate("iY")

# Rϕ with ϕ = π
gate(::GateName"Z") =
  [1  0
   0 -1]

gate(::GateName"σz") =
  gate("Z")

gate(::GateName"σ3") =
  gate("Z")

gate(::GateName"H") =
  [1/sqrt(2) 1/sqrt(2)
   1/sqrt(2) -1/sqrt(2)]

# Rϕ with ϕ = π/2
gate(::GateName"Phase") =
  [1  0
   0 im]

gate(::GateName"P") =
  gate("Phase")

gate(::GateName"S") =
  gate("Phase")

# Rϕ with ϕ = π/4
gate(::GateName"π/8") =
  [1  0
   0  1/sqrt(2) + im/sqrt(2)]

gate(::GateName"T") =
  gate("π/8")

# Rotation around X-axis
gate(::GateName"Rx"; θ::Number) =
  [    cos(θ/2)  -im*sin(θ/2)
   -im*sin(θ/2)      cos(θ/2)]

# Rotation around Y-axis
gate(::GateName"Ry"; θ::Number) =
  [cos(θ/2) -sin(θ/2)
   sin(θ/2)  cos(θ/2)]

# Rotation around Z-axis
gate(::GateName"Rz"; ϕ::Number) =
  [1         0
   0 exp(im*ϕ)]

#  [exp(-im*ϕ/2)           0
#   0            exp(im*ϕ/2)]

# Rotation around generic axis n̂
gate(::GateName"Rn";
    θ::Real, ϕ::Real, λ::Real) =
  [          cos(θ/2)    -exp(im*λ)*sin(θ/2)
   exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

gate(::GateName"Rn̂"; kwargs...) =
  gate("Rn"; kwargs...)

#
# 2-qubit gates
#

gate(::GateName"CNOT") =
  [1 0 0 0
   0 1 0 0
   0 0 0 1
   0 0 1 0]

gate(::GateName"CX") =
  gate("CNOT")

gate(::GateName"CY") =
  [1 0  0   0
   0 1  0   0
   0 0  0 -im
   0 0 im   0]

gate(::GateName"CZ") =
  [1 0 0  0
   0 1 0  0
   0 0 1  0
   0 0 0 -1]

gate(::GateName"SWAP") =
  [1 0 0 0
   0 0 1 0
   0 1 0 0
   0 0 0 1]

gate(::GateName"√SWAP") =
  [1        0        0 0
   0 (1+im)/2 (1-im)/2 0
   0 (1-im)/2 (1+im)/2 0
   0        0        0 1]

# Ising (XX) coupling gate
gate(::GateName"XX"; ϕ::Number) =
  [    cos(ϕ)          0          0 -im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
   -im*sin(ϕ)          0          0     cos(ϕ)]

# TODO: Ising (YY) coupling gate
#gate(::GateName"YY"; ϕ::Number) =
#  [...]

# TODO: Ising (ZZ) coupling gate
#gate(::GateName"ZZ"; ϕ::Number) =
#  [...]

#
# 3-qubit gates
#

gate(::GateName"Toffoli") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1
   0 0 0 0 0 0 1 0]

gate(::GateName"CCNOT") =
  gate("Toffoli")

gate(::GateName"CCX") =
  gate("Toffoli")

gate(::GateName"TOFF") =
  gate("Toffoli")

gate(::GateName"Fredkin") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 0 1 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1]

gate(::GateName"CSWAP") =
  gate("Fredkin")

gate(::GateName"CS") =
  gate("Fredkin")

#
# 4-qubit gates
#

gate(::GateName"CCCNOT") =
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
  
gate(::GateName"randn"; dim) =
    randn(dim)

gate(::GateName"noise"; dim) =
  randn(dim)

gate(gn::GateName; kwargs...) = error("Gate $gn not implemented.")

function gate(on::GateName, s::Index...; kwargs...)
  rs = reverse(s)
  return itensor(gate(on; kwargs...), prime.(rs)..., dag.(rs)...)
end

gate(gn::GateName"randn", s::Index...) =
  itensor(gate(gn; dim = dim(s) * dim(s)), prime.(s)..., dag.(s)...)

gate(gn::GateName"noise", s::Index...; krausind = Index(2, "kraus")) =
  itensor(gate(gn; dim = dim(s) * dim(s) * dim(krausind)), prime.(s)..., dag.(s)..., krausind)

gate(gn::String, s::Index...; kwargs...) =
  gate(GateName(gn), s...; kwargs...)

gate(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
  gate(GateName(gn), s[[ns...]]...; kwargs...)

ITensors.op(gn::GateName, ::SiteType"qubit", s::Index...; kwargs...) =
  gate(gn, s...; kwargs...)
