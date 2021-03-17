#
# Qubit site type
#

space(::SiteType"Qubit") = 2

state(::SiteType"Qubit", ::StateName"0") = 1

state(::SiteType"Qubit", ::StateName"1") = 2

# Use S=1/2 definition of any operators 
# not defined specifically for Qubit below
op(o::OpName, ::SiteType"Qubit") =
  op(o, SiteType("S=1/2"))

#
# 1-Qubit gates
#

op(::OpName"σx",t::SiteType"Qubit") =
  op("X",t) 

op(::OpName"σ1",t::SiteType"Qubit") =
  op("X",t) 

op(::OpName"√NOT",::SiteType"Qubit") =
  [(1+im)/2 (1-im)/2
   (1-im)/2 (1+im)/2]

op(::OpName"√X",t::SiteType"Qubit") =
  op("√NOT",t)
 
op(::OpName"σy",t::SiteType"Qubit") =
  op("Y",t) 

op(::OpName"σ2",t::SiteType"Qubit") =
  op("Y",t) 

op(::OpName"iσy",t::SiteType"Qubit") =
  op("iY",t)

op(::OpName"iσ2",t::SiteType"Qubit") =
  op("iY",t)

op(::OpName"σz",t::SiteType"Qubit") =
  op("Z",t)

op(::OpName"σ3",t::SiteType"Qubit") =
  op("Z",t)

op(::OpName"H",::SiteType"Qubit") =
  [1/sqrt(2) 1/sqrt(2)
   1/sqrt(2) -1/sqrt(2)]

# Rϕ with ϕ = π/2
op(::OpName"Phase",::SiteType"Qubit") =
  [1  0
   0 im]

op(::OpName"P",t::SiteType"Qubit") =
  op("Phase",t)

op(::OpName"S",t::SiteType"Qubit") =
  op("Phase",t)
 
## Rϕ with ϕ = π/4
op(::OpName"π/8",::SiteType"Qubit") =
  [1  0
   0  1/sqrt(2) + im/sqrt(2)]

op(::OpName"T",t::SiteType"Qubit") =
  op("π/8",t)

# Rotation around X-axis
op(::OpName"Rx",::SiteType"Qubit"; θ::Number) =
  [    cos(θ/2)  -im*sin(θ/2)
   -im*sin(θ/2)      cos(θ/2)]

# Rotation around Y-axis
op(::OpName"Ry",::SiteType"Qubit"; θ::Number) =
  [cos(θ/2) -sin(θ/2)
   sin(θ/2)  cos(θ/2)]

# Rotation around Z-axis
op(::OpName"Rz",::SiteType"Qubit"; ϕ::Number) =
  [1         0
   0 exp(im*ϕ)]

# Rotation around generic axis n̂
op(::OpName"Rn",::SiteType"Qubit";
    θ::Real, ϕ::Real, λ::Real) =
  [          cos(θ/2)    -exp(im*λ)*sin(θ/2)
   exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

op(::OpName"Rn̂",t::SiteType"Qubit"; kwargs...) =
  op("Rn",t; kwargs...)

#
# 2-Qubit gates
#

op(::OpName"CNOT",::SiteType"Qubit") =
  [1 0 0 0
   0 1 0 0
   0 0 0 1
   0 0 1 0]

op(::OpName"CX",t::SiteType"Qubit") =
  op("CNOT",t)

op(::OpName"CY",::SiteType"Qubit") =
  [1 0  0   0
   0 1  0   0
   0 0  0 -im
   0 0 im   0]

op(::OpName"CZ",::SiteType"Qubit") =
  [1 0 0  0
   0 1 0  0
   0 0 1  0
   0 0 0 -1]

op(::OpName"SWAP",::SiteType"Qubit") =
  [1 0 0 0
   0 0 1 0
   0 1 0 0
   0 0 0 1]

op(::OpName"√SWAP",::SiteType"Qubit") =
  [1        0        0 0
   0 (1+im)/2 (1-im)/2 0
   0 (1-im)/2 (1+im)/2 0
   0        0        0 1]

# Ising (XX) coupling gate
op(::OpName"XX",::SiteType"Qubit"; ϕ::Number) =
  [    cos(ϕ)          0          0 -im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
   -im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (YY) coupling gate
op(::OpName"YY",::SiteType"Qubit"; ϕ::Number) =
  [    cos(ϕ)          0          0  im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
    im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (ZZ) coupling gate
op(::OpName"ZZ",::SiteType"Qubit"; ϕ::Number) =
  [    exp(-im*ϕ)         0         0          0
            0     exp(im*ϕ)         0          0
            0             0 exp(im*ϕ)          0
            0             0         0 exp(-im*ϕ)]

#
# 3-Qubit gates
#

op(::OpName"Toffoli",::SiteType"Qubit") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1
   0 0 0 0 0 0 1 0]

op(::OpName"CCNOT",t::SiteType"Qubit") =
  op("Toffoli",t)

op(::OpName"CCX",t::SiteType"Qubit") =
  op("Toffoli",t)

op(::OpName"TOFF",t::SiteType"Qubit") =
  op("Toffoli",t)

op(::OpName"Fredkin",::SiteType"Qubit") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 0 1 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1]

op(::OpName"CSWAP",t::SiteType"Qubit") =
  op("Fredkin",t)

op(::OpName"CS",t::SiteType"Qubit") =
  op("Fredkin",t)

#
# 4-Qubit gates
#

op(::OpName"CCCNOT",::SiteType"Qubit") =
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
 
