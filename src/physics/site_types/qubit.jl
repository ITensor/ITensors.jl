#
# Qubit site type
#

# Define Qubit space in terms of
# S=1/2 space, but use different
# defaults for QN names

"""
    space(::SiteType"Qubit";
          conserve_qns = false,
          conserve_parity = conserve_qns,
          conserve_number = false,
          qnname_parity = "Parity",
          qnname_number = "Number")

Create the Hilbert space for a site of type "Qubit".

Optionally specify the conserved symmetries and their quantum number labels.
"""
space(::SiteType"Qubit";
      conserve_qns=false,
      conserve_parity=conserve_qns,
      conserve_number=false,
      qnname_parity="Parity",
      qnname_number="Number") =
  space(SiteType("S=1/2");
        conserve_qns=conserve_qns,
        conserve_sz=conserve_number,
        conserve_szparity=conserve_parity,
        qnname_sz=qnname_number,
        qnname_szparity=qnname_parity)


state(::SiteType"Qubit", ::StateName"0") = 1

state(::SiteType"Qubit", ::StateName"1") = 2

# Use S=1/2 definition of any operators 
# called using Qubit SiteType
op(o::OpName, ::SiteType"Qubit";kwargs...) =
  op(o, SiteType("S=1/2");kwargs...)

#
# 1-Qubit gates
#

op(::OpName"σx",t::SiteType"S=1/2") =
  op("X",t) 

op(::OpName"σ1",t::SiteType"S=1/2") =
  op("X",t) 

op(::OpName"√NOT",::SiteType"S=1/2") =
  [(1+im)/2 (1-im)/2
   (1-im)/2 (1+im)/2]

op(::OpName"√X",t::SiteType"S=1/2") =
  op("√NOT",t)
 
op(::OpName"σy",t::SiteType"S=1/2") =
  op("Y",t) 

op(::OpName"σ2",t::SiteType"S=1/2") =
  op("Y",t) 

op(::OpName"iσy",t::SiteType"S=1/2") =
  op("iY",t)

op(::OpName"iσ2",t::SiteType"S=1/2") =
  op("iY",t)

op(::OpName"σz",t::SiteType"S=1/2") =
  op("Z",t)

op(::OpName"σ3",t::SiteType"S=1/2") =
  op("Z",t)

op(::OpName"H",::SiteType"S=1/2") =
  [1/sqrt(2) 1/sqrt(2)
   1/sqrt(2) -1/sqrt(2)]

# Rϕ with ϕ = π/2
op(::OpName"Phase",::SiteType"S=1/2") =
  [1  0
   0 im]

op(::OpName"P",t::SiteType"S=1/2") =
  op("Phase",t)

op(::OpName"S",t::SiteType"S=1/2") =
  op("Phase",t)
 
## Rϕ with ϕ = π/4
op(::OpName"π/8",::SiteType"S=1/2") =
  [1  0
   0  1/sqrt(2) + im/sqrt(2)]

op(::OpName"T",t::SiteType"S=1/2") =
  op("π/8",t)

# Rotation around X-axis
op(::OpName"Rx",::SiteType"S=1/2"; θ::Number) =
  [    cos(θ/2)  -im*sin(θ/2)
   -im*sin(θ/2)      cos(θ/2)]

# Rotation around Y-axis
op(::OpName"Ry",::SiteType"S=1/2"; θ::Number) =
  [cos(θ/2) -sin(θ/2)
   sin(θ/2)  cos(θ/2)]

# Rotation around Z-axis
op(::OpName"Rz",::SiteType"S=1/2"; ϕ::Number) =
  [1         0
   0 exp(im*ϕ)]

# Rotation around generic axis n̂
op(::OpName"Rn",::SiteType"S=1/2";
    θ::Real, ϕ::Real, λ::Real) =
  [          cos(θ/2)    -exp(im*λ)*sin(θ/2)
   exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

op(::OpName"Rn̂",t::SiteType"S=1/2"; kwargs...) =
  op("Rn",t; kwargs...)

#
# 2-Qubit gates
#

op(::OpName"CNOT",::SiteType"S=1/2") =
  [1 0 0 0
   0 1 0 0
   0 0 0 1
   0 0 1 0]

op(::OpName"CX",t::SiteType"S=1/2") =
  op("CNOT",t)

op(::OpName"CY",::SiteType"S=1/2") =
  [1 0  0   0
   0 1  0   0
   0 0  0 -im
   0 0 im   0]

op(::OpName"CZ",::SiteType"S=1/2") =
  [1 0 0  0
   0 1 0  0
   0 0 1  0
   0 0 0 -1]

op(::OpName"SWAP",::SiteType"S=1/2") =
  [1 0 0 0
   0 0 1 0
   0 1 0 0
   0 0 0 1]

op(::OpName"√SWAP",::SiteType"S=1/2") =
  [1        0        0 0
   0 (1+im)/2 (1-im)/2 0
   0 (1-im)/2 (1+im)/2 0
   0        0        0 1]

# Ising (XX) coupling gate
op(::OpName"XX",::SiteType"S=1/2"; ϕ::Number) =
  [    cos(ϕ)          0          0 -im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
   -im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (YY) coupling gate
op(::OpName"YY",::SiteType"S=1/2"; ϕ::Number) =
  [    cos(ϕ)          0          0  im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
    im*sin(ϕ)          0          0     cos(ϕ)]

# Ising (ZZ) coupling gate
op(::OpName"ZZ",::SiteType"S=1/2"; ϕ::Number) =
  [    exp(-im*ϕ)         0         0          0
            0     exp(im*ϕ)         0          0
            0             0 exp(im*ϕ)          0
            0             0         0 exp(-im*ϕ)]

#
# 3-Qubit gates
#

op(::OpName"Toffoli",::SiteType"S=1/2") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1
   0 0 0 0 0 0 1 0]

op(::OpName"CCNOT",t::SiteType"S=1/2") =
  op("Toffoli",t)

op(::OpName"CCX",t::SiteType"S=1/2") =
  op("Toffoli",t)

op(::OpName"TOFF",t::SiteType"S=1/2") =
  op("Toffoli",t)

op(::OpName"Fredkin",::SiteType"S=1/2") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 0 1 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1]

op(::OpName"CSWAP",t::SiteType"S=1/2") =
  op("Fredkin",t)

op(::OpName"CS",t::SiteType"S=1/2") =
  op("Fredkin",t)

#
# 4-Qubit gates
#

op(::OpName"CCCNOT",::SiteType"S=1/2") =
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
 
