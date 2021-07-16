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
function ITensors.space(
  ::SiteType"Qubit";
  conserve_qns=false,
  conserve_parity=conserve_qns,
  conserve_number=false,
  qnname_parity="Parity",
  qnname_number="Number",
)
  if conserve_number && conserve_parity
    return [
      QN((qnname_number, 0), (qnname_parity, 0, 2)) => 1,
      QN((qnname_number, 1), (qnname_parity, 1, 2)) => 1,
    ]
  elseif conserve_number
    return [QN(qnname_number, 0) => 1, QN(qnname_number, 1) => 1]
  elseif conserve_parity
    return [QN(qnname_parity, 0, 2) => 1, QN(qnname_parity, 1, 2) => 1]
  end
  return 2
end

ITensors.val(::ValName"0", st::SiteType"Qubit") = 1
ITensors.val(::ValName"1", st::SiteType"Qubit") = 2

ITensors.state(::StateName"0", ::SiteType"Qubit") = [1.0, 0.0]
ITensors.state(::StateName"1", ::SiteType"Qubit") = [0.0, 1.0]

# Use S=1/2 definition of any operators 
# called using Qubit SiteType
ITensors.op(o::OpName, ::SiteType"Qubit"; kwargs...) = op(o, SiteType("S=1/2"); kwargs...)

#
# 1-Qubit gates
#

ITensors.op(::OpName"σx", t::SiteType"S=1/2") = op("X", t)

ITensors.op(::OpName"σ1", t::SiteType"S=1/2") = op("X", t)

function ITensors.op(::OpName"√NOT", ::SiteType"S=1/2")
  return [
    (1 + im)/2 (1 - im)/2
    (1 - im)/2 (1 + im)/2
  ]
end

ITensors.op(::OpName"√X", t::SiteType"S=1/2") = op("√NOT", t)

ITensors.op(::OpName"σy", t::SiteType"S=1/2") = op("Y", t)

ITensors.op(::OpName"σ2", t::SiteType"S=1/2") = op("Y", t)

ITensors.op(::OpName"iσy", t::SiteType"S=1/2") = op("iY", t)

ITensors.op(::OpName"iσ2", t::SiteType"S=1/2") = op("iY", t)

ITensors.op(::OpName"σz", t::SiteType"S=1/2") = op("Z", t)

ITensors.op(::OpName"σ3", t::SiteType"S=1/2") = op("Z", t)

ITensors.op(::OpName"H", ::SiteType"S=1/2") = [
  1/sqrt(2) 1/sqrt(2)
  1/sqrt(2) -1/sqrt(2)
]

# Rϕ with ϕ = π/2
ITensors.op(::OpName"Phase", ::SiteType"S=1/2") = [
  1 0
  0 im
]

ITensors.op(::OpName"P", t::SiteType"S=1/2") = op("Phase", t)

ITensors.op(::OpName"S", t::SiteType"S=1/2") = op("Phase", t)

## Rϕ with ϕ = π/4
ITensors.op(::OpName"π/8", ::SiteType"S=1/2") = [
  1 0
  0 1 / sqrt(2)+im / sqrt(2)
]

ITensors.op(::OpName"T", t::SiteType"S=1/2") = op("π/8", t)

# Rotation around X-axis
function ITensors.op(::OpName"Rx", ::SiteType"S=1/2"; θ::Number)
  return [
    cos(θ / 2) -im*sin(θ / 2)
    -im*sin(θ / 2) cos(θ / 2)
  ]
end

# Rotation around Y-axis
function ITensors.op(::OpName"Ry", ::SiteType"S=1/2"; θ::Number)
  return [
    cos(θ / 2) -sin(θ / 2)
    sin(θ / 2) cos(θ / 2)
  ]
end

# Rotation around Z-axis
ITensors.op(::OpName"Rz", ::SiteType"S=1/2"; ϕ::Number) = [
  1 0
  0 exp(im * ϕ)
]

# Rotation around generic axis n̂
function ITensors.op(::OpName"Rn", ::SiteType"S=1/2"; θ::Real, ϕ::Real, λ::Real)
  return [
    cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end

ITensors.op(::OpName"Rn̂", t::SiteType"S=1/2"; kwargs...) = ITensors.op("Rn", t; kwargs...)

#
# 2-Qubit gates
#

ITensors.op(::OpName"CNOT", ::SiteType"S=1/2") = [
  1 0 0 0
  0 1 0 0
  0 0 0 1
  0 0 1 0
]

ITensors.op(::OpName"CX", t::SiteType"S=1/2") = op("CNOT", t)

ITensors.op(::OpName"CY", ::SiteType"S=1/2") = [
  1 0 0 0
  0 1 0 0
  0 0 0 -im
  0 0 im 0
]

ITensors.op(::OpName"CZ", ::SiteType"S=1/2") = [
  1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 -1
]

ITensors.op(::OpName"SWAP", ::SiteType"S=1/2") = [
  1 0 0 0
  0 0 1 0
  0 1 0 0
  0 0 0 1
]

function ITensors.op(::OpName"√SWAP", ::SiteType"S=1/2")
  return [
    1 0 0 0
    0 (1 + im)/2 (1 - im)/2 0
    0 (1 - im)/2 (1 + im)/2 0
    0 0 0 1
  ]
end

# Ising (XX) coupling gate
function ITensors.op(::OpName"XX", ::SiteType"S=1/2"; ϕ::Number)
  return [
    cos(ϕ) 0 0 -im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    -im*sin(ϕ) 0 0 cos(ϕ)
  ]
end

# Ising (YY) coupling gate
function ITensors.op(::OpName"YY", ::SiteType"S=1/2"; ϕ::Number)
  return [
    cos(ϕ) 0 0 im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    im*sin(ϕ) 0 0 cos(ϕ)
  ]
end

# Ising (ZZ) coupling gate
function ITensors.op(::OpName"ZZ", ::SiteType"S=1/2"; ϕ::Number)
  return [
    exp(-im * ϕ) 0 0 0
    0 exp(im * ϕ) 0 0
    0 0 exp(im * ϕ) 0
    0 0 0 exp(-im * ϕ)
  ]
end

#
# 3-Qubit gates
#

function ITensors.op(::OpName"Toffoli", ::SiteType"S=1/2")
  return [
    1 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
    0 0 1 0 0 0 0 0
    0 0 0 1 0 0 0 0
    0 0 0 0 1 0 0 0
    0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 1
    0 0 0 0 0 0 1 0
  ]
end

ITensors.op(::OpName"CCNOT", t::SiteType"S=1/2") = op("Toffoli", t)

ITensors.op(::OpName"CCX", t::SiteType"S=1/2") = op("Toffoli", t)

ITensors.op(::OpName"TOFF", t::SiteType"S=1/2") = op("Toffoli", t)

function ITensors.op(::OpName"Fredkin", ::SiteType"S=1/2")
  return [
    1 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
    0 0 1 0 0 0 0 0
    0 0 0 1 0 0 0 0
    0 0 0 0 1 0 0 0
    0 0 0 0 0 0 1 0
    0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 1
  ]
end

ITensors.op(::OpName"CSWAP", t::SiteType"S=1/2") = op("Fredkin", t)

ITensors.op(::OpName"CS", t::SiteType"S=1/2") = op("Fredkin", t)

#
# 4-Qubit gates
#

function ITensors.op(::OpName"CCCNOT", ::SiteType"S=1/2")
  return [
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
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
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
  ]
end
