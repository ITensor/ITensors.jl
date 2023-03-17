#
# Qubit site type
#

# Define Qubit space in terms of
# Qubit/2 space, but use different
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

ITensors.val(::ValName"0", ::SiteType"Qubit") = 1
ITensors.val(::ValName"1", ::SiteType"Qubit") = 2
ITensors.val(::ValName"Up", ::SiteType"Qubit") = 1
ITensors.val(::ValName"Dn", ::SiteType"Qubit") = 2
ITensors.val(::ValName"↑", ::SiteType"Qubit") = 1
ITensors.val(::ValName"↓", ::SiteType"Qubit") = 2

ITensors.state(::StateName"0", ::SiteType"Qubit") = [1.0, 0.0]
ITensors.state(::StateName"1", ::SiteType"Qubit") = [0.0, 1.0]
ITensors.state(::StateName"+", ::SiteType"Qubit") = [1.0, 1.0] / √2
ITensors.state(::StateName"-", ::SiteType"Qubit") = [1.0, -1.0] / √2
ITensors.state(::StateName"i", ::SiteType"Qubit") = [1.0, im] / √2
ITensors.state(::StateName"-i", ::SiteType"Qubit") = [1.0, -im] / √2
ITensors.state(::StateName"Up", t::SiteType"Qubit") = state(StateName("0"), t)
ITensors.state(::StateName"Dn", t::SiteType"Qubit") = state(StateName("1"), t)
ITensors.state(::StateName"↑", t::SiteType"Qubit") = state(StateName("0"), t)
ITensors.state(::StateName"↓", t::SiteType"Qubit") = state(StateName("1"), t)

# Pauli eingenstates
ITensors.state(::StateName"X+", t::SiteType"Qubit") = state(StateName("+"), t)
ITensors.state(::StateName"Xp", t::SiteType"Qubit") = state(StateName("+"), t)
ITensors.state(::StateName"X-", t::SiteType"Qubit") = state(StateName("-"), t)
ITensors.state(::StateName"Xm", t::SiteType"Qubit") = state(StateName("-"), t)

ITensors.state(::StateName"Y+", t::SiteType"Qubit") = state(StateName("i"), t)
ITensors.state(::StateName"Yp", t::SiteType"Qubit") = state(StateName("i"), t)
ITensors.state(::StateName"Y-", t::SiteType"Qubit") = state(StateName("-i"), t)
ITensors.state(::StateName"Ym", t::SiteType"Qubit") = state(StateName("-i"), t)

ITensors.state(::StateName"Z+", t::SiteType"Qubit") = state(StateName("0"), t)
ITensors.state(::StateName"Zp", t::SiteType"Qubit") = state(StateName("0"), t)
ITensors.state(::StateName"Z-", t::SiteType"Qubit") = state(StateName("1"), t)
ITensors.state(::StateName"Zm", t::SiteType"Qubit") = state(StateName("1"), t)

# SIC-POVMs
state(::StateName"Tetra1", t::SiteType"Qubit") = state(StateName("Z+"), t)
state(::StateName"Tetra2", t::SiteType"Qubit") = [
  1 / √3
  √2 / √3
]
state(::StateName"Tetra3", t::SiteType"Qubit") = [
  1 / √3
  √2 / √3 * exp(im * 2π / 3)
]
state(::StateName"Tetra4", t::SiteType"Qubit") = [
  1 / √3
  √2 / √3 * exp(im * 4π / 3)
]

#
# 1-Qubit gates
#
ITensors.op(::OpName"X", ::SiteType"Qubit") = [
  0 1
  1 0
]

ITensors.op(::OpName"σx", t::SiteType"Qubit") = op("X", t)

ITensors.op(::OpName"σ1", t::SiteType"Qubit") = op("X", t)

ITensors.op(::OpName"Y", ::SiteType"Qubit") = [
  0.0 -1.0im
  1.0im 0.0
]

ITensors.op(::OpName"σy", t::SiteType"Qubit") = op("Y", t)

ITensors.op(::OpName"σ2", t::SiteType"Qubit") = op("Y", t)

ITensors.op(::OpName"iY", ::SiteType"Qubit") = [
  0 1
  -1 0
]
ITensors.op(::OpName"iσy", t::SiteType"Qubit") = op("iY", t)

ITensors.op(::OpName"iσ2", t::SiteType"Qubit") = op("iY", t)

ITensors.op(::OpName"Z", ::SiteType"Qubit") = [
  1 0
  0 -1
]

ITensors.op(::OpName"σz", t::SiteType"Qubit") = op("Z", t)

ITensors.op(::OpName"σ3", t::SiteType"Qubit") = op("Z", t)

function ITensors.op(::OpName"√NOT", ::SiteType"Qubit")
  return [
    (1 + im)/2 (1 - im)/2
    (1 - im)/2 (1 + im)/2
  ]
end

ITensors.op(::OpName"√X", t::SiteType"Qubit") = op("√NOT", t)

ITensors.op(::OpName"H", ::SiteType"Qubit") = [
  1/sqrt(2) 1/sqrt(2)
  1/sqrt(2) -1/sqrt(2)
]

# Rϕ with ϕ = π/2
ITensors.op(::OpName"Phase", ::SiteType"Qubit"; ϕ::Number=π / 2) = [
  1 0
  0 exp(im * ϕ)
]

ITensors.op(::OpName"P", t::SiteType"Qubit"; kwargs...) = op("Phase", t; kwargs...)

ITensors.op(::OpName"S", t::SiteType"Qubit") = op("Phase", t; ϕ=π / 2)

## Rϕ with ϕ = π/4
ITensors.op(::OpName"π/8", ::SiteType"Qubit") = [
  1 0
  0 1 / sqrt(2)+im / sqrt(2)
]

ITensors.op(::OpName"T", t::SiteType"Qubit") = op("π/8", t)

# Rotation around X-axis
function ITensors.op(::OpName"Rx", ::SiteType"Qubit"; θ::Number)
  return [
    cos(θ / 2) -im*sin(θ / 2)
    -im*sin(θ / 2) cos(θ / 2)
  ]
end

# Rotation around Y-axis
function ITensors.op(::OpName"Ry", ::SiteType"Qubit"; θ::Number)
  return [
    cos(θ / 2) -sin(θ / 2)
    sin(θ / 2) cos(θ / 2)
  ]
end

# Rotation around Z-axis
function ITensors.op(::OpName"Rz", ::SiteType"Qubit"; θ=nothing, ϕ=nothing)
  isone(count(isnothing, (θ, ϕ))) || error(
    "Must specify the keyword argument `θ` (or the deprecated `ϕ`) when creating an Rz gate, but not both.",
  )
  isnothing(θ) && (θ = ϕ)
  return [
    exp(-im * θ / 2) 0
    0 exp(im * θ / 2)
  ]
end

# Rotation around generic axis n̂
function ITensors.op(::OpName"Rn", ::SiteType"Qubit"; θ::Real, ϕ::Real, λ::Real)
  return [
    cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end

function ITensors.op(::OpName"Rn̂", t::SiteType"Qubit"; kwargs...)
  return ITensors.op(OpName("Rn"), t; kwargs...)
end

#
# 2-Qubit gates
#

ITensors.op(::OpName"CNOT", ::SiteType"Qubit") = [
  1 0 0 0
  0 1 0 0
  0 0 0 1
  0 0 1 0
]

ITensors.op(::OpName"CX", t::SiteType"Qubit") = op("CNOT", t)

ITensors.op(::OpName"CY", ::SiteType"Qubit") = [
  1 0 0 0
  0 1 0 0
  0 0 0 -im
  0 0 im 0
]

ITensors.op(::OpName"CZ", ::SiteType"Qubit") = [
  1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 -1
]

function ITensors.op(::OpName"CPHASE", ::SiteType"Qubit"; ϕ::Number)
  return [
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 exp(im * ϕ)
  ]
end
ITensors.op(::OpName"Cphase", t::SiteType"Qubit"; kwargs...) = op("CPHASE", t; kwargs...)

function ITensors.op(::OpName"CRx", ::SiteType"Qubit"; θ::Number)
  return [
    1 0 0 0
    0 1 0 0
    0 0 cos(θ / 2) -im*sin(θ / 2)
    0 0 -im*sin(θ / 2) cos(θ / 2)
  ]
end
ITensors.op(::OpName"CRX", t::SiteType"Qubit"; kwargs...) = ITensors.op("CRx", t; kwargs...)

function ITensors.op(::OpName"CRy", ::SiteType"Qubit"; θ::Number)
  return [
    1 0 0 0
    0 1 0 0
    0 0 cos(θ / 2) -sin(θ / 2)
    0 0 sin(θ / 2) cos(θ / 2)
  ]
end
ITensors.op(::OpName"CRY", t::SiteType"Qubit"; kwargs...) = ITensors.op("CRy", t; kwargs...)

function ITensors.op(::OpName"CRz", ::SiteType"Qubit"; ϕ=nothing, θ=nothing)
  isone(count(isnothing, (θ, ϕ))) || error(
    "Must specify the keyword argument `θ` (or the deprecated `ϕ`) when creating a CRz gate, but not both.",
  )
  isnothing(θ) && (θ = ϕ)
  return [
    1 0 0 0
    0 1 0 0
    0 0 exp(-im * θ / 2) 0
    0 0 0 exp(im * θ / 2)
  ]
end
ITensors.op(::OpName"CRZ", t::SiteType"Qubit"; kwargs...) = ITensors.op("CRz", t; kwargs...)

function ITensors.op(::OpName"CRn", ::SiteType"Qubit"; θ::Number, ϕ::Number, λ::Number)
  return [
    1 0 0 0
    0 1 0 0
    0 0 cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    0 0 exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end
function ITensors.op(::OpName"CRn̂", t::SiteType"Qubit"; kwargs...)
  return ITensors.op("CRn", t; kwargs...)
end

ITensors.op(::OpName"SWAP", ::SiteType"Qubit") = [
  1 0 0 0
  0 0 1 0
  0 1 0 0
  0 0 0 1
]
ITensors.op(::OpName"Swap", t::SiteType"Qubit") = op("SWAP", t)

function ITensors.op(::OpName"√SWAP", ::SiteType"Qubit")
  return [
    1 0 0 0
    0 (1 + im)/2 (1 - im)/2 0
    0 (1 - im)/2 (1 + im)/2 0
    0 0 0 1
  ]
end
ITensors.op(::OpName"√Swap", t::SiteType"Qubit") = op("√SWAP", t)

ITensors.op(::OpName"iSWAP", t::SiteType"Qubit") = [
  1 0 0 0
  0 0 im 0
  0 im 0 0
  0 0 0 1
]
ITensors.op(::OpName"iSwap", t::SiteType"Qubit") = op("iSWAP", t)

function ITensors.op(::OpName"√iSWAP", t::SiteType"Qubit")
  return [
    1 0 0 0
    0 1/√2 im/√2 0
    0 im/√2 1/√2 0
    0 0 0 1
  ]
end
ITensors.op(::OpName"√iSwap", t::SiteType"Qubit") = op("√iSWAP", t)

# Ising (XX) coupling gate
function ITensors.op(::OpName"Rxx", t::SiteType"Qubit"; ϕ::Number)
  return [
    cos(ϕ) 0 0 -im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    -im*sin(ϕ) 0 0 cos(ϕ)
  ]
end
ITensors.op(::OpName"RXX", t::SiteType"Qubit"; kwargs...) = op("Rxx", t; kwargs...)

# Ising (YY) coupling gate
function ITensors.op(::OpName"Ryy", ::SiteType"Qubit"; ϕ::Number)
  return [
    cos(ϕ) 0 0 im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    im*sin(ϕ) 0 0 cos(ϕ)
  ]
end
ITensors.op(::OpName"RYY", t::SiteType"Qubit"; kwargs...) = op("Ryy", t; kwargs...)

# Ising (XY) coupling gate
function ITensors.op(::OpName"Rxy", t::SiteType"Qubit"; ϕ::Number)
  return [
    1 0 0 0
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    0 0 0 1
  ]
end
ITensors.op(::OpName"RXY", t::SiteType"Qubit"; kwargs...) = op("Rxy", t; kwargs...)

# Ising (ZZ) coupling gate
function ITensors.op(::OpName"Rzz", ::SiteType"Qubit"; ϕ::Number)
  return [
    exp(-im * ϕ) 0 0 0
    0 exp(im * ϕ) 0 0
    0 0 exp(im * ϕ) 0
    0 0 0 exp(-im * ϕ)
  ]
end
ITensors.op(::OpName"RZZ", t::SiteType"Qubit"; kwargs...) = op("Rzz", t; kwargs...)

#
# 3-Qubit gates
#

function ITensors.op(::OpName"Toffoli", ::SiteType"Qubit")
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

ITensors.op(::OpName"CCNOT", t::SiteType"Qubit") = op("Toffoli", t)

ITensors.op(::OpName"CCX", t::SiteType"Qubit") = op("Toffoli", t)

ITensors.op(::OpName"TOFF", t::SiteType"Qubit") = op("Toffoli", t)

function ITensors.op(::OpName"Fredkin", ::SiteType"Qubit")
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

ITensors.op(::OpName"CSWAP", t::SiteType"Qubit") = op("Fredkin", t)
ITensors.op(::OpName"CSwap", t::SiteType"Qubit") = op("Fredkin", t)

ITensors.op(::OpName"CS", t::SiteType"Qubit") = op("Fredkin", t)

#
# 4-Qubit gates
#

function ITensors.op(::OpName"CCCNOT", ::SiteType"Qubit")
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

# spin-full operators
ITensors.op(::OpName"Sz", ::SiteType"Qubit") = [
  0.5 0.0
  0.0 -0.5
]

ITensors.op(::OpName"Sᶻ", t::SiteType"Qubit") = op(OpName("Sz"), t)

ITensors.op(::OpName"S+", ::SiteType"Qubit") = [
  0 1
  0 0
]

ITensors.op(::OpName"S⁺", t::SiteType"Qubit") = op(OpName("S+"), t)

ITensors.op(::OpName"Splus", t::SiteType"Qubit") = op(OpName("S+"), t)

ITensors.op(::OpName"S-", ::SiteType"Qubit") = [
  0 0
  1 0
]

ITensors.op(::OpName"S⁻", t::SiteType"Qubit") = op(OpName("S-"), t)

ITensors.op(::OpName"Sminus", t::SiteType"Qubit") = op(OpName("S-"), t)

ITensors.op(::OpName"Sx", ::SiteType"Qubit") = [
  0.0 0.5
  0.5 0.0
]

ITensors.op(::OpName"Sˣ", t::SiteType"Qubit") = op(OpName("Sx"), t)

ITensors.op(::OpName"iSy", ::SiteType"Qubit") = [
  0.0 0.5
  -0.5 0.0
]

ITensors.op(::OpName"iSʸ", t::SiteType"Qubit") = op(OpName("iSy"), t)

ITensors.op(::OpName"Sy", ::SiteType"Qubit") = [
  0.0 -0.5im
  0.5im 0.0
]

ITensors.op(::OpName"Sʸ", t::SiteType"Qubit") = op(OpName("Sy"), t)

ITensors.op(::OpName"S2", ::SiteType"Qubit") = [
  0.75 0.0
  0.0 0.75
]

ITensors.op(::OpName"S²", t::SiteType"Qubit") = op(OpName("S2"), t)

ITensors.op(::OpName"ProjUp", ::SiteType"Qubit") = [
  1 0
  0 0
]

ITensors.op(::OpName"projUp", t::SiteType"Qubit") = op(OpName("ProjUp"), t)

ITensors.op(::OpName"Proj0", t::SiteType"Qubit") = op(OpName("ProjUp"), t)

ITensors.op(::OpName"ProjDn", ::SiteType"Qubit") = [
  0 0
  0 1
]

ITensors.op(::OpName"projDn", t::SiteType"Qubit") = op(OpName("ProjDn"), t)

ITensors.op(::OpName"Proj1", t::SiteType"Qubit") = op(OpName("ProjDn"), t)
