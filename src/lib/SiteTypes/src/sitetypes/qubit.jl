using ..ITensors: ITensors

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
function space(
        ::SiteType"Qubit";
        conserve_qns = false,
        conserve_parity = conserve_qns,
        conserve_number = false,
        qnname_parity = "Parity",
        qnname_number = "Number",
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

val(::ValName"0", ::SiteType"Qubit") = 1
val(::ValName"1", ::SiteType"Qubit") = 2
val(::ValName"Up", ::SiteType"Qubit") = 1
val(::ValName"Dn", ::SiteType"Qubit") = 2
val(::ValName"↑", ::SiteType"Qubit") = 1
val(::ValName"↓", ::SiteType"Qubit") = 2

state(::StateName"0", ::SiteType"Qubit") = [1.0, 0.0]
state(::StateName"1", ::SiteType"Qubit") = [0.0, 1.0]
state(::StateName"+", ::SiteType"Qubit") = [1.0, 1.0] / √2
state(::StateName"-", ::SiteType"Qubit") = [1.0, -1.0] / √2
state(::StateName"i", ::SiteType"Qubit") = [1.0, im] / √2
state(::StateName"-i", ::SiteType"Qubit") = [1.0, -im] / √2
state(::StateName"Up", t::SiteType"Qubit") = state(StateName("0"), t)
state(::StateName"Dn", t::SiteType"Qubit") = state(StateName("1"), t)
state(::StateName"↑", t::SiteType"Qubit") = state(StateName("0"), t)
state(::StateName"↓", t::SiteType"Qubit") = state(StateName("1"), t)

# Pauli eingenstates
state(::StateName"X+", t::SiteType"Qubit") = state(StateName("+"), t)
state(::StateName"Xp", t::SiteType"Qubit") = state(StateName("+"), t)
state(::StateName"X-", t::SiteType"Qubit") = state(StateName("-"), t)
state(::StateName"Xm", t::SiteType"Qubit") = state(StateName("-"), t)

state(::StateName"Y+", t::SiteType"Qubit") = state(StateName("i"), t)
state(::StateName"Yp", t::SiteType"Qubit") = state(StateName("i"), t)
state(::StateName"Y-", t::SiteType"Qubit") = state(StateName("-i"), t)
state(::StateName"Ym", t::SiteType"Qubit") = state(StateName("-i"), t)

state(::StateName"Z+", t::SiteType"Qubit") = state(StateName("0"), t)
state(::StateName"Zp", t::SiteType"Qubit") = state(StateName("0"), t)
state(::StateName"Z-", t::SiteType"Qubit") = state(StateName("1"), t)
state(::StateName"Zm", t::SiteType"Qubit") = state(StateName("1"), t)

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
op(::OpName"X", ::SiteType"Qubit") = [
    0 1
    1 0
]

op(::OpName"σx", t::SiteType"Qubit") = op("X", t)

op(::OpName"σ1", t::SiteType"Qubit") = op("X", t)

op(::OpName"Y", ::SiteType"Qubit") = [
    0.0 -1.0im
    1.0im 0.0
]

op(::OpName"σy", t::SiteType"Qubit") = op("Y", t)

op(::OpName"σ2", t::SiteType"Qubit") = op("Y", t)

op(::OpName"iY", ::SiteType"Qubit") = [
    0 1
    -1 0
]
op(::OpName"iσy", t::SiteType"Qubit") = op("iY", t)

op(::OpName"iσ2", t::SiteType"Qubit") = op("iY", t)

op(::OpName"Z", ::SiteType"Qubit") = [
    1 0
    0 -1
]

op(::OpName"σz", t::SiteType"Qubit") = op("Z", t)

op(::OpName"σ3", t::SiteType"Qubit") = op("Z", t)

function op(::OpName"√NOT", ::SiteType"Qubit")
    return [
        (1 + im) / 2 (1 - im) / 2
        (1 - im) / 2 (1 + im) / 2
    ]
end

op(::OpName"√X", t::SiteType"Qubit") = op("√NOT", t)

op(::OpName"H", ::SiteType"Qubit") = [
    1 / sqrt(2) 1 / sqrt(2)
    1 / sqrt(2) -1 / sqrt(2)
]

# Rϕ with ϕ = π/2
op(::OpName"Phase", ::SiteType"Qubit"; ϕ::Number = π / 2) = [
    1 0
    0 exp(im * ϕ)
]

op(::OpName"P", t::SiteType"Qubit"; kwargs...) = op("Phase", t; kwargs...)

op(::OpName"S", t::SiteType"Qubit") = op("Phase", t; ϕ = π / 2)

## Rϕ with ϕ = π/4
op(::OpName"π/8", ::SiteType"Qubit") = [
    1 0
    0 1 / sqrt(2) + im / sqrt(2)
]

op(::OpName"T", t::SiteType"Qubit") = op("π/8", t)

# Rotation around X-axis
function op(::OpName"Rx", ::SiteType"Qubit"; θ::Number)
    return [
        cos(θ / 2) -im * sin(θ / 2)
        -im * sin(θ / 2) cos(θ / 2)
    ]
end

# Rotation around Y-axis
function op(::OpName"Ry", ::SiteType"Qubit"; θ::Number)
    return [
        cos(θ / 2) -sin(θ / 2)
        sin(θ / 2) cos(θ / 2)
    ]
end

# Rotation around Z-axis
function op(::OpName"Rz", ::SiteType"Qubit"; θ = nothing, ϕ = nothing)
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
function op(::OpName"Rn", ::SiteType"Qubit"; θ::Real, ϕ::Real, λ::Real)
    return [
        cos(θ / 2) -exp(im * λ) * sin(θ / 2)
        exp(im * ϕ) * sin(θ / 2) exp(im * (ϕ + λ)) * cos(θ / 2)
    ]
end

function op(on::OpName"Rn̂", t::SiteType"Qubit"; kwargs...)
    return op(alias(on), t; kwargs...)
end

#
# 2-Qubit gates
#

op(::OpName"CNOT", ::SiteType"Qubit") = [
    1 0 0 0
    0 1 0 0
    0 0 0 1
    0 0 1 0
]

op(::OpName"CX", t::SiteType"Qubit") = op("CNOT", t)

op(::OpName"CY", ::SiteType"Qubit") = [
    1 0 0 0
    0 1 0 0
    0 0 0 -im
    0 0 im 0
]

op(::OpName"CZ", ::SiteType"Qubit") = [
    1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 -1
]

function op(::OpName"CPHASE", ::SiteType"Qubit"; ϕ::Number)
    return [
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 exp(im * ϕ)
    ]
end
op(::OpName"Cphase", t::SiteType"Qubit"; kwargs...) = op("CPHASE", t; kwargs...)

function op(::OpName"CRx", ::SiteType"Qubit"; θ::Number)
    return [
        1 0 0 0
        0 1 0 0
        0 0 cos(θ / 2) -im * sin(θ / 2)
        0 0 -im * sin(θ / 2) cos(θ / 2)
    ]
end
op(::OpName"CRX", t::SiteType"Qubit"; kwargs...) = op("CRx", t; kwargs...)

function op(::OpName"CRy", ::SiteType"Qubit"; θ::Number)
    return [
        1 0 0 0
        0 1 0 0
        0 0 cos(θ / 2) -sin(θ / 2)
        0 0 sin(θ / 2) cos(θ / 2)
    ]
end
op(::OpName"CRY", t::SiteType"Qubit"; kwargs...) = op("CRy", t; kwargs...)

function op(::OpName"CRz", ::SiteType"Qubit"; ϕ = nothing, θ = nothing)
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
op(::OpName"CRZ", t::SiteType"Qubit"; kwargs...) = op("CRz", t; kwargs...)

function op(::OpName"CRn", ::SiteType"Qubit"; θ::Number, ϕ::Number, λ::Number)
    return [
        1 0 0 0
        0 1 0 0
        0 0 cos(θ / 2) -exp(im * λ) * sin(θ / 2)
        0 0 exp(im * ϕ) * sin(θ / 2) exp(im * (ϕ + λ)) * cos(θ / 2)
    ]
end
function op(::OpName"CRn̂", t::SiteType"Qubit"; kwargs...)
    return op("CRn", t; kwargs...)
end

op(::OpName"SWAP", ::SiteType"Qubit") = [
    1 0 0 0
    0 0 1 0
    0 1 0 0
    0 0 0 1
]
op(::OpName"Swap", t::SiteType"Qubit") = op("SWAP", t)

function op(::OpName"√SWAP", ::SiteType"Qubit")
    return [
        1 0 0 0
        0 (1 + im) / 2 (1 - im) / 2 0
        0 (1 - im) / 2 (1 + im) / 2 0
        0 0 0 1
    ]
end
op(::OpName"√Swap", t::SiteType"Qubit") = op("√SWAP", t)

op(::OpName"iSWAP", t::SiteType"Qubit") = [
    1 0 0 0
    0 0 im 0
    0 im 0 0
    0 0 0 1
]
op(::OpName"iSwap", t::SiteType"Qubit") = op("iSWAP", t)

function op(::OpName"√iSWAP", t::SiteType"Qubit")
    return [
        1 0 0 0
        0 1 / √2 im / √2 0
        0 im / √2 1 / √2 0
        0 0 0 1
    ]
end
op(::OpName"√iSwap", t::SiteType"Qubit") = op("√iSWAP", t)

# Ising (XX) coupling gate
function op(::OpName"Rxx", t::SiteType"Qubit"; ϕ::Number)
    return [
        cos(ϕ) 0 0 -im * sin(ϕ)
        0 cos(ϕ) -im * sin(ϕ) 0
        0 -im * sin(ϕ) cos(ϕ) 0
        -im * sin(ϕ) 0 0 cos(ϕ)
    ]
end
op(::OpName"RXX", t::SiteType"Qubit"; kwargs...) = op("Rxx", t; kwargs...)

# Ising (YY) coupling gate
function op(::OpName"Ryy", ::SiteType"Qubit"; ϕ::Number)
    return [
        cos(ϕ) 0 0 im * sin(ϕ)
        0 cos(ϕ) -im * sin(ϕ) 0
        0 -im * sin(ϕ) cos(ϕ) 0
        im * sin(ϕ) 0 0 cos(ϕ)
    ]
end
op(::OpName"RYY", t::SiteType"Qubit"; kwargs...) = op("Ryy", t; kwargs...)

# Ising (XY) coupling gate
function op(::OpName"Rxy", t::SiteType"Qubit"; ϕ::Number)
    return [
        1 0 0 0
        0 cos(ϕ) -im * sin(ϕ) 0
        0 -im * sin(ϕ) cos(ϕ) 0
        0 0 0 1
    ]
end
op(::OpName"RXY", t::SiteType"Qubit"; kwargs...) = op("Rxy", t; kwargs...)

# Ising (ZZ) coupling gate
function op(::OpName"Rzz", ::SiteType"Qubit"; ϕ::Number)
    return [
        exp(-im * ϕ) 0 0 0
        0 exp(im * ϕ) 0 0
        0 0 exp(im * ϕ) 0
        0 0 0 exp(-im * ϕ)
    ]
end
op(::OpName"RZZ", t::SiteType"Qubit"; kwargs...) = op("Rzz", t; kwargs...)

#
# 3-Qubit gates
#

function op(::OpName"Toffoli", ::SiteType"Qubit")
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

op(::OpName"CCNOT", t::SiteType"Qubit") = op("Toffoli", t)

op(::OpName"CCX", t::SiteType"Qubit") = op("Toffoli", t)

op(::OpName"TOFF", t::SiteType"Qubit") = op("Toffoli", t)

function op(::OpName"Fredkin", ::SiteType"Qubit")
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

op(::OpName"CSWAP", t::SiteType"Qubit") = op("Fredkin", t)
op(::OpName"CSwap", t::SiteType"Qubit") = op("Fredkin", t)

op(::OpName"CS", t::SiteType"Qubit") = op("Fredkin", t)

#
# 4-Qubit gates
#

function op(::OpName"CCCNOT", ::SiteType"Qubit")
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
op(::OpName"Sz", ::SiteType"Qubit") = [
    0.5 0.0
    0.0 -0.5
]

op(on::OpName"Sᶻ", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"S+", ::SiteType"Qubit") = [
    0 1
    0 0
]

op(on::OpName"S⁺", t::SiteType"Qubit") = op(alias(on), t)

op(on::OpName"Splus", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"S-", ::SiteType"Qubit") = [
    0 0
    1 0
]

op(on::OpName"S⁻", t::SiteType"Qubit") = op(alias(on), t)

op(on::OpName"Sminus", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"Sx", ::SiteType"Qubit") = [
    0.0 0.5
    0.5 0.0
]

op(on::OpName"Sˣ", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"iSy", ::SiteType"Qubit") = [
    0.0 0.5
    -0.5 0.0
]

op(on::OpName"iSʸ", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"Sy", ::SiteType"Qubit") = [
    0.0 -0.5im
    0.5im 0.0
]

op(on::OpName"Sʸ", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"S2", ::SiteType"Qubit") = [
    0.75 0.0
    0.0 0.75
]

op(on::OpName"S²", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"ProjUp", ::SiteType"Qubit") = [
    1 0
    0 0
]

op(on::OpName"projUp", t::SiteType"Qubit") = op(alias(on), t)

op(on::OpName"Proj0", t::SiteType"Qubit") = op(alias(on), t)

op(::OpName"ProjDn", ::SiteType"Qubit") = [
    0 0
    0 1
]

op(on::OpName"projDn", t::SiteType"Qubit") = op(alias(on), t)

op(on::OpName"Proj1", t::SiteType"Qubit") = op(alias(on), t)
