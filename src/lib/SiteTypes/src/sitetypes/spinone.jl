using ..ITensors: complex!, QN

alias(::SiteType"SpinOne") = SiteType"S=1"()

"""
    space(::SiteType"S=1";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          qnname_sz = "Sz")

Create the Hilbert space for a site of type "S=1".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function space(
        ::SiteType"S=1"; conserve_qns = false, conserve_sz = conserve_qns, qnname_sz = "Sz"
    )
    if conserve_sz
        return [QN(qnname_sz, +2) => 1, QN(qnname_sz, 0) => 1, QN(qnname_sz, -2) => 1]
    end
    return 3
end

val(::ValName"Up", ::SiteType"S=1") = 1
val(::ValName"Z0", ::SiteType"S=1") = 2
val(::ValName"Dn", ::SiteType"S=1") = 3

val(::ValName"↑", st::SiteType"S=1") = 1
val(::ValName"0", st::SiteType"S=1") = 2
val(::ValName"↓", st::SiteType"S=1") = 3

val(::ValName"Z+", ::SiteType"S=1") = 1
# -- Z0 is already defined above --
val(::ValName"Z-", ::SiteType"S=1") = 3

state(::StateName"Up", ::SiteType"S=1") = [1.0, 0.0, 0.0]
state(::StateName"Z0", ::SiteType"S=1") = [0.0, 1.0, 0.0]
state(::StateName"Dn", ::SiteType"S=1") = [0.0, 0.0, 1.0]

state(::StateName"↑", st::SiteType"S=1") = [1.0, 0.0, 0.0]
state(::StateName"0", st::SiteType"S=1") = [0.0, 1.0, 0.0]
state(::StateName"↓", st::SiteType"S=1") = [0.0, 0.0, 1.0]

state(::StateName"Z+", st::SiteType"S=1") = [1.0, 0.0, 0.0]
# -- Z0 is already defined above --
state(::StateName"Z-", st::SiteType"S=1") = [0.0, 0.0, 1.0]

state(::StateName"X+", ::SiteType"S=1") = [1 / 2, 1 / sqrt(2), 1 / 2]
state(::StateName"X0", ::SiteType"S=1") = [-1 / sqrt(2), 0, 1 / sqrt(2)]
state(::StateName"X-", ::SiteType"S=1") = [1 / 2, -1 / sqrt(2), 1 / 2]

state(::StateName"Y+", ::SiteType"S=1") = [-1 / 2, -im / sqrt(2), 1 / 2]
state(::StateName"Y0", ::SiteType"S=1") = [1 / sqrt(2), 0, 1 / sqrt(2)]
state(::StateName"Y-", ::SiteType"S=1") = [-1 / 2, im / sqrt(2), 1 / 2]

op(::OpName"Sz", ::SiteType"S=1") = [
    1.0 0.0 0.0
    0.0 0.0 0.0
    0.0 0.0 -1.0
]

op(on::OpName"Sᶻ", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"S+", ::SiteType"S=1") = [
    0.0 √2 0.0
    0.0 0.0 √2
    0.0 0.0 0.0
]

op(on::OpName"S⁺", t::SiteType"S=1") = op(alias(on), t)
op(on::OpName"Splus", t::SiteType"S=1") = op(alias(on), t)
op(on::OpName"Sp", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"S-", ::SiteType"S=1") = [
    0.0 0.0 0.0
    √2 0.0 0.0
    0.0 √2 0.0
]

op(on::OpName"S⁻", t::SiteType"S=1") = op(alias(on), t)
op(on::OpName"Sminus", t::SiteType"S=1") = op(alias(on), t)
op(on::OpName"Sm", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"Sx", ::SiteType"S=1") = [
    0.0 1 / √2 0.0
    1 / √2 0.0 1 / √2
    0.0 1 / √2 0.0
]

op(on::OpName"Sˣ", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"iSy", ::SiteType"S=1") = [
    0.0 1 / √2 0.0
    -1 / √2 0.0 1 / √2
    0.0 -1 / √2 0.0
]

op(on::OpName"iSʸ", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"Sy", ::SiteType"S=1") = [
    0.0 -im / √2 0.0
    im / √2 0.0 -im / √2
    0.0 im / √2 0.0
]

op(on::OpName"Sʸ", t::SiteType"S=1") = op(alias(on), t)

op(::OpName"Sz2", ::SiteType"S=1") = [
    1.0 0.0 0.0
    0.0 0.0 0.0
    0.0 0.0 1.0
]

op(::OpName"Sx2", ::SiteType"S=1") = [
    0.5 0.0 0.5
    0.0 1.0 0.0
    0.5 0.0 0.5
]

op(::OpName"Sy2", ::SiteType"S=1") = [
    0.5 0.0 -0.5
    0.0 1.0 0.0
    -0.5 0.0 0.5
]

op(::OpName"S2", ::SiteType"S=1") = [
    2.0 0.0 0.0
    0.0 2.0 0.0
    0.0 0.0 2.0
]

op(on::OpName"S²", t::SiteType"S=1") = op(alias(on), t)

space(st::SiteType"SpinOne"; kwargs...) = space(alias(st); kwargs...)

state(name::StateName, st::SiteType"SpinOne") = state(name, alias(st))
val(name::ValName, st::SiteType"SpinOne") = val(name, alias(st))

function op!(Op::ITensor, o::OpName, st::SiteType"SpinOne", s::Index)
    return op!(Op, o, alias(st), s)
end

op(o::OpName, st::SiteType"SpinOne") = op(o, alias(st))
