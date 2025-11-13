"""
    space(::SiteType"Electron";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          conserve_nf = conserve_qns,
          conserve_nfparity = conserve_qns,
          qnname_sz = "Sz",
          qnname_nf = "Nf",
          qnname_nfparity = "NfParity")

Create the Hilbert space for a site of type "Electron".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function space(
        ::SiteType"Electron";
        conserve_qns = false,
        conserve_sz = conserve_qns,
        conserve_nf = conserve_qns,
        conserve_nfparity = conserve_qns,
        qnname_sz = "Sz",
        qnname_nf = "Nf",
        qnname_nfparity = "NfParity",
        # Deprecated
        conserve_parity = nothing,
    )
    if !isnothing(conserve_parity)
        conserve_nfparity = conserve_parity
    end
    if conserve_sz && conserve_nf
        return [
            QN((qnname_nf, 0, -1), (qnname_sz, 0)) => 1
            QN((qnname_nf, 1, -1), (qnname_sz, +1)) => 1
            QN((qnname_nf, 1, -1), (qnname_sz, -1)) => 1
            QN((qnname_nf, 2, -1), (qnname_sz, 0)) => 1
        ]
    elseif conserve_nf
        return [
            QN(qnname_nf, 0, -1) => 1
            QN(qnname_nf, 1, -1) => 2
            QN(qnname_nf, 2, -1) => 1
        ]
    elseif conserve_sz
        return [
            QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
            QN((qnname_sz, +1), (qnname_nfparity, 1, -2)) => 1
            QN((qnname_sz, -1), (qnname_nfparity, 1, -2)) => 1
            QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
        ]
    elseif conserve_nfparity
        return [
            QN(qnname_nfparity, 0, -2) => 1
            QN(qnname_nfparity, 1, -2) => 2
            QN(qnname_nfparity, 0, -2) => 1
        ]
    end
    return 4
end

val(::ValName"Emp", ::SiteType"Electron") = 1
val(::ValName"Up", ::SiteType"Electron") = 2
val(::ValName"Dn", ::SiteType"Electron") = 3
val(::ValName"UpDn", ::SiteType"Electron") = 4
val(::ValName"0", st::SiteType"Electron") = val(ValName("Emp"), st)
val(::ValName"↑", st::SiteType"Electron") = val(ValName("Up"), st)
val(::ValName"↓", st::SiteType"Electron") = val(ValName("Dn"), st)
val(::ValName"↑↓", st::SiteType"Electron") = val(ValName("UpDn"), st)

state(::StateName"Emp", ::SiteType"Electron") = [1.0, 0, 0, 0]
state(::StateName"Up", ::SiteType"Electron") = [0.0, 1, 0, 0]
state(::StateName"Dn", ::SiteType"Electron") = [0.0, 0, 1, 0]
state(::StateName"UpDn", ::SiteType"Electron") = [0.0, 0, 0, 1]
state(::StateName"0", st::SiteType"Electron") = state(StateName("Emp"), st)
state(::StateName"↑", st::SiteType"Electron") = state(StateName("Up"), st)
state(::StateName"↓", st::SiteType"Electron") = state(StateName("Dn"), st)
state(::StateName"↑↓", st::SiteType"Electron") = state(StateName("UpDn"), st)

function op(::OpName"Nup", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↑", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Ndn", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↓", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Nupdn", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
    ]
end
function op(on::OpName"n↑↓", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Ntot", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 2.0
    ]
end
function op(on::OpName"ntot", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Cup", ::SiteType"Electron")
    return [
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
    ]
end
function op(on::OpName"c↑", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Cdagup", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
    ]
end
function op(on::OpName"c†↑", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Cdn", ::SiteType"Electron")
    return [
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end
function op(on::OpName"c↓", st::SiteType"Electron")
    return op(alias(on), st)
end

function op(::OpName"Cdagdn", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
    ]
end
function op(::OpName"c†↓", st::SiteType"Electron")
    return op(OpName("Cdagdn"), st)
end

function op(::OpName"Aup", ::SiteType"Electron")
    return [
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
    ]
end
function op(::OpName"a↑", st::SiteType"Electron")
    return op(OpName("Aup"), st)
end

function op(::OpName"Adagup", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
    ]
end
function op(::OpName"a†↑", st::SiteType"Electron")
    return op(OpName("Adagup"), st)
end

function op(::OpName"Adn", ::SiteType"Electron")
    return [
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 1.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end
function op(::OpName"a↓", st::SiteType"Electron")
    return op(OpName("Adn"), st)
end

function op(::OpName"Adagdn", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
    ]
end
function op(::OpName"a†↓", st::SiteType"Electron")
    return op(OpName("Adagdn"), st)
end

function op(::OpName"F", ::SiteType"Electron")
    return [
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
        0.0 0.0 -1.0 0.0
        0.0 0.0 0.0 1.0
    ]
end

function op(::OpName"Fup", ::SiteType"Electron")
    return [
        1.0 0.0 0.0 0.0
        0.0 -1.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 -1.0
    ]
end
function op(::OpName"F↑", st::SiteType"Electron")
    return op(OpName("Fup"), st)
end

function op(::OpName"Fdn", ::SiteType"Electron")
    return [
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 -1.0 0.0
        0.0 0.0 0.0 -1.0
    ]
end
function op(::OpName"F↓", st::SiteType"Electron")
    return op(OpName("Fdn"), st)
end

function op(::OpName"Sz", ::SiteType"Electron")
    #Op[s' => 2, s => 2] = +0.5
    #return Op[s' => 3, s => 3] = -0.5
    return [
        0.0 0.0 0.0 0.0
        0.0 0.5 0.0 0.0
        0.0 0.0 -0.5 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"Sᶻ", st::SiteType"Electron")
    return op(OpName("Sz"), st)
end

function op(::OpName"Sx", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.5 0.0
        0.0 0.5 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"Sˣ", st::SiteType"Electron")
    return op(OpName("Sx"), st)
end

function op(::OpName"S+", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"S⁺", st::SiteType"Electron")
    return op(OpName("S+"), st)
end
function op(::OpName"Sp", st::SiteType"Electron")
    return op(OpName("S+"), st)
end
function op(::OpName"Splus", st::SiteType"Electron")
    return op(OpName("S+"), st)
end

function op(::OpName"S-", ::SiteType"Electron")
    return [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]
end

function op(::OpName"S⁻", st::SiteType"Electron")
    return op(OpName("S-"), st)
end
function op(::OpName"Sm", st::SiteType"Electron")
    return op(OpName("S-"), st)
end
function op(::OpName"Sminus", st::SiteType"Electron")
    return op(OpName("S-"), st)
end

has_fermion_string(::OpName"Cup", ::SiteType"Electron") = true
function has_fermion_string(on::OpName"c↑", st::SiteType"Electron")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdagup", ::SiteType"Electron") = true
function has_fermion_string(on::OpName"c†↑", st::SiteType"Electron")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdn", ::SiteType"Electron") = true
function has_fermion_string(on::OpName"c↓", st::SiteType"Electron")
    return has_fermion_string(alias(on), st)
end
has_fermion_string(::OpName"Cdagdn", ::SiteType"Electron") = true
function has_fermion_string(on::OpName"c†↓", st::SiteType"Electron")
    return has_fermion_string(alias(on), st)
end
