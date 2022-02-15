
import ITensors: space, state, op!, has_fermion_string

function space(
  ::SiteType"ElecK",
  n::Int;
  conserve_qns=false,
  conserve_sz=conserve_qns,
  conserve_nf=conserve_qns,
  conserve_nfparity=conserve_qns,
  conserve_ky=false,
  qnname_sz="Sz",
  qnname_nf="Nf",
  qnname_nfparity="NfParity",
  qnname_ky="Ky",
  modulus_ky=nothing,
  # Deprecated
  conserve_parity=nothing,
)
  if !isnothing(conserve_parity)
    conserve_nfparity = conserve_parity
  end
  if conserve_ky && conserve_sz && conserve_nf
    mod = (n - 1) % modulus_ky
    mod2 = (2 * mod) % modulus_ky
    return [
      QN((qnname_nf, 0, -1), (qnname_sz, 0), (qnname_ky, 0, modulus_ky)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, 1), (qnname_ky, mod, modulus_ky)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, -1), (qnname_ky, mod, modulus_ky)) => 1
      QN((qnname_nf, 2, -1), (qnname_sz, 0), (qnname_ky, mod2, modulus_ky)) => 1
    ]
  elseif conserve_ky
    error("Cannot conserve ky without conserving sz and nf")
  elseif conserve_sz && conserve_nf
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

state(::StateName"Emp", ::SiteType"ElecK") = [1.0 0.0 0.0 0.0]
state(::StateName"Up", ::SiteType"ElecK") = [0.0 1.0 0.0 0.0]
state(::StateName"Dn", ::SiteType"ElecK") = [0.0 0.0 1.0 0.0]
state(::StateName"UpDn", ::SiteType"ElecK") = [0.0 0.0 0.0 1.0]
state(::StateName"0", st::SiteType"ElecK") = state(StateName("Emp"), st)
state(::StateName"↑", st::SiteType"ElecK") = state(StateName("Up"), st)
state(::StateName"↓", st::SiteType"ElecK") = state(StateName("Dn"), st)
state(::StateName"↑↓", st::SiteType"ElecK") = state(StateName("UpDn"), st)

function op!(Op::ITensor, ::OpName"Nup", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 2] = 1.0
  return Op[s' => 4, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Ndn", ::SiteType"ElecK", s::Index)
  Op[s' => 3, s => 3] = 1.0
  return Op[s' => 4, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Nupdn", ::SiteType"ElecK", s::Index)
  return Op[s' => 4, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Ntot", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 2] = 1.0
  Op[s' => 3, s => 3] = 1.0
  return Op[s' => 4, s => 4] = 2.0
end

function op!(Op::ITensor, ::OpName"Cup", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 2] = 1.0
  return Op[s' => 3, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Cdagup", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 1] = 1.0
  return Op[s' => 4, s => 3] = 1.0
end

function op!(Op::ITensor, ::OpName"Cdn", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 3] = 1.0
  return Op[s' => 2, s => 4] = -1.0
end

function op!(Op::ITensor, ::OpName"Cdagdn", ::SiteType"ElecK", s::Index)
  Op[s' => 3, s => 1] = 1.0
  return Op[s' => 4, s => 2] = -1.0
end

function op!(Op::ITensor, ::OpName"Aup", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 2] = 1.0
  return Op[s' => 3, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Adagup", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 1] = 1.0
  return Op[s' => 4, s => 3] = 1.0
end

function op!(Op::ITensor, ::OpName"Adn", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 3] = 1.0
  return Op[s' => 2, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"Adagdn", ::SiteType"ElecK", s::Index)
  Op[s' => 3, s => 1] = 1.0
  return Op[s' => 2, s => 4] = 1.0
end

function op!(Op::ITensor, ::OpName"F", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = -1.0
  Op[s' => 3, s => 3] = -1.0
  return Op[s' => 4, s => 4] = +1.0
end

function op!(Op::ITensor, ::OpName"Fup", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = -1.0
  Op[s' => 3, s => 3] = +1.0
  return Op[s' => 4, s => 4] = -1.0
end

function op!(Op::ITensor, ::OpName"Fdn", ::SiteType"ElecK", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = +1.0
  Op[s' => 3, s => 3] = -1.0
  return Op[s' => 4, s => 4] = -1.0
end

function op!(Op::ITensor, ::OpName"Sz", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 2] = +0.5
  return Op[s' => 3, s => 3] = -0.5
end

op!(Op::ITensor, ::OpName"Sᶻ", st::SiteType"ElecK", s::Index) = op!(Op, OpName("Sz"), st, s)

function op!(Op::ITensor, ::OpName"Sx", ::SiteType"ElecK", s::Index)
  Op[s' => 2, s => 3] = 0.5
  return Op[s' => 3, s => 2] = 0.5
end

op!(Op::ITensor, ::OpName"Sˣ", st::SiteType"ElecK", s::Index) = op!(Op, OpName("Sx"), st, s)

function op!(Op::ITensor, ::OpName"S+", ::SiteType"ElecK", s::Index)
  return Op[s' => 2, s => 3] = 1.0
end

op!(Op::ITensor, ::OpName"S⁺", st::SiteType"ElecK", s::Index) = op!(Op, OpName("S+"), st, s)
op!(Op::ITensor, ::OpName"Sp", st::SiteType"ElecK", s::Index) = op!(Op, OpName("S+"), st, s)
function op!(Op::ITensor, ::OpName"Splus", st::SiteType"ElecK", s::Index)
  return op!(Op, OpName("S+"), st, s)
end

function op!(Op::ITensor, ::OpName"S-", ::SiteType"ElecK", s::Index)
  return Op[s' => 3, s => 2] = 1.0
end

op!(Op::ITensor, ::OpName"S⁻", st::SiteType"ElecK", s::Index) = op!(Op, OpName("S-"), st, s)
op!(Op::ITensor, ::OpName"Sm", st::SiteType"ElecK", s::Index) = op!(Op, OpName("S-"), st, s)
function op!(Op::ITensor, ::OpName"Sminus", st::SiteType"ElecK", s::Index)
  return op!(Op, OpName("S-"), st, s)
end

has_fermion_string(::OpName"Cup", ::SiteType"ElecK") = true
has_fermion_string(::OpName"Cdagup", ::SiteType"ElecK") = true
has_fermion_string(::OpName"Cdn", ::SiteType"ElecK") = true
has_fermion_string(::OpName"Cdagdn", ::SiteType"ElecK") = true
