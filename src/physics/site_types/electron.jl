
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
function ITensors.space(
  ::SiteType"Electron";
  conserve_qns=false,
  conserve_sz=conserve_qns,
  conserve_nf=conserve_qns,
  conserve_nfparity=conserve_qns,
  qnname_sz="Sz",
  qnname_nf="Nf",
  qnname_nfparity="NfParity",
  # Deprecated
  conserve_parity=nothing,
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

ITensors.val(::ValName"Emp", ::SiteType"Electron") = 1
ITensors.val(::ValName"Up", ::SiteType"Electron") = 2
ITensors.val(::ValName"Dn", ::SiteType"Electron") = 3
ITensors.val(::ValName"UpDn", ::SiteType"Electron") = 4
ITensors.val(::ValName"0", st::SiteType"Electron") = val(ValName("Emp"), st)
ITensors.val(::ValName"↑", st::SiteType"Electron") = val(ValName("Up"), st)
ITensors.val(::ValName"↓", st::SiteType"Electron") = val(ValName("Dn"), st)
ITensors.val(::ValName"↑↓", st::SiteType"Electron") = val(ValName("UpDn"), st)

ITensors.state(::StateName"Emp", ::SiteType"Electron") = [1.0, 0, 0, 0]
ITensors.state(::StateName"Up", ::SiteType"Electron") = [0.0, 1, 0, 0]
ITensors.state(::StateName"Dn", ::SiteType"Electron") = [0.0, 0, 1, 0]
ITensors.state(::StateName"UpDn", ::SiteType"Electron") = [0.0, 0, 0, 1]
ITensors.state(::StateName"0", st::SiteType"Electron") = state(StateName("Emp"), st)
ITensors.state(::StateName"↑", st::SiteType"Electron") = state(StateName("Up"), st)
ITensors.state(::StateName"↓", st::SiteType"Electron") = state(StateName("Dn"), st)
ITensors.state(::StateName"↑↓", st::SiteType"Electron") = state(StateName("UpDn"), st)

alias(::OpName"c↑") = OpName("Cup")
alias(::OpName"c↓") = OpName("Cdn")
alias(::OpName"c†↑") = OpName("Cdagup")
alias(::OpName"c†↓") = OpName("Cdagdn")
alias(::OpName"n↑") = OpName("Nup")
alias(::OpName"n↓") = OpName("Ndn")
alias(::OpName"n↑↓") = OpName("Nupdn")
alias(::OpName"ntot") = OpName("Ntot")
alias(::OpName"F↑") = OpName("Fup")
alias(::OpName"F↓") = OpName("Fdn")

function ITensors.op!(Op::ITensor, ::OpName"Nup", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 2] = 1.0
  return Op[s' => 4, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"n↑", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Ndn", ::SiteType"Electron", s::Index)
  Op[s' => 3, s => 3] = 1.0
  return Op[s' => 4, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"n↓", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Nupdn", ::SiteType"Electron", s::Index)
  return Op[s' => 4, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"n↑↓", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Ntot", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 2] = 1.0
  Op[s' => 3, s => 3] = 1.0
  return Op[s' => 4, s => 4] = 2.0
end
function ITensors.op!(Op::ITensor, on::OpName"ntot", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Cup", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 2] = 1.0
  return Op[s' => 3, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"c↑", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Cdagup", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 1] = 1.0
  return Op[s' => 4, s => 3] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"c†↑", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Cdn", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 3] = 1.0
  return Op[s' => 2, s => 4] = -1.0
end
function ITensors.op!(Op::ITensor, on::OpName"c↓", st::SiteType"Electron", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Cdagdn", ::SiteType"Electron", s::Index)
  Op[s' => 3, s => 1] = 1.0
  return Op[s' => 4, s => 2] = -1.0
end
function ITensors.op!(Op::ITensor, ::OpName"c†↓", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Cdagdn"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Aup", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 2] = 1.0
  return Op[s' => 3, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, ::OpName"a↑", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Aup"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Adagup", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 1] = 1.0
  return Op[s' => 4, s => 3] = 1.0
end
function ITensors.op!(Op::ITensor, ::OpName"a†↑", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Adagup"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Adn", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 3] = 1.0
  return Op[s' => 2, s => 4] = 1.0
end
function ITensors.op!(Op::ITensor, ::OpName"a↓", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Adn"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Adagdn", ::SiteType"Electron", s::Index)
  Op[s' => 3, s => 1] = 1.0
  return Op[s' => 4, s => 2] = 1.0
end
function ITensors.op!(Op::ITensor, ::OpName"a†↓", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Adagdn"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"F", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = -1.0
  Op[s' => 3, s => 3] = -1.0
  return Op[s' => 4, s => 4] = +1.0
end

function ITensors.op!(Op::ITensor, ::OpName"Fup", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = -1.0
  Op[s' => 3, s => 3] = +1.0
  return Op[s' => 4, s => 4] = -1.0
end
function ITensors.op!(Op::ITensor, ::OpName"F↑", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Fup"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Fdn", ::SiteType"Electron", s::Index)
  Op[s' => 1, s => 1] = +1.0
  Op[s' => 2, s => 2] = +1.0
  Op[s' => 3, s => 3] = -1.0
  return Op[s' => 4, s => 4] = -1.0
end
function ITensors.op!(Op::ITensor, ::OpName"F↓", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Fdn"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sz", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 2] = +0.5
  return Op[s' => 3, s => 3] = -0.5
end

function ITensors.op!(Op::ITensor, ::OpName"Sᶻ", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Sz"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sx", ::SiteType"Electron", s::Index)
  Op[s' => 2, s => 3] = 0.5
  return Op[s' => 3, s => 2] = 0.5
end

function ITensors.op!(Op::ITensor, ::OpName"Sˣ", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("Sx"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"S+", ::SiteType"Electron", s::Index)
  return Op[s' => 2, s => 3] = 1.0
end

function ITensors.op!(Op::ITensor, ::OpName"S⁺", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S+"), st, s)
end
function ITensors.op!(Op::ITensor, ::OpName"Sp", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S+"), st, s)
end
function ITensors.op!(Op::ITensor, ::OpName"Splus", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S+"), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"S-", ::SiteType"Electron", s::Index)
  return Op[s' => 3, s => 2] = 1.0
end

function ITensors.op!(Op::ITensor, ::OpName"S⁻", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S-"), st, s)
end
function ITensors.op!(Op::ITensor, ::OpName"Sm", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S-"), st, s)
end
function ITensors.op!(Op::ITensor, ::OpName"Sminus", st::SiteType"Electron", s::Index)
  return op!(Op, OpName("S-"), st, s)
end

ITensors.has_fermion_string(::OpName"Cup", ::SiteType"Electron") = true
function ITensors.has_fermion_string(on::OpName"c↑", st::SiteType"Electron")
  return has_fermion_string(alias(on), st)
end
ITensors.has_fermion_string(::OpName"Cdagup", ::SiteType"Electron") = true
function ITensors.has_fermion_string(on::OpName"c†↑", st::SiteType"Electron")
  return has_fermion_string(alias(on), st)
end
ITensors.has_fermion_string(::OpName"Cdn", ::SiteType"Electron") = true
function ITensors.has_fermion_string(on::OpName"c↓", st::SiteType"Electron")
  return has_fermion_string(alias(on), st)
end
ITensors.has_fermion_string(::OpName"Cdagdn", ::SiteType"Electron") = true
function ITensors.has_fermion_string(on::OpName"c†↓", st::SiteType"Electron")
  return has_fermion_string(alias(on), st)
end
