
"""
    space(::SiteType"Fermion"; 
          conserve_qns=false,
          conserve_nf=conserve_qns,
          conserve_nfparity=conserve_qns,
          qnname_nf = "Nf",
          qnname_nfparity = "NfParity",
          qnname_sz = "Sz",
          conserve_sz = false)

Create the Hilbert space for a site of type "Fermion".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
  ::SiteType"Fermion";
  conserve_qns=false,
  conserve_nf=conserve_qns,
  conserve_nfparity=conserve_qns,
  qnname_nf="Nf",
  qnname_nfparity="NfParity",
  qnname_sz="Sz",
  conserve_sz=false,
  # Deprecated
  conserve_parity=nothing,
)
  if !isnothing(conserve_parity)
    conserve_nfparity = conserve_parity
  end
  if conserve_sz == true
    conserve_sz = "Up"
  end
  if conserve_nf && conserve_sz == "Up"
    zer = QN((qnname_nf, 0, -1), (qnname_sz, 0)) => 1
    one = QN((qnname_nf, 1, -1), (qnname_sz, 1)) => 1
    return [zer, one]
  elseif conserve_nf && conserve_sz == "Dn"
    zer = QN((qnname_nf, 0, -1), (qnname_sz, 0)) => 1
    one = QN((qnname_nf, 1, -1), (qnname_sz, -1)) => 1
    return [zer, one]
  elseif conserve_nfparity && conserve_sz == "Up"
    zer = QN((qnname_nfparity, 0, -2), (qnname_sz, 0)) => 1
    one = QN((qnname_nfparity, 1, -2), (qnname_sz, 1)) => 1
    return [zer, one]
  elseif conserve_nfparity && conserve_sz == "Dn"
    zer = QN((qnname_nfparity, 0, -2), (qnname_sz, 0)) => 1
    one = QN((qnname_nfparity, 1, -2), (qnname_sz, -1)) => 1
    return [zer, one]
  elseif conserve_nf
    zer = QN(qnname_nf, 0, -1) => 1
    one = QN(qnname_nf, 1, -1) => 1
    return [zer, one]
  elseif conserve_nfparity
    zer = QN(qnname_nfparity, 0, -2) => 1
    one = QN(qnname_nfparity, 1, -2) => 1
    return [zer, one]
  end
  return 2
end

ITensors.val(::ValName"Emp", ::SiteType"Fermion") = 1
ITensors.val(::ValName"Occ", ::SiteType"Fermion") = 2
ITensors.val(::ValName"0", st::SiteType"Fermion") = val(ValName("Emp"), st)
ITensors.val(::ValName"1", st::SiteType"Fermion") = val(ValName("Occ"), st)

ITensors.state(::StateName"Emp", ::SiteType"Fermion") = [1.0 0.0]
ITensors.state(::StateName"Occ", ::SiteType"Fermion") = [0.0 1.0]
ITensors.state(::StateName"0", st::SiteType"Fermion") = state(StateName("Emp"), st)
ITensors.state(::StateName"1", st::SiteType"Fermion") = state(StateName("Occ"), st)

function ITensors.op!(Op::ITensor, ::OpName"N", ::SiteType"Fermion", s::Index)
  return Op[s' => 2, s => 2] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"n", st::SiteType"Fermion", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"C", ::SiteType"Fermion", s::Index)
  return Op[s' => 1, s => 2] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"c", st::SiteType"Fermion", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Cdag", ::SiteType"Fermion", s::Index)
  return Op[s' => 2, s => 1] = 1.0
end
function ITensors.op!(Op::ITensor, on::OpName"c†", st::SiteType"Fermion", s::Index)
  return op!(Op, alias(on), st, s)
end
function ITensors.op!(Op::ITensor, on::OpName"cdag", st::SiteType"Fermion", s::Index)
  return op!(Op, alias(on), st, s)
end

function ITensors.op!(Op::ITensor, ::OpName"F", ::SiteType"Fermion", s::Index)
  Op[s' => 1, s => 1] = +1.0
  return Op[s' => 2, s => 2] = -1.0
end

ITensors.has_fermion_string(::OpName"C", ::SiteType"Fermion") = true
function ITensors.has_fermion_string(on::OpName"c", st::SiteType"Fermion")
  return has_fermion_string(alias(on), st)
end
ITensors.has_fermion_string(::OpName"Cdag", ::SiteType"Fermion") = true
function ITensors.has_fermion_string(on::OpName"c†", st::SiteType"Fermion")
  return has_fermion_string(alias(on), st)
end
function ITensors.has_fermion_string(on::OpName"cdag", st::SiteType"Fermion")
  return has_fermion_string(alias(on), st)
end
