
"""
    space(::SiteType"tJ";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          conserve_nf = conserve_qns,
          conserve_nfparity = conserve_qns,
          qnname_sz = "Sz",
          qnname_nf = "Nf",
          qnname_nfparity = "NfParity")

Create the Hilbert space for a site of type "tJ".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function space(::SiteType"tJ";
               conserve_qns = false,
               conserve_sz = conserve_qns,
               conserve_nf = conserve_qns,
               conserve_nfparity = conserve_qns,
               qnname_sz = "Sz",
               qnname_nf = "Nf",
               qnname_nfparity = "NfParity",
               # Deprecated
               conserve_parity = nothing)
  if !isnothing(conserve_parity)
    conserve_nfparity = conserve_parity
  end
  if conserve_sz && conserve_nf
    return [
       QN((qnname_nf,0,-1),(qnname_sz, 0)) => 1
       QN((qnname_nf,1,-1),(qnname_sz,+1)) => 1
       QN((qnname_nf,1,-1),(qnname_sz,-1)) => 1
      ]
  elseif conserve_nf
    return [
       QN(qnname_nf,0,-1) => 1
       QN(qnname_nf,1,-1) => 2
      ]
  elseif conserve_sz
    return [
       QN((qnname_sz, 0),(qnname_nfparity,0,-2)) => 1
       QN((qnname_sz,+1),(qnname_nfparity,1,-2)) => 1
       QN((qnname_sz,-1),(qnname_nfparity,1,-2)) => 1
      ]
  elseif conserve_nfparity
    return [
       QN(qnname_nfparity,0,-2) => 1
       QN(qnname_nfparity,1,-2) => 2
      ]
  end
  return 3
end

state(::SiteType"tJ",::StateName"Emp")  = 1
state(::SiteType"tJ",::StateName"Up")   = 2
state(::SiteType"tJ",::StateName"Dn")   = 3
state(st::SiteType"tJ",::StateName"0")    = state(st,StateName("Emp"))
state(st::SiteType"tJ",::StateName"↑")    = state(st,StateName("Up"))
state(st::SiteType"tJ",::StateName"↓")    = state(st,StateName("Dn"))

function op!(Op::ITensor,
             ::OpName"Nup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Ndn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Ntot",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
  Op[s'=>3,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdagup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdagdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Aup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adagup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>3] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Adagdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"F",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(Op::ITensor,
             ::OpName"Fup",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
  Op[s'=>3,s=>3] = +1.0
end

function op!(Op::ITensor,
             ::OpName"Fdn",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(Op::ITensor,
             ::OpName"Sz",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>2] = +0.5
  Op[s'=>3,s=>3] = -0.5
end

op!(Op::ITensor,
    ::OpName"Sᶻ",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("Sz"),st,s)

function op!(Op::ITensor,
             ::OpName"Sx",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>3] = 0.5
  Op[s'=>3,s=>2] = 0.5
end

op!(Op::ITensor,
    ::OpName"Sˣ",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("Sx"),st,s)

function op!(Op::ITensor,
             ::OpName"S+",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>2,s=>3] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁺",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)
op!(Op::ITensor,
    ::OpName"Sp",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)
op!(Op::ITensor,
    ::OpName"Splus",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S+"),st,s)

function op!(Op::ITensor,
             ::OpName"S-",
             ::SiteType"tJ",
             s::Index)
  Op[s'=>3,s=>2] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁻",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)
op!(Op::ITensor,
    ::OpName"Sm",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)
op!(Op::ITensor,
    ::OpName"Sminus",
    st::SiteType"tJ",
    s::Index) = op!(Op,OpName("S-"),st,s)

has_fermion_string(::OpName"Cup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdn", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagdn", ::SiteType"tJ") = true
