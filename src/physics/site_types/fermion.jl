
function space(::SiteType"Fermion"; 
               conserve_qns=false,
               conserve_nf=conserve_qns,
               conserve_nfparity=conserve_qns,
               qnname_nf = "Nf",
               qnname_nfparity = "NfParity",
               qnname_sz = "Sz",
               conserve_sz = false,
               # Deprecated
               conserve_parity=nothing)
  if !isnothing(conserve_parity)
    conserve_nfparity = conserve_parity
  end
  if conserve_sz == true
    conserve_sz = "Up"
  end
  if conserve_nf && conserve_sz == "Up"
    zer = QN((qnname_nf,0,-1), (qnname_sz,0)) => 1
    one = QN((qnname_nf,1,-1), (qnname_sz,1)) => 1
    return [zer,one]
  elseif conserve_nf && conserve_sz == "Dn"
    zer = QN((qnname_nf,0,-1), (qnname_sz,0)) => 1
    one = QN((qnname_nf,1,-1), (qnname_sz,-1)) => 1
    return [zer,one]
  elseif conserve_nfparity && conserve_sz == "Up"
    zer = QN((qnname_nfparity,0,-2), (qnname_sz,0)) => 1
    one = QN((qnname_nfparity,1,-2), (qnname_sz,1)) => 1
    return [zer,one]
  elseif conserve_nfparity && conserve_sz == "Dn"
    zer = QN((qnname_nfparity,0,-2), (qnname_sz,0)) => 1
    one = QN((qnname_nfparity,1,-2), (qnname_sz,-1)) => 1
    return [zer,one]
  elseif conserve_nf
    zer = QN(qnname_nf,0,-1) => 1
    one = QN(qnname_nf,1,-1) => 1
    return [zer,one]
  elseif conserve_nfparity
    zer = QN(qnname_nfparity,0,-2) => 1
    one = QN(qnname_nfparity,1,-2) => 1
    return [zer,one]
  end
  return 2
end

state(::SiteType"Fermion",::StateName"Emp")  = 1
state(::SiteType"Fermion",::StateName"Occ")  = 2
state(st::SiteType"Fermion",::StateName"0") = state(st,StateName("Emp"))
state(st::SiteType"Fermion",::StateName"1") = state(st,StateName("Occ"))

function op!(Op::ITensor,
             ::OpName"N",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>2,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"C",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>1,s=>2] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Cdag",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>2,s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"F",
             ::SiteType"Fermion",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>2,s=>2] = -1.0
end


has_fermion_string(::OpName"C", ::SiteType"Fermion") = true
has_fermion_string(::OpName"Cdag", ::SiteType"Fermion") = true

