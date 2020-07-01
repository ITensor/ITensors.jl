

function space(::SiteType"Electron"; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  conserve_nf = get(kwargs,:conserve_nf,conserve_qns)
  conserve_parity = get(kwargs,:conserve_parity,conserve_qns)
  if conserve_sz && conserve_nf
    em = QN(("Nf",0,-1),("Sz", 0)) => 1
    up = QN(("Nf",1,-1),("Sz",+1)) => 1
    dn = QN(("Nf",1,-1),("Sz",-1)) => 1
    ud = QN(("Nf",2,-1),("Sz", 0)) => 1
    return [em,up,dn,ud]
  elseif conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 2
    two = QN("Nf",2,-1) => 1
    return [zer,one,two]
  elseif conserve_sz
    em = QN(("Sz", 0),("Pf",0,-2)) => 1
    up = QN(("Sz",+1),("Pf",1,-2)) => 1
    dn = QN(("Sz",-1),("Pf",1,-2)) => 1
    ud = QN(("Sz", 0),("Pf",0,-2)) => 1
    return [em,up,dn,ud]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 2
    two = QN("Pf",0,-2) => 1
    return [zer,one,two]
  end
  return 4
end

state(::SiteType"Electron",::StateName"Emp")  = 1
state(::SiteType"Electron",::StateName"Up")   = 2
state(::SiteType"Electron",::StateName"Dn")   = 3
state(::SiteType"Electron",::StateName"UpDn") = 4
state(::SiteType"Electron",::StateName"0")    = 1
state(::SiteType"Electron",::StateName"↑")    = 2
state(::SiteType"Electron",::StateName"↓")    = 3
state(::SiteType"Electron",::StateName"↑↓")   = 4

function op(::SiteType"Electron",
            s::Index,
            opname::AbstractString)::ITensor
  Emp   = s(1)
  EmpP  = s'(1)
  Up    = s(2)
  UpP   = s'(2)
  Dn    = s(3)
  DnP   = s'(3)
  UpDn  = s(4)
  UpDnP = s'(4)

  Op = emptyITensor(s',dag(s))

  if opname == "Nup"
    Op[UpP, Up] = 1.
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Ndn"
    Op[DnP, Dn] = 1.
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Nupdn"
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Ntot"
    Op[UpP, Up] = 1.
    Op[DnP, Dn] = 1.
    Op[UpDnP, UpDn] = 2.
  elseif opname == "Cup"
    Op[EmpP, Up]  = +1.
    Op[DnP, UpDn] = +1.
  elseif opname == "Cdagup"
    Op[UpP, Emp]  = +1.
    Op[UpDnP, Dn] = +1.
  elseif opname == "Cdn"
    Op[EmpP, Dn]  = +1.
    Op[UpP, UpDn] = -1.
  elseif opname == "Cdagdn"
    Op[DnP, Emp]  = +1.
    Op[UpDnP, Up] = -1.
  # Aup,Adagup,Adn,Adagdn below
  # are "bosonic" versions of
  # the creation/annihilation
  # C operators defined above
  elseif opname == "Aup"
    Op[EmpP, Up]  = 1.
    Op[DnP, UpDn] = 1.
  elseif opname == "Adagup"
    Op[UpP, Emp]  = 1.
    Op[UpDnP, Dn] = 1.
  elseif opname == "Adn"
    Op[EmpP, Dn]  = 1.
    Op[UpP, UpDn] = 1.
  elseif opname == "Adagdn"
    Op[DnP, Emp]  = 1.
    Op[UpDnP, Up] = 1.
  elseif opname=="F" || opname=="FermiPhase" || opname=="FP"
    Op[UpP, Up] = -1.
    Op[EmpP, Emp] = 1.
    Op[DnP, Dn] = -1.
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Fup"
    Op[EmpP, Emp] = 1.
    Op[UpP, Up] = -1.
    Op[DnP, Dn] = 1.
    Op[UpDnP, UpDn] = -1.
  elseif opname == "Fdn"
    Op[EmpP, Emp] = 1.
    Op[UpP, Up] = 1.
    Op[DnP, Dn] = -1.
    Op[UpDnP, UpDn] = -1.
  elseif opname == "Sᶻ" || opname == "Sz"
    Op[UpP, Up] = 0.5
    Op[DnP, Dn] = -0.5
  elseif opname == "Sˣ" || opname == "Sx"
    Op[DnP, Up] = 0.5
    Op[UpP, Dn] = 0.5 
  elseif opname=="S+" || opname=="Sp" || opname == "S⁺" || opname == "Splus"
    Op[UpP, Dn] = 1.0
  elseif opname=="S-" || opname=="Sm" || opname == "S⁻" || opname == "Sminus"
    Op[DnP, Up] = 1.0
  elseif opname == "Emp" || opname == "0"
    pEmp = emptyITensor(s)
    pEmp[Emp] = 1.0
    return pEmp
  elseif opname == "Up" || opname == "↑"
    pU = emptyITensor(s)
    pU[Up] = 1.0
    return pU
  elseif opname == "Dn" || opname == "↓"
    pD = emptyITensor(s)
    pD[Dn] = 1.0
    return pD
  elseif opname == "UpDn" || opname == "↑↓"
    pUD = emptyITensor(s)
    pUD[UpDn] = 1.0
    return pUD
  else
    throw(ArgumentError("Operator name $opname not recognized for \"Electron\" site"))
  end
  return Op
end

has_fermion_string(::SiteType"Electron",::OpName"Cup") = true
has_fermion_string(::SiteType"Electron",::OpName"Cdagup") = true
has_fermion_string(::SiteType"Electron",::OpName"Cdn") = true
has_fermion_string(::SiteType"Electron",::OpName"Cdagdn") = true
