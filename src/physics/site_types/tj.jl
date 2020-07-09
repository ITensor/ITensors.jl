
function space(::SiteType"tJ";
               conserve_qns=false,
               conserve_sz=conserve_qns,
               conserve_nf=conserve_qns,
               conserve_parity=conserve_qns)
  if conserve_sz && conserve_nf
    em = QN(("Nf",0,-1),("Sz", 0)) => 1
    up = QN(("Nf",1,-1),("Sz",+1)) => 1
    dn = QN(("Nf",1,-1),("Sz",-1)) => 1
    return [em,up,dn]
  elseif conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 2
    return [zer,one]
  elseif conserve_sz
    em = QN(("Sz", 0),("Pf",0,-2)) => 1
    up = QN(("Sz",+1),("Pf",1,-2)) => 1
    dn = QN(("Sz",-1),("Pf",1,-2)) => 1
    return [em,up,dn]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 2
    return [zer,one]
  end
  return 3
end

state(::SiteType"tJ",::StateName"Emp")  = 1
state(::SiteType"tJ",::StateName"Up")   = 2
state(::SiteType"tJ",::StateName"Dn")   = 3
state(st::SiteType"tJ",::StateName"0")    = state(st,StateName("Emp"))
state(st::SiteType"tJ",::StateName"↑")    = state(st,StateName("Up"))
state(st::SiteType"tJ",::StateName"↓")    = state(st,StateName("Dn"))

function op(::SiteType"tJ",
            s::Index,
            opname::AbstractString)::ITensor
  Emp = s(1)
  EmpP = s'(1)
  Up = s(2)
  UpP = s'(2)
  Dn = s(3)
  DnP = s'(3)

  Op = emptyITensor(s',dag(s))
  if opname == "Nup"
    Op[UpP, Up] = 1.
  elseif opname == "Ndn"
    Op[DnP, Dn] = 1.
  elseif opname == "Ntot"
    Op[UpP, Up] = 1.
    Op[DnP, Dn] = 1.
  elseif opname == "Cup" || opname == "Aup"
    Op[EmpP, Up] = 1.
  elseif opname == "Cdagup" || opname == "Adagup"
    Op[UpP, Emp] = 1.
  elseif opname == "Cdn" || opname == "Adn"
    Op[EmpP, Dn] = 1.
  elseif opname == "Cdagdn" || opname == "Adagdn"
    Op[DnP, Emp] = 1.
  elseif opname == "FermiPhase" || opname == "FP"
    Op[UpP, Up] = -1.
    Op[EmpP, Emp] = 1.
    Op[DnP, Dn] = -1.
  elseif opname == "Fup"
    Op[UpP, Up] = -1.
    Op[EmpP, Emp] = 1.
    Op[DnP, Dn] = 1.
  elseif opname == "Fdn"
    Op[UpP, Up] = 1.
    Op[EmpP, Emp] = 1.
    Op[DnP, Dn] = -1.
  elseif opname == "Sᶻ" || opname == "Sz"
    Op[UpP, Up] = 0.5
    Op[DnP, Dn] = -0.5
  elseif opname == "Sˣ" || opname == "Sx"
    Op[UpP, Dn] = 1.0
    Op[DnP, Up] = 1.0 
  elseif opname == "S⁺" || opname == "Splus"
    Op[UpP, Dn] = 1.
  elseif opname == "S⁻" || opname == "Sminus"
    Op[DnP, Up] = 1.
  else
    throw(ArgumentError("Operator name '$opname' not recognized for \"tJ\" site"))
  end
  return Op
end

has_fermion_string(::OpName"Cup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagup", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdn", ::SiteType"tJ") = true
has_fermion_string(::OpName"Cdagdn", ::SiteType"tJ") = true
