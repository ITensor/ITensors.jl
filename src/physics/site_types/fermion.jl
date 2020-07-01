
function space(::SiteType"Fermion"; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_nf = get(kwargs,:conserve_nf,conserve_qns)
  conserve_parity = get(kwargs,:conserve_parity,conserve_qns)
  if conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 1
    return [zer,one]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 1
    return [zer,one]
  end
  return 2
end

state(::SiteType"Fermion",::StateName"Emp")  = 1
state(::SiteType"Fermion",::StateName"Occ")  = 2
state(::SiteType"Fermion",::StateName"0")  = 1
state(::SiteType"Fermion",::StateName"1")  = 2

function op(::SiteType"Fermion",
            s::Index,
            opname::AbstractString)::ITensor
  Emp   = s(1)
  EmpP  = s'(1)
  Occ   = s(2)
  OccP  = s'(2)

  Op = emptyITensor(s',dag(s))

  if opname == "N"
    Op[OccP, Occ] = 1.
  elseif opname == "C"
    Op[EmpP, Occ] = 1.
  elseif opname == "Cdag"
    Op[OccP, Emp] = 1.
  elseif opname=="F" || opname=="FermiPhase" || opname=="FP"
    Op[EmpP,Emp] =  1.
    Op[OccP,Occ] = -1.
  elseif opname == "Emp" || opname == "0"
    pEmp = emptyITensor(s)
    pEmp[Emp] = 1.0
    return pEmp
  elseif opname == "Occ" || opname == "1"
    pOcc = emptyITensor(s)
    pOcc[Occ] = 1.0
    return pOcc
  else
    throw(ArgumentError("Operator name $opname not recognized for \"Fermion\" site"))
  end
  return Op
end

function has_fermion_string(::SiteType"Fermion",
            s::Index,
            opname::AbstractString)::Bool
  if opname=="C" || opname=="Cdag"
    return true
  end
  return false
end
