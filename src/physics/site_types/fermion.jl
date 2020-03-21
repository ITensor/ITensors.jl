export FermionSite

const FermionSite = TagType"Fermion"

function siteinds(::FermionSite, 
                  N::Int; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_nf = get(kwargs,:conserve_nf,conserve_qns)
  conserve_parity = get(kwargs,:conserve_parity,conserve_qns)
  if conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 1
    return [Index(zer,one;tags="Site,Fermion,n=$n") for n=1:N]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 1
    return [Index(zer,one;tags="Site,Fermion,n=$n") for n=1:N]
  end
  return [Index(2,"Site,Fermion,n=$n") for n=1:N]
end

function state(::FermionSite,
               st::AbstractString)
  if st == "Emp" || st == "0"
    return 1
  elseif st == "Occ" || st == "1"
    return 2
  end
  throw(ArgumentError("State string \"$st\" not recognized for Fermion site"))
  return 0
end

function op(::FermionSite,
            s::Index,
            opname::AbstractString)::ITensor
  sP = prime(s)
  Emp   = s(1)
  EmpP  = sP(1)
  Occ   = s(2)
  OccP  = sP(2)

  Op = ITensor(s',dag(s))

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
    pEmp = ITensor(s)
    pEmp[Emp] = 1.0
    return pEmp
  elseif opname == "Occ" || opname == "1"
    pOcc = ITensor(s)
    pOcc[Occ] = 1.0
    return pOcc
  else
    throw(ArgumentError("Operator name $opname not recognized for FermionSite"))
  end
  return Op
end
