export tJSite

const tJSite = TagType"tJ"

function siteinds(::tJSite,
                  N::Int; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  conserve_nf = get(kwargs,:conserve_nf,conserve_qns)
  conserve_parity = get(kwargs,:conserve_parity,conserve_qns)
  if conserve_sz && conserve_nf
    em = QN(("Nf",0,-1),("Sz", 0)) => 1
    up = QN(("Nf",1,-1),("Sz",+1)) => 1
    dn = QN(("Nf",1,-1),("Sz",-1)) => 1
    return [Index(em,up,dn;tags="Site,tJ,n=$n") for n=1:N]
  elseif conserve_nf
    zer = QN("Nf",0,-1) => 1
    one = QN("Nf",1,-1) => 2
    return [Index(zer,one;tags="Site,tJ,n=$n") for n=1:N]
  elseif conserve_sz
    em = QN(("Sz", 0),("Pf",0,-2)) => 1
    up = QN(("Sz",+1),("Pf",1,-2)) => 1
    dn = QN(("Sz",-1),("Pf",1,-2)) => 1
    return [Index(em,up,dn;tags="Site,tJ,n=$n") for n=1:N]
  elseif conserve_parity
    zer = QN("Pf",0,-2) => 1
    one = QN("Pf",1,-2) => 2
    return [Index(zer,one;tags="Site,tJ,n=$n") for n=1:N]
  end
  return [Index(3,"Site,tJ,n=$n") for n=1:N]
end

function state(::tJSite,
               st::AbstractString)
  if st == "0" || st == "Emp"
    return 1
  elseif st == "Up" || st == "↑"
    return 2
  elseif st == "Dn" || st == "↓"
    return 3
  end
  throw(ArgumentError("State string \"$st\" not recognized for tJ site"))
  return 0
end

function op(::tJSite,
            s::Index,
            opname::AbstractString)::ITensor
  sP = prime(s)
  Emp = s(1)
  EmpP = sP(1)
  Up = s(2)
  UpP = sP(2)
  Dn = s(3)
  DnP = sP(3)

  Op = ITensor(s',dag(s))
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
  elseif opname == "Emp" || opname == "0"
    pEmp = ITensor(s)
    pEmp[Emp] = 1.
    return pEmp
  elseif opname == "Up" || opname == "↑"
    pU = ITensor(s)
    pU[Up] = 1.
    return pU
  elseif opname == "Dn" || opname == "↓"
    pD = ITensor(s)
    pD[Dn] = 1.
    return pD
  else
    throw(ArgumentError("Operator name '$opname' not recognized for tJSite"))
  end
  return Op
end

