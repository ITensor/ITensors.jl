export tJSite

const tJSite = TagType"tJ"

function siteinds(::tJSite,
                  N::Int; kwargs...)
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

