export ElectronSite

const ElectronSite = TagType"Electron"

function siteinds(::ElectronSite, 
                  N::Int; kwargs...)
  return [Index(4,"Site,Electron,n=$n") for n=1:N]
end

function state(::ElectronSite,
               st::AbstractString)
  if st == "0" || st == "Emp"
    return 1
  elseif st == "Up" || st == "↑"
    return 2
  elseif st == "Dn" || st == "↓"
    return 3
  elseif st == "UpDn" || st == "↑↓"
    return 4
  end
  throw(ArgumentError("State string \"$st\" not recognized for Electron site"))
  return 0
end

function op(::ElectronSite,
            s::Index,
            opname::AbstractString)::ITensor
  sP = prime(s)
  Emp   = s(1)
  EmpP  = sP(1)
  Up    = s(2)
  UpP   = sP(2)
  Dn    = s(3)
  DnP   = sP(3)
  UpDn  = s(4)
  UpDnP = sP(4)

  Op = ITensor(s',dag(s))

  if opname == "Nup"
    Op[UpP, Up] = 1.
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Ndn"
    Op[DnP, Dn] = 1.
    Op[UpDnP, UpDn] = 1.
  elseif opname == "Ntot"
    Op[UpP, Up] = 1.
    Op[DnP, Dn] = 1.
    Op[UpDnP, UpDn] = 2.
  elseif opname == "Cup" || opname == "Aup"
    Op[EmpP, Up] = 1.
    Op[DnP, UpDn] = 1.
  elseif opname == "Cdagup" || opname == "Adagup"
    Op[UpP, Emp] = 1.
    Op[UpDnP, Dn] = 1.
  elseif opname == "Cdn" || opname == "Adn"
    Op[EmpP, Dn] = 1.
    Op[UpP, UpDn] = 1.
  elseif opname == "Cdagdn" || opname == "Adagdn"
    Op[DnP, Emp] = 1.
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
    pEmp = ITensor(s)
    pEmp[Emp] = 1.0
    return pEmp
  elseif opname == "Up" || opname == "↑"
    pU = ITensor(s)
    pU[Up] = 1.0
    return pU
  elseif opname == "Dn" || opname == "↓"
    pD = ITensor(s)
    pD[Dn] = 1.0
    return pD
  elseif opname == "UpDn" || opname == "↑↓"
    pUD = ITensor(s)
    pUD[UpDn] = 1.0
    return pUD
  else
    throw(ArgumentError("Operator name $opname not recognized for ElectronSite"))
  end
  return Op
end
