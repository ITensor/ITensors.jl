export ElectronSite,
       electronSites

function electronSites(N::Int; kwargs...)
  return [Index(4,"Site,Electron,n=$n") for n=1:N]
end

const ElectronSite = makeTagType("Electron")

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

  Op = ITensor(dag(s), s')

  if opname == "Nup"
    Op[Up, UpP] = 1.
    Op[UpDn, UpDnP] = 1.
  elseif opname == "Ndn"
    Op[Dn, DnP] = 1.
    Op[UpDn, UpDnP] = 1.
  elseif opname == "Ntot"
    Op[Up, UpP] = 1.
    Op[Dn, DnP] = 1.
    Op[UpDn, UpDnP] = 2.
  elseif opname == "Cup" || opname == "Aup"
    Op[Up, EmpP] = 1.
    Op[UpDn, DnP] = 1.
  elseif opname == "Cdagup" || opname == "Adagup"
    Op[Emp, UpP] = 1.
    Op[Dn, UpDnP] = 1.
  elseif opname == "Cdn" || opname == "Adn"
    Op[Dn, EmpP] = 1.
    Op[UpDn, UpP] = 1.
  elseif opname == "Cdagdn" || opname == "Adagdn"
    Op[Emp, DnP] = 1.
    Op[Up, UpDnP] = 1.
  elseif opname=="F" || opname=="FermiPhase" || opname=="FP"
    Op[Up, UpP] = -1.
    Op[Emp, EmpP] = 1.
    Op[Dn, DnP] = -1.
    Op[UpDn, UpDnP] = 1.
  elseif opname == "Fup"
    Op[Emp, EmpP] = 1.
    Op[Up, UpP] = -1.
    Op[Dn, DnP] = 1.
    Op[UpDn, UpDnP] = -1.
  elseif opname == "Fdn"
    Op[Emp, EmpP] = 1.
    Op[Up, UpP] = 1.
    Op[Dn, DnP] = -1.
    Op[UpDn, UpDnP] = -1.
  elseif opname == "Sᶻ" || opname == "Sz"
    Op[Up, UpP] = 0.5
    Op[Dn, DnP] = -0.5
  elseif opname == "Sˣ" || opname == "Sx"
    Op[Up, DnP] = 0.5
    Op[Dn, UpP] = 0.5 
  elseif opname=="S+" || opname=="Sp" || opname == "S⁺" || opname == "Splus"
    Op[Dn, UpP] = 1.0
  elseif opname=="S-" || opname=="Sm" || opname == "S⁻" || opname == "Sminus"
    Op[Up, DnP] = 1.0
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
