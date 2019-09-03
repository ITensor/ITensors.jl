export tJSite,
       tJSites

struct tJSite <: Site end

dim(::tJSite) = 3
defaultTags(::tJSite,n::Int) = TagSet("Site,tJ,n=$n")

function tJSites(N::Int;kwargs...)::SiteSet
  sites = SiteSet(N)
  for n=1:N
    setSite!(sites,n,tJSite())
  end
  return sites
end

function operator(s::Index,
                  site::tJSite, 
                  opname::AbstractString)::ITensor
  sP = prime(s)
  Emp = s(1)
  EmpP = sP(1)
  Up = s(2)
  UpP = sP(2)
  Dn = s(3)
  DnP = sP(3)

  Op = ITensor(dag(s), s')
  if opname == "Nup"
    Op[Up, UpP] = 1.
  elseif opname == "Ndn"
    Op[Dn, DnP] = 1.
  elseif opname == "Ntot"
    Op[Up, UpP] = 1.
    Op[Dn, DnP] = 1.
  elseif opname == "Cup" || opname == "Aup"
    Op[Up, EmpP] = 1.
  elseif opname == "Cdagup" || opname == "Adagup"
    Op[Emp, UpP] = 1.
  elseif opname == "Cdn" || opname == "Adn"
    Op[Dn, EmpP] = 1.
  elseif opname == "Cdagdn" || opname == "Adagdn"
    Op[Emp, DnP] = 1.
  elseif opname == "FermiPhase" || opname == "FP"
    Op[Up, UpP] = -1.
    Op[Emp, EmpP] = 1.
    Op[Dn, DnP] = -1.
  elseif opname == "Fup"
    Op[Up, UpP] = -1.
    Op[Emp, EmpP] = 1.
    Op[Dn, DnP] = 1.
  elseif opname == "Fdn"
    Op[Up, UpP] = 1.
    Op[Emp, EmpP] = 1.
    Op[Dn, DnP] = -1.
  elseif opname == "Sᶻ" || opname == "Sz"
    Op[Up, UpP] = 0.5
    Op[Dn, DnP] = -0.5
  elseif opname == "Sˣ" || opname == "Sx"
    Op[Up, DnP] = 1.0
    Op[Dn, UpP] = 1.0 
  elseif opname == "S⁺" || opname == "Splus"
    Op[Dn, UpP] = 1.
  elseif opname == "S⁻" || opname == "Sminus"
    Op[Up, DnP] = 1.
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
