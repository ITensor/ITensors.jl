export ElectronSite,
       electronSites

struct ElectronSite <: Site
  s::Index
  ElectronSite(I::Index) = new(I)
end
ElectronSite(n::Int) = ElectronSite(Index(4,"Site,Electron,n=$n"))

function electronSites(N::Int)::SiteSet
  sites = SiteSet(N)
  for n=1:N
    set(sites,n,ElectronSite(n))
  end
  return sites
end

function operator(site::ElectronSite, 
                  opname::String)::ITensor
    s  = site.s
    sP = prime(site.s)
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
    elseif opname == "FermiPhase" || opname == "FP"
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
    elseif opname == "UpDn" || opname == "↑↓"
        pUD = ITensor(s)
        pUD[UpDn] = 1.
        return pUD
    else
      error("Operator name $opname not recognized for ElectronSite")
    end
    return Op
end
