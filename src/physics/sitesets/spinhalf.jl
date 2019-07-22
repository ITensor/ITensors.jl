export SpinHalfSite,
       spinHalfSites

struct SpinHalfSite <: Site
    s::Index
    SpinHalfSite(I::Index) = new(I)
end
SpinHalfSite(n::Int) = SpinHalfSite(Index(2, "Site,S=1/2,n=$n"))

function spinHalfSites(N::Int)::SiteSet
  sites = SiteSet(N)
  for n=1:N
    set(sites,n,SpinHalfSite(n))
  end
  return sites
end

function operator(site::SpinHalfSite, 
                  opname::AbstractString)::ITensor
    s = site.s
    sP = prime(site.s)
    Up = s(1)
    UpP = sP(1)
    Dn = s(2)
    DnP = sP(2)
   
    Op = ITensor(dag(s), s')

    if opname == "S⁺" || opname == "Splus" || opname == "S+"
        Op[Dn, UpP] = 1.
    elseif opname == "S⁻" || opname == "Sminus" || opname == "S-"
        Op[Up, DnP] = 1.
    elseif opname == "Sˣ" || opname == "Sx"
        Op[Up, DnP] = 0.5
        Op[Dn, UpP] = 0.5
    elseif opname == "iSʸ" || opname == "iSy"
        Op[Up, DnP] = -0.5
        Op[Dn, UpP] = 0.5
    elseif opname == "Sʸ" || opname == "Sy"
        Op[Up, DnP] = 0.5*im
        Op[Dn, UpP] = -0.5*im
    elseif opname == "Sᶻ" || opname == "Sz"
        Op[Up, UpP] = 0.5
        Op[Dn, DnP] = -0.5
    elseif opname == "projUp"
        Op[Up, UpP] = 1.
    elseif opname == "projDn"
        Op[Dn, DnP] = 1.
    elseif opname == "Up" || opname == "↑"
        pU = ITensor(s)
        pU[Up] = 1.
        return pU
    elseif opname == "Dn" || opname == "↓"
        pD = ITensor(s)
        pD[Dn] = 1.
        return pD
    else
      error("Operator name '$opname' not recognized for SpinHalfSite")
    end
    return Op
end
