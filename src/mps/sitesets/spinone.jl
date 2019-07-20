export SpinOneSite,
       spinOneSites

struct SpinOneSite <: Site
    s::Index
    SpinOneSite(I::Index) = new(I)
end
SpinOneSite(n::Int) = SpinOneSite(Index(3, "Site,S=1,n=$n"))

function spinOneSites(N::Int)::SiteSet
  sites = SiteSet(N)
  for n=1:N
    set(sites,n,SpinOneSite(n))
  end
  return sites
end

function operator(site::SpinOneSite, 
                  opname::AbstractString)::ITensor
    s = site.s
    sP = prime(site.s)
    Up = s(1)
    UpP = sP(1)
    Z0 = s(2)
    Z0P = sP(2)
    Dn = s(3)
    DnP = sP(3)
   
    Op = ITensor(dag(s), s')

    if opname == "S⁺" || opname == "Splus" || opname == "S+"
        Op[Dn, Z0P] = √2 
        Op[Z0, UpP] = √2 
    elseif opname == "S⁻" || opname == "Sminus" || opname == "S-"
        Op[Up, Z0P] = √2 
        Op[Z0, DnP] = √2 
    elseif opname == "Sˣ" || opname == "Sx"
        Op[Up, Z0P] = im*√2
        Op[Z0, UpP] = im*√2
        Op[Z0, DnP] = im*√2
        Op[Dn, Z0P] = im*√2
    elseif opname == "iSʸ" || opname == "iSy"
        Op[Up, Z0P] = -im*√2
        Op[Z0, UpP] = im*√2
        Op[Z0, DnP] = -im*√2
        Op[Dn, Z0P] = im*√2
    elseif opname == "Sʸ" || opname == "Sy"
        Op[Up, Z0P] = -√2
        Op[Z0, UpP] = √2
        Op[Z0, DnP] = -√2
        Op[Dn, Z0P] = √2
    elseif opname == "Sᶻ" || opname == "Sz"
        Op[Up, UpP] = 1.0 
        Op[Dn, DnP] = -1.0
    elseif opname == "Sᶻ²" || opname == "Sz2"
        Op[Up, UpP] = 1.0 
        Op[Dn, DnP] = 1.0
    elseif opname == "Sˣ²" || opname == "Sx2"
        Op[Up, UpP] = 0.5
        Op[Up, DnP] = 0.5
        Op[Z0, Z0P] = 1.0 
        Op[Dn, UpP] = 0.5 
        Op[Dn, DnP] = 0.5 
    elseif opname == "Sʸ²" || opname == "Sy2"
        Op[Up, UpP] = 0.5
        Op[Up, DnP] = -0.5
        Op[Z0, Z0P] = 1.0 
        Op[Dn, UpP] = -0.5 
        Op[Dn, DnP] = 0.5 
    elseif opname == "projUp"
        Op[Up, UpP] = 1.
    elseif opname == "projZ0"
        Op[Z0, Z0P] = 1.
    elseif opname == "projDn"
        Op[Dn, DnP] = 1.
    elseif opname == "XUp"
        xup = ITensor(s)
        xup[Up] = 0.5
        xup[Z0] = im*√2
        xup[Dn] = 0.5
        return xup
    elseif opname == "XZ0"
        xZ0 = ITensor(s)
        xZ0[Up] = im*√2
        xZ0[Dn] = -im*√2
        return xZ0
    elseif opname == "XDn"
        xdn = ITensor(s)
        xdn[Up] = 0.5
        xdn[Z0] = -im*√2
        xdn[Dn] = 0.5
        return xdn
    else
      error("Operator name '$opname' not recognized for SpinOneSite")
    end
    return Op
end
