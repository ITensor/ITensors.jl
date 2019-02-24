
abstract type SiteSet end

function show(io::IO,
              sites::SiteSet)
  print(io,"SiteSet")
  (length(sites) > 0) && print(io,"\n")
  for i=1:length(sites)
    println(io,"  $(sites[i])")
  end
end

length(sites::SiteSet) = length(inds(sites))
getindex(sites::SiteSet,i::Integer) = getindex(inds(sites),i)

function state(sites::SiteSet,
               n::Integer,
               st::Integer)::IndexVal
  return sites[n](st)
end

function state(sites::SiteSet,
               n::Integer,
               st::String)::IndexVal
  error("String version of 'state' SiteSet function not defined for this site set type")
  return sites[1](1)
end

struct Sites <: SiteSet
  inds::IndexSet
  Sites() = new(IndexSet())
  function Sites(N::Integer, d::Integer)
    inds_ = IndexSet(N)
    for n=1:N
      inds_[n] = Index(d,"n=$n,Site")
    end
    new(inds_)
  end
end

inds(s::Sites) = s.inds

abstract type Site end
struct SpinSite{N} <: Site
    s::Index
    SpinSite{N}(is::Index) where N = new{N}(is)
end
SpinSite{Val{1//2}}(n::Int) = SpinSite{Val{1//2}}(Index(2, "Site,S=1/2,n=$n"))
SpinSite{Val{1}}(n::Int) = SpinSite{Val{1}}(Index(3, "Site,S=1,n=$n"))

function op(site::SpinSite{Val{1//2}}, opname::String)
    s = site.s
    sP = prime(site.s)
    Up = site.s(1);
    UpP = sP(1);
    Dn = site.s(2);
    DnP = sP(2);
    if opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(s, s')
        S⁺[Dn, UpP] = 1.
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(s, s')
        S⁻[Up, DnP] = 1.
        return S⁻
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(s, s')
        Sˣ[Up, DnP] = 0.5
        Sˣ[Dn, UpP] = 0.5
        return Sˣ
    elseif opname == "iSʸ" || opname == "iSy"
        Sʸ = ITensor(s, s')
        Sʸ[Up, DnP] = -0.5
        Sʸ[Dn, UpP] = 0.5
        return Sʸ
    elseif opname == "Sʸ" || opname == "Sy"
        Sʸ = ITensor(s, s')
        Sʸ[Up, DnP] = 0.5*im
        Sʸ[Dn, UpP] = -0.5*im
        return Sʸ
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(s, s')
        Sᶻ[Up, UpP] = 0.5
        Sᶻ[Dn, DnP] = -0.5
        return Sᶻ
    elseif opname == "projUp"
        pU = ITensor(s, s')
        pU[Up, UpP] = 1.
        return pU
    elseif opname == "projDn"
        pD = ITensor(s, s')
        pD[Dn, DnP] = 1.
        return pD
    elseif opname == "Up" || opname == "↑"
        pU = ITensor(s)
        pU[Up] = 1.
        return pU
    elseif opname == "Dn" || opname == "↓"
        pD = ITensor(s)
        pD[Dn] = 1.
        return pD
    end
end

function op(site::SpinSite{Val{1}}, opname::String)
    s = site.s
    sP = prime(site.s)
    Up = site.s(1);
    UpP = sP(1);
    Z0 = site.s(2);
    Z0P = sP(2);
    Dn = site.s(3);
    DnP = sP(3);
    if opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(s, s')
        S⁺[Dn, Z0P] = √2 
        S⁺[Z0, UpP] = √2 
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(s, s')
        S⁻[Up, Z0P] = √2 
        S⁻[Z0, DnP] = √2 
        return S⁻
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(s, s')
        Sˣ[Up, Z0P] = im*√2
        Sˣ[Z0, UpP] = im*√2
        Sˣ[Z0, DnP] = im*√2
        Sˣ[Dn, Z0P] = im*√2
        return Sˣ
    elseif opname == "iSʸ" || opname == "iSy"
        Sʸ = ITensor(s, s')
        Sʸ[Up, Z0P] = -im*√2
        Sʸ[Z0, UpP] = im*√2
        Sʸ[Z0, DnP] = -im*√2
        Sʸ[Dn, Z0P] = im*√2
        return Sʸ
    elseif opname == "Sʸ" || opname == "Sy"
        Sʸ = ITensor(s, s')
        Sʸ[Up, Z0P] = -√2
        Sʸ[Z0, UpP] = √2
        Sʸ[Z0, DnP] = -√2
        Sʸ[Dn, Z0P] = √2
        return Sʸ
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(s, s')
        Sᶻ[Up, UpP] = 1.0 
        Sᶻ[Dn, DnP] = -1.0
        return Sᶻ
    elseif opname == "Sᶻ²" || opname == "Sz2"
        Sᶻ = ITensor(s, s')
        Sᶻ[Up, UpP] = 1.0 
        Sᶻ[Dn, DnP] = 1.0
        return Sᶻ
    elseif opname == "Sˣ²" || opname == "Sx2"
        Sˣ = ITensor(s, s')
        Sˣ[Up, UpP] = 0.5
        Sˣ[Up, DnP] = 0.5
        Sˣ[Z0, Z0P] = 1.0 
        Sˣ[Dn, UpP] = 0.5 
        Sˣ[Dn, DnP] = 0.5 
        return Sˣ
    elseif opname == "Sʸ²" || opname == "Sy2"
        Sʸ = ITensor(s, s')
        Sʸ[Up, UpP] = 0.5
        Sʸ[Up, DnP] = -0.5
        Sʸ[Z0, Z0P] = 1.0 
        Sʸ[Dn, UpP] = -0.5 
        Sʸ[Dn, DnP] = 0.5 
        return Sᶻ
    elseif opname == "projUp"
        pU = ITensor(s, s')
        pU[Up, UpP] = 1.
        return pU
    elseif opname == "projZ0"
        pZ = ITensor(s, s')
        pZ[Z0, Z0P] = 1.
        return Z0 
    elseif opname == "projDn"
        pD = ITensor(s, s')
        pD[Dn, DnP] = 1.
        return pD
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
    end
end
