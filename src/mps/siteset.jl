
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
struct tJSite{N} <: Site
    s::Index
    tJSite{N}(is::Index) where N = new{N}(is)
end
function tJSite{Val{1//2}}(n::Int)
    # handle QN stuff later
    # index size 3 bc empty site is possible
    tJSite{Val{1//2}}(Index(rand(IDType), 3, Out, "Site,tJ,n=$n"))
end
struct HubbardSite{N} <: Site
    s::Index
    HubbardSite{N}(is::Index) where N = new{N}(is)
end
function HubbardSite{Val{1//2}}(n::Int)
    # handle QN stuff later
    # index size 4 bc empty site and doublon are possible
    HubbardSite{Val{1//2}}(Index(rand(IDType), 4, Out, "Site,Hubbard,n=$n"))
end

function op(site::HubbardSite{Val{1//2}}, 
            opname::String; store_type::DataType=Float64)
    s = site.s
    sP = prime(site.s)
    Emp = site.s(1);
    EmpP = sP(1);
    Up = site.s(2);
    UpP = sP(2);
    Dn = site.s(3);
    DnP = sP(3);
    UpDn = site.s(4);
    UpDnP = sP(4);
    if opname == "Nup"
        Nu = ITensor(store_type, dag(s), s')
        Nu[Up, UpP] = 1.
        Nu[UpDn, UpDnP] = 1.
        return Nup
    elseif opname == "Ndn"
        Nd = ITensor(store_type, dag(s), s')
        Nd[Dn, DnP] = 1.
        Nd[UpDn, UpDnP] = 1.
        return Nd
    elseif opname == "Ntot"
        Nt = ITensor(store_type, dag(s), s')
        Nt[Up, UpP] = 1.
        Nt[Dn, DnP] = 1.
        Nt[UpDn, UpDnP] = 2.
        return Nt
    elseif opname == "Cup" || opname == "Aup"
        Cu = ITensor(store_type, dag(s), s')
        Cu[Up, EmpP] = 1.
        Cu[UpDn, DnP] = 1.
        return Cu
    elseif opname == "Cdagup" || opname == "Adagup"
        Cu = ITensor(store_type, dag(s), s')
        Cu[Emp, UpP] = 1.
        Cu[Dn, UpDnP] = 1.
        return Cu
    elseif opname == "Cdn" || opname == "Adn"
        Cd = ITensor(store_type, dag(s), s')
        Cd[Dn, EmpP] = 1.
        Cd[UpDn, UpP] = 1.
        return Cd
    elseif opname == "Cdagdn" || opname == "Adagdn"
        Cd = ITensor(store_type, dag(s), s')
        Cd[Emp, DnP] = 1.
        Cd[Up, UpDnP] = 1.
        return Cd
    elseif opname == "FermiPhase" || opname == "FP"
        FP = ITensor(store_type, dag(s), s')
        FP[Up, UpP] = -1.
        FP[Emp, EmpP] = 1.
        FP[Dn, DnP] = -1.
        FP[UpDn, UpDnP] = 1.
        return FP
    elseif opname == "Fup"
        FUp = ITensor(store_type, dag(s), s')
        FUp[Emp, EmpP] = 1.
        FUp[Up, UpP] = -1.
        FUp[Dn, DnP] = 1.
        FUp[UpDn, UpDnP] = -1.
        return FUp
    elseif opname == "Fdn"
        FDn = ITensor(store_type, dag(s), s')
        FDn[Emp, EmpP] = 1.
        FDn[Up, UpP] = 1.
        FDn[Dn, DnP] = -1.
        FDn[UpDn, UpDnP] = -1.
        return FDn
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(store_type, dag(s), s')
        Sᶻ[Up, UpP] = 0.5
        Sᶻ[Dn, DnP] = -0.5
        return Sᶻ
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(store_type, dag(s), s')
        Sˣ[Up, DnP] = 1.0
        Sˣ[Dn, UpP] = 1.0 
        return Sˣ
    elseif opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(store_type, dag(s), s')
        S⁺[Dn, UpP] = 1.
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(store_type, dag(s), s')
        S⁻[Up, DnP] = 1.
        return S⁻
    elseif opname == "Emp" || opname == "0"
        pEmp = ITensor(store_type, s)
        pEmp[Emp] = 1.
        return pEmp
    elseif opname == "Up" || opname == "↑"
        pU = ITensor(store_type, s)
        pU[Up] = 1.
        return pU
    elseif opname == "Dn" || opname == "↓"
        pD = ITensor(store_type, s)
        pD[Dn] = 1.
        return pD
    elseif opname == "UpDn" || opname == "↑↓"
        pUD = ITensor(store_type, s)
        pUD[UpDn] = 1.
        return pUD
    end
end

function op(site::tJSite{Val{1//2}}, opname::String; store_type::DataType=Float64)
    s = site.s
    sP = prime(site.s)
    Emp = site.s(1);
    EmpP = sP(1);
    Up = site.s(2);
    UpP = sP(2);
    Dn = site.s(3);
    DnP = sP(3);
    if opname == "Nup"
        Nu = ITensor(store_type, dag(s), s')
        Nu[Up, UpP] = 1.
        return Nup
    elseif opname == "Ndn"
        Nd = ITensor(store_type, dag(s), s')
        Nd[Dn, DnP] = 1.
        return Nd
    elseif opname == "Ntot"
        Nt = ITensor(store_type, dag(s), s')
        Nt[Up, UpP] = 1.
        Nt[Dn, DnP] = 1.
        return Nt
    elseif opname == "Cup" || opname == "Aup"
        Cu = ITensor(store_type, dag(s), s')
        Cu[Up, EmpP] = 1.
        return Cu
    elseif opname == "Cdagup" || opname == "Adagup"
        Cu = ITensor(store_type, dag(s), s')
        Cu[Emp, UpP] = 1.
        return Cu
    elseif opname == "Cdn" || opname == "Adn"
        Cd = ITensor(store_type, dag(s), s')
        Cd[Dn, EmpP] = 1.
        return Cd
    elseif opname == "Cdagdn" || opname == "Adagdn"
        Cd = ITensor(store_type, dag(s), s')
        Cd[Emp, DnP] = 1.
        return Cd
    elseif opname == "FermiPhase" || opname == "FP"
        FP = ITensor(store_type, dag(s), s')
        FP[Up, UpP] = -1.
        FP[Emp, EmpP] = 1.
        FP[Dn, DnP] = -1.
        return FP
    elseif opname == "Fup"
        FUp = ITensor(store_type, dag(s), s')
        FUp[Up, UpP] = -1.
        FUp[Emp, EmpP] = 1.
        FUp[Dn, DnP] = 1.
        return FUp
    elseif opname == "Fdn"
        FDn = ITensor(store_type, dag(s), s')
        FDn[Up, UpP] = 1.
        FDn[Emp, EmpP] = 1.
        FDn[Dn, DnP] = -1.
        return FDn
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(store_type, dag(s), s')
        Sᶻ[Up, UpP] = 0.5
        Sᶻ[Dn, DnP] = -0.5
        return Sᶻ
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(store_type, dag(s), s')
        Sˣ[Up, DnP] = 1.0
        Sˣ[Dn, UpP] = 1.0 
        return Sˣ
    elseif opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(store_type, dag(s), s')
        S⁺[Dn, UpP] = 1.
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(store_type, dag(s), s')
        S⁻[Up, DnP] = 1.
        return S⁻
    elseif opname == "Emp" || opname == "0"
        pEmp = ITensor(store_type, s)
        pEmp[Emp] = 1.
        return pEmp
    elseif opname == "Up" || opname == "↑"
        pU = ITensor(store_type, s)
        pU[Up] = 1.
        return pU
    elseif opname == "Dn" || opname == "↓"
        pD = ITensor(store_type, s)
        pD[Dn] = 1.
        return pD
    end
end

function op(site::SpinSite{Val{1//2}}, opname::String; store_type::DataType=Float64)
    s = site.s
    sP = prime(site.s)
    Up = site.s(1);
    UpP = sP(1);
    Dn = site.s(2);
    DnP = sP(2);
    if opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(store_type, s, s')
        S⁺[Dn, UpP] = 1.
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(store_type, s, s')
        S⁻[Up, DnP] = 1.
        return S⁻
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(store_type, s, s')
        Sˣ[Up, DnP] = 0.5
        Sˣ[Dn, UpP] = 0.5
        return Sˣ
    elseif opname == "iSʸ" || opname == "iSy"
        Sʸ = ITensor(store_type, s, s')
        Sʸ[Up, DnP] = -0.5
        Sʸ[Dn, UpP] = 0.5
        return Sʸ
    elseif opname == "Sʸ" || opname == "Sy"
        Sʸ = ITensor(store_type, s, s')
        Sʸ[Up, DnP] = 0.5*im
        Sʸ[Dn, UpP] = -0.5*im
        return Sʸ
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(store_type, s, s')
        Sᶻ[Up, UpP] = 0.5
        Sᶻ[Dn, DnP] = -0.5
        return Sᶻ
    elseif opname == "projUp"
        pU = ITensor(store_type, s, s')
        pU[Up, UpP] = 1.
        return pU
    elseif opname == "projDn"
        pD = ITensor(store_type, s, s')
        pD[Dn, DnP] = 1.
        return pD
    elseif opname == "Up" || opname == "↑"
        pU = ITensor(store_type, s)
        pU[Up] = 1.
        return pU
    elseif opname == "Dn" || opname == "↓"
        pD = ITensor(store_type, s)
        pD[Dn] = 1.
        return pD
    end
end

function op(site::SpinSite{Val{1}}, opname::String; store_type::DataType=Float64)
    s = site.s
    sP = prime(site.s)
    Up = site.s(1);
    UpP = sP(1);
    Z0 = site.s(2);
    Z0P = sP(2);
    Dn = site.s(3);
    DnP = sP(3);
    if opname == "S⁺" || opname == "Splus"
        S⁺ = ITensor(store_type, s, s')
        S⁺[Dn, Z0P] = √2 
        S⁺[Z0, UpP] = √2 
        return S⁺
    elseif opname == "S⁻" || opname == "Sminus"
        S⁻ = ITensor(store_type, s, s')
        S⁻[Up, Z0P] = √2 
        S⁻[Z0, DnP] = √2 
        return S⁻
    elseif opname == "Sˣ" || opname == "Sx"
        Sˣ = ITensor(store_type, s, s')
        Sˣ[Up, Z0P] = im*√2
        Sˣ[Z0, UpP] = im*√2
        Sˣ[Z0, DnP] = im*√2
        Sˣ[Dn, Z0P] = im*√2
        return Sˣ
    elseif opname == "iSʸ" || opname == "iSy"
        Sʸ = ITensor(store_type, s, s')
        Sʸ[Up, Z0P] = -im*√2
        Sʸ[Z0, UpP] = im*√2
        Sʸ[Z0, DnP] = -im*√2
        Sʸ[Dn, Z0P] = im*√2
        return Sʸ
    elseif opname == "Sʸ" || opname == "Sy"
        Sʸ = ITensor(store_type, s, s')
        Sʸ[Up, Z0P] = -√2
        Sʸ[Z0, UpP] = √2
        Sʸ[Z0, DnP] = -√2
        Sʸ[Dn, Z0P] = √2
        return Sʸ
    elseif opname == "Sᶻ" || opname == "Sz"
        Sᶻ = ITensor(store_type, s, s')
        Sᶻ[Up, UpP] = 1.0 
        Sᶻ[Dn, DnP] = -1.0
        return Sᶻ
    elseif opname == "Sᶻ²" || opname == "Sz2"
        Sᶻ = ITensor(store_type, s, s')
        Sᶻ[Up, UpP] = 1.0 
        Sᶻ[Dn, DnP] = 1.0
        return Sᶻ
    elseif opname == "Sˣ²" || opname == "Sx2"
        Sˣ = ITensor(store_type, s, s')
        Sˣ[Up, UpP] = 0.5
        Sˣ[Up, DnP] = 0.5
        Sˣ[Z0, Z0P] = 1.0 
        Sˣ[Dn, UpP] = 0.5 
        Sˣ[Dn, DnP] = 0.5 
        return Sˣ
    elseif opname == "Sʸ²" || opname == "Sy2"
        Sʸ = ITensor(store_type, s, s')
        Sʸ[Up, UpP] = 0.5
        Sʸ[Up, DnP] = -0.5
        Sʸ[Z0, Z0P] = 1.0 
        Sʸ[Dn, UpP] = -0.5 
        Sʸ[Dn, DnP] = 0.5 
        return Sᶻ
    elseif opname == "projUp"
        pU = ITensor(store_type, s, s')
        pU[Up, UpP] = 1.
        return pU
    elseif opname == "projZ0"
        pZ = ITensor(store_type, s, s')
        pZ[Z0, Z0P] = 1.
        return Z0 
    elseif opname == "projDn"
        pD = ITensor(store_type, s, s')
        pD[Dn, DnP] = 1.
        return pD
    elseif opname == "XUp"
        xup = ITensor(store_type, s)
        xup[Up] = 0.5
        xup[Z0] = im*√2
        xup[Dn] = 0.5
        return xup
    elseif opname == "XZ0"
        xZ0 = ITensor(store_type, s)
        xZ0[Up] = im*√2
        xZ0[Dn] = -im*√2
        return xZ0
    elseif opname == "XDn"
        xdn = ITensor(store_type, s)
        xdn[Up] = 0.5
        xdn[Z0] = -im*√2
        xdn[Dn] = 0.5
        return xdn
    end
end
