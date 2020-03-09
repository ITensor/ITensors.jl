export SpinOneSite

const SpinOneSite = Union{TagType"S=1", TagType"SpinOne"}

function siteinds(::SpinOneSite,
                  N::Int; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  if conserve_sz
    up = QN("Sz",+2) => 1
    z0 = QN("Sz", 0) => 1
    dn = QN("Sz",-2) => 1
    return [Index(up,z0,dn;tags="Site,S=1,n=$n") for n=1:N]
  end
  return [Index(3,"Site,S=1,n=$n") for n=1:N]
end

function state(::SpinOneSite,
               st::AbstractString)
  if st == "Up" || st == "↑"
    return 1
  elseif st == "Z0" || st == "0"
    return 2
  elseif st == "Dn" || st == "↓"
    return 3
  end
  throw(ArgumentError("State string \"$st\" not recognized for SpinOne site"))
  return 0
end

function op(::SpinOneSite,
            s::Index,
            opname::AbstractString)::ITensor
  sP = prime(s)
  Up = s(1)
  UpP = sP(1)
  Z0 = s(2)
  Z0P = sP(2)
  Dn = s(3)
  DnP = sP(3)
 
  Op = ITensor(s',dag(s))

  if opname == "S⁺" || opname == "Splus" || opname == "S+"
    Op[Z0P, Dn] = √2 
    Op[UpP, Z0] = √2 
  elseif opname == "S⁻" || opname == "Sminus" || opname == "S-"
    Op[Z0P, Up] = √2 
    Op[DnP, Z0] = √2 
  elseif opname == "Sˣ" || opname == "Sx"
    Op[Z0P, Up] = 1.0/√2
    Op[UpP, Z0] = 1.0/√2
    Op[DnP, Z0] = 1.0/√2
    Op[Z0P, Dn] = 1.0/√2
  elseif opname == "iSʸ" || opname == "iSy"
    Op[Z0P, Up] = -1.0/√2
    Op[UpP, Z0] = +1.0/√2
    Op[DnP, Z0] = -1.0/√2
    Op[Z0P, Dn] = +1.0/√2
  elseif opname == "Sʸ" || opname == "Sy"
    Op = complex(Op) 
    Op[Z0P, Up] = +1.0/√2im
    Op[UpP, Z0] = -1.0/√2im
    Op[DnP, Z0] = +1.0/√2im
    Op[Z0P, Dn] = -1.0/√2im
  elseif opname == "Sᶻ" || opname == "Sz"
    Op[UpP, Up] = 1.0 
    Op[DnP, Dn] = -1.0
  elseif opname == "Sᶻ²" || opname == "Sz2"
    Op[UpP, Up] = 1.0 
    Op[DnP, Dn] = 1.0
  elseif opname == "Sˣ²" || opname == "Sx2"
    Op[UpP, Up] = 0.5
    Op[DnP, Up] = 0.5
    Op[Z0P, Z0] = 1.0 
    Op[UpP, Dn] = 0.5 
    Op[DnP, Dn] = 0.5 
  elseif opname == "Sʸ²" || opname == "Sy2"
    Op[UpP, Up] = 0.5
    Op[DnP, Up] = -0.5
    Op[Z0P, Z0] = 1.0 
    Op[UpP, Dn] = -0.5 
    Op[DnP, Dn] = 0.5 
  elseif opname == "projUp"
    Op[UpP, Up] = 1.
  elseif opname == "projZ0"
    Op[Z0P, Z0] = 1.
  elseif opname == "projDn"
    Op[DnP, Dn] = 1.
  elseif opname == "XUp"
    xup = ITensor(ComplexF64,s)
    xup[Up] = 0.5
    xup[Z0] = im*√2
    xup[Dn] = 0.5
    return xup
  elseif opname == "XZ0"
    xZ0 = ITensor(ComplexF64,s)
    xZ0[Up] = im*√2
    xZ0[Dn] = -im*√2
    return xZ0
  elseif opname == "XDn"
    xdn = ITensor(ComplexF64,s)
    xdn[Up] = 0.5
    xdn[Z0] = -im*√2
    xdn[Dn] = 0.5
    return xdn
  else
    throw(ArgumentError("Operator name '$opname' not recognized for SpinOneSite"))
  end
  return Op
end

