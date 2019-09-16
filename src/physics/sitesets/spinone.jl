export SpinOneSite,
       spinOneSites

function spinOneSites(N::Int; kwargs...)
  return [Index(3,"Site,SpinOne,n=$n") for n=1:N]
end

const SpinOneSite = makeTagType("SpinOne")

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
 
  Op = ITensor(dag(s), s')

  if opname == "S⁺" || opname == "Splus" || opname == "S+"
    Op[Dn, Z0P] = √2 
    Op[Z0, UpP] = √2 
  elseif opname == "S⁻" || opname == "Sminus" || opname == "S-"
    Op[Up, Z0P] = √2 
    Op[Z0, DnP] = √2 
  elseif opname == "Sˣ" || opname == "Sx"
    Op[Up, Z0P] = 1.0/√2
    Op[Z0, UpP] = 1.0/√2
    Op[Z0, DnP] = 1.0/√2
    Op[Dn, Z0P] = 1.0/√2
  elseif opname == "iSʸ" || opname == "iSy"
    Op[Up, Z0P] = -1.0/√2
    Op[Z0, UpP] = +1.0/√2
    Op[Z0, DnP] = -1.0/√2
    Op[Dn, Z0P] = +1.0/√2
  elseif opname == "Sʸ" || opname == "Sy"
    Op = complex(Op) 
    Op[Up, Z0P] = +1.0/√2im
    Op[Z0, UpP] = -1.0/√2im
    Op[Z0, DnP] = +1.0/√2im
    Op[Dn, Z0P] = -1.0/√2im
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

