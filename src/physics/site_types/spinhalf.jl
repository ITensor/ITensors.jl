export SpinHalfSite

const SpinHalfSite = Union{TagType"S=1/2", TagType"SpinHalf"}

function siteinds(::SpinHalfSite,
                  N::Int; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  if conserve_sz
    return [Index(QN("Sz",+1)=>1,QN("Sz",-1)=>1;tags="Site,S=1/2,n=$n") for n=1:N]
  end
  return [Index(2,"Site,S=1/2,n=$n") for n=1:N]
end

function state(::SpinHalfSite,
               st::AbstractString)
  if st == "Up" || st == "↑"
    return 1
  elseif st == "Dn" || st == "↓"
    return 2
  end
  throw(ArgumentError("State string \"$st\" not recognized for SpinHalf site"))
  return 0
end

function op(::SpinHalfSite,
            s::Index,
            opname::AbstractString;kwargs...)::ITensor
  sP = prime(s)
  Up = s(1)
  UpP = sP(1)
  Dn = s(2)
  DnP = sP(2)
 
  Op = ITensor(s',dag(s))

  if opname == "S⁺" || opname == "Splus" || opname == "S+"
    Op[UpP, Dn] = 1.
  elseif opname == "S⁻" || opname == "Sminus" || opname == "S-"
    Op[DnP, Up] = 1.
  elseif opname == "Sˣ" || opname == "Sx"
    Op[UpP, Dn] = 0.5
    Op[DnP, Up] = 0.5
  elseif opname == "iSʸ" || opname == "iSy"
     Op[UpP, Dn] = 0.5
     Op[DnP, Up] = -0.5
  elseif opname == "Sʸ" || opname == "Sy"
     Op = complex(Op) 
     Op[UpP, Dn] = -0.5*im
     Op[DnP, Up] = 0.5*im
  elseif opname == "Sᶻ" || opname == "Sz"
     Op[UpP, Up] = 0.5
     Op[DnP, Dn] = -0.5
  elseif opname == "projUp"
     Op[UpP, Up] = 1.
  elseif opname == "projDn"
    Op[DnP, Dn] = 1.
  elseif opname == "Up" || opname == "↑"
    pU = ITensor(s)
    pU[Up] = 1.
    return pU
  elseif opname == "Dn" || opname == "↓"
    pD = ITensor(s)
    pD[Dn] = 1.
    return pD
  else
    throw(ArgumentError("Operator name '$opname' not recognized for SpinHalfSite"))
  end
  return Op
end
