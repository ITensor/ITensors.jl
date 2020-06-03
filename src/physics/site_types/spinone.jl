
function siteinds(::SiteType"S=1",
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

siteinds(::SiteType"SpinOne",
         N::Int; kwargs...) = siteinds(SiteType("S=1"),N;kwargs...)

function state(::SiteType"S=1",
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

state(::SiteType"SpinOne",st::AbstractString) = state(SiteType("S=1"),st)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sz",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"S+",
             s::Index)
  Op[s'=>2,s=>3] = sqrt(2)
  Op[s'=>1,s=>2] = sqrt(2)
end

op!(Op::ITensor,
    tt::SiteType"S=1",
    on::OpName"Splus",
    s::Index) = op!(Op,tt,OpName("S+"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"S-",
             s::Index)
  Op[s'=>3,s=>2] = sqrt(2)
  Op[s'=>2,s=>1] = sqrt(2)
end

op!(Op::ITensor,
    tt::SiteType"S=1",
    on::OpName"Sminus",
    s::Index) = op!(Op,tt,OpName("S-"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sx",
             s::Index)
  Op[s'=>2,s=>1] = 1/sqrt(2)
  Op[s'=>1,s=>2] = 1/sqrt(2)
  Op[s'=>3,s=>2] = 1/sqrt(2)
  Op[s'=>2,s=>3] = 1/sqrt(2)
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"iSy",
             s::Index)
  Op[s'=>2,s=>1] = -1/sqrt(2)
  Op[s'=>1,s=>2] = +1/sqrt(2)
  Op[s'=>3,s=>2] = -1/sqrt(2)
  Op[s'=>2,s=>3] = +1/sqrt(2)
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sy",
             s::Index)
  complex!(Op)
  Op[s'=>2,s=>1] = -1im/sqrt(2)
  Op[s'=>1,s=>2] = +1im/sqrt(2)
  Op[s'=>3,s=>2] = -1im/sqrt(2)
  Op[s'=>2,s=>3] = +1im/sqrt(2)
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sz2",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = +1.0
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sx2",
             s::Index)
  Op[s'=>1,s=>1] = 0.5
  Op[s'=>3,s=>1] = 0.5
  Op[s'=>2,s=>2] = 1.0
  Op[s'=>1,s=>3] = 0.5
  Op[s'=>3,s=>3] = 0.5
end

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sy2",
             s::Index)
  Op[s'=>1,s=>1] = +0.5
  Op[s'=>3,s=>1] = -0.5
  Op[s'=>2,s=>2] = +1.0
  Op[s'=>1,s=>3] = -0.5
  Op[s'=>3,s=>3] = +0.5
end

op!(Op::ITensor,
    ::SiteType"SpinOne",
    o::OpName,
    s::Index) = op!(Op,SiteType("S=1"),o,s)
