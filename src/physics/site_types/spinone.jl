
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

function op!(::TagType"S=1",
             ::OpName"Sz",
             Op::ITensor,
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

function op!(::TagType"S=1",
             ::OpName"S+",
             Op::ITensor,
             s::Index)
  Op[s'=>2,s=>3] = sqrt(2)
  Op[s'=>1,s=>2] = sqrt(2)
end

op!(tt::TagType"S=1",
    on::OpName"Splus",
    Op::ITensor,s::Index) = op!(tt,OpName("S+"),Op,s)

function op!(::TagType"S=1",
             ::OpName"S-",
             Op::ITensor,
             s::Index)
  Op[s'=>3,s=>2] = sqrt(2)
  Op[s'=>2,s=>1] = sqrt(2)
end

op!(tt::TagType"S=1",
    on::OpName"Sminus",
    Op::ITensor,s::Index) = op!(tt,OpName("S-"),Op,s)

function op!(::TagType"S=1",
             ::OpName"Sx",
             Op::ITensor,
             s::Index)
  Op[s'=>2,s=>1] = 1/sqrt(2)
  Op[s'=>1,s=>2] = 1/sqrt(2)
  Op[s'=>3,s=>2] = 1/sqrt(2)
  Op[s'=>2,s=>3] = 1/sqrt(2)
end

function op!(::TagType"S=1",
             ::OpName"iSy",
             Op::ITensor,
             s::Index)
  Op[s'=>2,s=>1] = -1/sqrt(2)
  Op[s'=>1,s=>2] = +1/sqrt(2)
  Op[s'=>3,s=>2] = -1/sqrt(2)
  Op[s'=>2,s=>3] = +1/sqrt(2)
end

function op!(::TagType"S=1",
             ::OpName"Sy",
             Op::ITensor,
             s::Index)
  complex!(Op)
  Op[s'=>2,s=>1] = -1im/sqrt(2)
  Op[s'=>1,s=>2] = +1im/sqrt(2)
  Op[s'=>3,s=>2] = -1im/sqrt(2)
  Op[s'=>2,s=>3] = +1im/sqrt(2)
end

function op!(::TagType"S=1",
             ::OpName"Sz2",
             Op::ITensor,
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = +1.0
end

function op!(::TagType"S=1",
             ::OpName"Sx2",
             Op::ITensor,
             s::Index)
  Op[s'=>1,s=>1] = 0.5
  Op[s'=>3,s=>1] = 0.5
  Op[s'=>2,s=>2] = 1.0
  Op[s'=>1,s=>3] = 0.5
  Op[s'=>3,s=>3] = 0.5
end

function op!(::TagType"S=1",
             ::OpName"Sy2",
             Op::ITensor,
             s::Index)
  Op[s'=>1,s=>1] = +0.5
  Op[s'=>3,s=>1] = -0.5
  Op[s'=>2,s=>2] = +1.0
  Op[s'=>1,s=>3] = -0.5
  Op[s'=>3,s=>3] = +0.5
end

op!(::TagType"SpinOne",
    o::OpName,
    Op::ITensor,
    s::Index) = op!(TagType("S=1"),o,Op,s)
