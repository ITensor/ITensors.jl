
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


function op!(::TagType"S=1/2",
             ::Union{OpName"Sz"},
             Op::ITensor,
             s::Index)
  Op[s'=>1, s=>1] = 0.5
  Op[s'=>2, s=>2] = -0.5
end

function op!(::TagType"S=1/2",
             ::Union{OpName"S+",OpName"Splus"},
             Op::ITensor,
             s::Index)
  Op[s'=>1, s=>2] = 1.0
end

function op!(::TagType"S=1/2",
             ::Union{OpName"S-",OpName"Sminus"},
             Op::ITensor,
             s::Index)
  Op[s'=>2, s=>1] = 1.0
end

function op!(::TagType"S=1/2",
             ::Union{OpName"Sx"},
             Op::ITensor,
             s::Index)
  Op[s'=>1, s=>2] = 0.5
  Op[s'=>2, s=>1] = 0.5
end

function op!(::TagType"S=1/2",
             ::Union{OpName"iSy"},
             Op::ITensor,
             s::Index)
  Op[s'=>1, s=>2] = +0.5
  Op[s'=>2, s=>1] = -0.5
end

function op!(::TagType"S=1/2",
             ::Union{OpName"Sy"},
             Op::ITensor,
             s::Index)
  complex!(Op)
  Op[s'=>1, s=>2] = -0.5im
  Op[s'=>2, s=>1] = 0.5im
end

op!(::TagType"SpinHalf",
    o::AbstractOpName,
    Op::ITensor,
    s::Index) = op!(TagType"S=1/2"(),o,Op,s)

