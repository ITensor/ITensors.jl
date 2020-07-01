
function space(::SiteType"S=1/2"; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  if conserve_sz
    return [QN("Sz",+1)=>1,QN("Sz",-1)=>1]
  end
  return 2
end


state(::SiteType"S=1/2",::StateName"Up") = 1
state(::SiteType"S=1/2",::StateName"Dn") = 2
state(::SiteType"S=1/2",::StateName"↑") = 1
state(::SiteType"S=1/2",::StateName"↓") = 2

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"Sz",
             s::Index)
  Op[s'=>1, s=>1] = 0.5
  Op[s'=>2, s=>2] = -0.5
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"Sᶻ",s::Index) = op!(Op,t,OpName("Sz"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"S+",
             s::Index)
  Op[s'=>1, s=>2] = 1.0
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"S⁺",s::Index) = op!(Op,t,OpName("S+"),s)
op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"Splus",s::Index) = op!(Op,t,OpName("S+"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"S-",
             s::Index)
  Op[s'=>2, s=>1] = 1.0
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"S⁻",s::Index) = op!(Op,t,OpName("S-"),s)
op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"Sminus",s::Index) = op!(Op,t,OpName("S-"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"Sx",
             s::Index)
  Op[s'=>1, s=>2] = 0.5
  Op[s'=>2, s=>1] = 0.5
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"Sˣ",s::Index) = op!(Op,t,OpName("Sx"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"iSy",
             s::Index)
  Op[s'=>1, s=>2] = +0.5
  Op[s'=>2, s=>1] = -0.5
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"iSʸ",s::Index) = op!(Op,t,OpName("iSy"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"Sy",
             s::Index)
  complex!(Op)
  Op[s'=>1, s=>2] = -0.5im
  Op[s'=>2, s=>1] = 0.5im
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"Sʸ",s::Index) = op!(Op,t,OpName("Sy"),s)

function op!(Op::ITensor,
             ::SiteType"S=1/2",
             ::OpName"S2",
             s::Index)
  Op[s'=>1, s=>1] = 0.75
  Op[s'=>2, s=>2] = 0.75
end

op!(Op::ITensor,t::SiteType"S=1/2",
    ::OpName"S²",s::Index) = op!(Op,t,OpName("S2"),s)


# Support the tag "SpinHalf" as equivalent to "S=1/2"

space(::SiteType"SpinHalf"; kwargs...) = space(SiteType("S=1/2");kwargs...)
state(::SiteType"SpinHalf",n::StateName) = state(SiteType("S=1/2"),n)

op!(Op::ITensor,
    ::SiteType"SpinHalf",
    o::OpName,
    s::Index) = op!(Op,SiteType("S=1/2"),o,s)

