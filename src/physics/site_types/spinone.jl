
function space(::SiteType"S=1"; kwargs...)
  conserve_qns = get(kwargs,:conserve_qns,false)
  conserve_sz = get(kwargs,:conserve_sz,conserve_qns)
  if conserve_sz
    return [QN("Sz",+2)=>1,QN("Sz",0)=>1,QN("Sz",-2)=>1]
  end
  return 3
end

state(::SiteType"S=1",::StateName"Up") = 1
state(::SiteType"S=1",::StateName"Z0") = 2
state(::SiteType"S=1",::StateName"Dn") = 3
state(::SiteType"S=1",::StateName"↑") = 1
state(::SiteType"S=1",::StateName"0") = 2
state(::SiteType"S=1",::StateName"↓") = 3


function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sz",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"Sᶻ",s::Index) = op!(Op,t,OpName("Sz"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"S+",
             s::Index)
  Op[s'=>2,s=>3] = sqrt(2)
  Op[s'=>1,s=>2] = sqrt(2)
end

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"S⁺",s::Index) = op!(Op,t,OpName("S+"),s)
op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"Splus",s::Index) = op!(Op,t,OpName("S+"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"S-",
             s::Index)
  Op[s'=>3,s=>2] = sqrt(2)
  Op[s'=>2,s=>1] = sqrt(2)
end

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"S⁻",s::Index) = op!(Op,t,OpName("S-"),s)
op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"Sminus",s::Index) = op!(Op,t,OpName("S-"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"Sx",
             s::Index)
  Op[s'=>2,s=>1] = 1/sqrt(2)
  Op[s'=>1,s=>2] = 1/sqrt(2)
  Op[s'=>3,s=>2] = 1/sqrt(2)
  Op[s'=>2,s=>3] = 1/sqrt(2)
end

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"Sˣ",s::Index) = op!(Op,t,OpName("Sx"),s)

function op!(Op::ITensor,
             ::SiteType"S=1",
             ::OpName"iSy",
             s::Index)
  Op[s'=>2,s=>1] = -1/sqrt(2)
  Op[s'=>1,s=>2] = +1/sqrt(2)
  Op[s'=>3,s=>2] = -1/sqrt(2)
  Op[s'=>2,s=>3] = +1/sqrt(2)
end

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"iSʸ",s::Index) = op!(Op,t,OpName("iSy"),s)

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

op!(Op::ITensor,t::SiteType"S=1",
    ::OpName"Sʸ",s::Index) = op!(Op,t,OpName("Sy"),s)

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

space(::SiteType"SpinOne"; kwargs...) = space(SiteType("S=1");kwargs...)
state(::SiteType"SpinOne",st::AbstractString) = state(SiteType("S=1"),st)

op!(Op::ITensor,
    ::SiteType"SpinOne",
    o::OpName,
    s::Index) = op!(Op,SiteType("S=1"),o,s)
