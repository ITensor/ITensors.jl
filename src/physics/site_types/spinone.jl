
"""
    space(::SiteType"S=1";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          qnname_sz = "Sz")

Create the Hilbert space for a site of type "S=1".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function space(::SiteType"S=1";
               conserve_qns = false,
               conserve_sz = conserve_qns,
               qnname_sz = "Sz")
  if conserve_sz
    return [QN(qnname_sz,+2)=>1,
            QN(qnname_sz,0)=>1,
            QN(qnname_sz,-2)=>1]
  end
  return 3
end

state(::SiteType"S=1",::StateName"Up") = 1
state(::SiteType"S=1",::StateName"Z0") = 2
state(::SiteType"S=1",::StateName"Dn") = 3

state(st::SiteType"S=1",::StateName"↑") = state(st,StateName("Up"))
state(st::SiteType"S=1",::StateName"0") = state(st,StateName("Z0"))
state(st::SiteType"S=1",::StateName"↓") = state(st,StateName("Dn"))


function op!(Op::ITensor,
             ::OpName"Sz",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = -1.0
end

op!(Op::ITensor,
    ::OpName"Sᶻ",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("Sz"), t, s)

function op!(Op::ITensor,
             ::OpName"S+",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>2,s=>3] = sqrt(2)
  Op[s'=>1,s=>2] = sqrt(2)
end

op!(Op::ITensor,
    ::OpName"S⁺",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("S+"), t, s)

op!(Op::ITensor,
    ::OpName"Splus",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("S+"), t, s)

function op!(Op::ITensor,
             ::OpName"S-",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>3,s=>2] = sqrt(2)
  Op[s'=>2,s=>1] = sqrt(2)
end

op!(Op::ITensor,
    ::OpName"S⁻",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("S-"), t, s)

op!(Op::ITensor,
    ::OpName"Sminus",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("S-"), t, s)

function op!(Op::ITensor,
             ::OpName"Sx",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>2,s=>1] = 1/sqrt(2)
  Op[s'=>1,s=>2] = 1/sqrt(2)
  Op[s'=>3,s=>2] = 1/sqrt(2)
  Op[s'=>2,s=>3] = 1/sqrt(2)
end

op!(Op::ITensor,
    ::OpName"Sˣ",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("Sx"), t, s)

function op!(Op::ITensor,
             ::OpName"iSy",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>2,s=>1] = -1/sqrt(2)
  Op[s'=>1,s=>2] = +1/sqrt(2)
  Op[s'=>3,s=>2] = -1/sqrt(2)
  Op[s'=>2,s=>3] = +1/sqrt(2)
end

op!(Op::ITensor,
    ::OpName"iSʸ",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("iSy"), t, s)

function op!(Op::ITensor,
             ::OpName"Sy",
             ::SiteType"S=1",
             s::Index)
  complex!(Op)
  Op[s'=>2,s=>1] = -1im/sqrt(2)
  Op[s'=>1,s=>2] = +1im/sqrt(2)
  Op[s'=>3,s=>2] = -1im/sqrt(2)
  Op[s'=>2,s=>3] = +1im/sqrt(2)
end

op!(Op::ITensor,
    ::OpName"Sʸ",
    t::SiteType"S=1",
    s::Index) = op!(Op, OpName("Sy"), t, s)

function op!(Op::ITensor,
             ::OpName"Sz2",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>1,s=>1] = +1.0
  Op[s'=>3,s=>3] = +1.0
end

function op!(Op::ITensor,
             ::OpName"Sx2",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>1,s=>1] = 0.5
  Op[s'=>3,s=>1] = 0.5
  Op[s'=>2,s=>2] = 1.0
  Op[s'=>1,s=>3] = 0.5
  Op[s'=>3,s=>3] = 0.5
end

function op!(Op::ITensor,
             ::OpName"Sy2",
             ::SiteType"S=1",
             s::Index)
  Op[s'=>1,s=>1] = +0.5
  Op[s'=>3,s=>1] = -0.5
  Op[s'=>2,s=>2] = +1.0
  Op[s'=>1,s=>3] = -0.5
  Op[s'=>3,s=>3] = +0.5
end

space(::SiteType"SpinOne"; kwargs...) =
  space(SiteType("S=1");kwargs...)

state(::SiteType"SpinOne",
      st::AbstractString) = state(SiteType("S=1"), st)

op!(Op::ITensor,
    o::OpName,
    ::SiteType"SpinOne",
    s::Index) = op!(Op, o, SiteType("S=1"), s)
