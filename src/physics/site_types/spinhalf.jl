
function space(::SiteType"S=1/2";
               conserve_qns = false,
               conserve_sz = conserve_qns,
               conserve_szparity = false,
               qnname_sz = "Sz",
               qnname_szparity = "SzParity")
  if conserve_sz && conserve_szparity
    return [QN((qnname_sz, +1), (qnname_szparity, 1, 2)) => 1,
            QN((qnname_sz, -1), (qnname_szparity, 0, 2)) => 1]
  elseif conserve_sz
    return [QN(qnname_sz, +1) => 1,
            QN(qnname_sz, -1) => 1]
  elseif conserve_szparity
    return [QN(qnname_szparity, 1, 2) => 1,
            QN(qnname_szparity, 0, 2) => 1]
  end
  return 2
end

state(::SiteType"S=1/2", ::StateName"Up") = 1
state(::SiteType"S=1/2", ::StateName"Dn") = 2

state(st::SiteType"S=1/2", ::StateName"↑") =
  state(st, StateName("Up"))
state(st::SiteType"S=1/2", ::StateName"↓") =
  state(st, StateName("Dn"))

function op!(Op::ITensor,
             ::OpName"Z",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>1] = 1.0
  Op[s'=>2, s=>2] = -1.0
end

function op!(Op::ITensor,
             ::OpName"Sz",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>1] = 0.5
  Op[s'=>2, s=>2] = -0.5
end

op!(Op::ITensor,
    ::OpName"Sᶻ",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("Sz"), t, s)

function op!(Op::ITensor,
             ::OpName"S+",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>2] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁺",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("S+"), t, s)

op!(Op::ITensor,
    ::OpName"Splus",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("S+"), t, s)

function op!(Op::ITensor,
             ::OpName"S-",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>2, s=>1] = 1.0
end

op!(Op::ITensor,
    ::OpName"S⁻",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("S-"), t, s)

op!(Op::ITensor,
    ::OpName"Sminus",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("S-"), t, s)

function op!(Op::ITensor,
             ::OpName"X",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>2] = 1.0
  Op[s'=>2, s=>1] = 1.0
end

function op!(Op::ITensor,
             ::OpName"Sx",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>2] = 0.5
  Op[s'=>2, s=>1] = 0.5
end


op!(Op::ITensor,
    ::OpName"Sˣ",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("Sx"),t, s)

function op!(Op::ITensor,
             ::OpName"iSy",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>2] = +0.5
  Op[s'=>2, s=>1] = -0.5
end

op!(Op::ITensor,
    ::OpName"iSʸ",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("iSy"), t, s)

function op!(Op::ITensor,
             ::OpName"Y",
             ::SiteType"S=1/2",
             s::Index)
  complex!(Op)
  Op[s'=>1, s=>2] = -1.0im
  Op[s'=>2, s=>1] = 1.0im
end

function op!(Op::ITensor,
             ::OpName"Sy",
             ::SiteType"S=1/2",
             s::Index)
  complex!(Op)
  Op[s'=>1, s=>2] = -0.5im
  Op[s'=>2, s=>1] = 0.5im
end

op!(Op::ITensor,
    ::OpName"Sʸ",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("Sy"), t, s)

function op!(Op::ITensor,
             ::OpName"S2",
             ::SiteType"S=1/2",
             s::Index)
  Op[s'=>1, s=>1] = 0.75
  Op[s'=>2, s=>2] = 0.75
end

op!(Op::ITensor,
    ::OpName"S²",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("S2"), t, s)

function op!(Op::ITensor,
             ::OpName"ProjUp",
             ::SiteType"S=1/2",
             s::Index)
  Op[s' => 1, s => 1] = 1
end

op!(Op::ITensor,
    ::OpName"projUp",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("ProjUp"), t, s)

function op!(Op::ITensor,
             ::OpName"ProjDn",
             ::SiteType"S=1/2",
             s::Index)
  Op[s' => 2, s => 2] = 1
end

op!(Op::ITensor,
    ::OpName"projDn",
    t::SiteType"S=1/2",
    s::Index) = op!(Op, OpName("ProjDn"), t, s)

# Support the tag "SpinHalf" as equivalent to "S=1/2"

space(::SiteType"SpinHalf"; kwargs...) =
  space(SiteType("S=1/2"); kwargs...)

state(::SiteType"SpinHalf", n::StateName) =
  state(SiteType("S=1/2"), n)

op!(Op::ITensor, o::OpName, ::SiteType"SpinHalf", s::Index...) =
  op!(Op, o, SiteType("S=1/2"), s...)

# Support the tag "S=½" as equivalent to "S=1/2"

space(::SiteType"S=½"; kwargs...) =
  space(SiteType("S=1/2"); kwargs...)

state(::SiteType"S=½", n::StateName) =
  state(SiteType("S=1/2"), n)

op!(Op::ITensor, o::OpName, ::SiteType"S=½", s::Index...) =
  op!(Op, o, SiteType("S=1/2"), s...)

