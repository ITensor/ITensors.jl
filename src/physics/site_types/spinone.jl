
"""
    space(::SiteType"S=1";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          qnname_sz = "Sz")

Create the Hilbert space for a site of type "S=1".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
  ::SiteType"S=1"; conserve_qns=false, conserve_sz=conserve_qns, qnname_sz="Sz"
)
  if conserve_sz
    return [QN(qnname_sz, +2) => 1, QN(qnname_sz, 0) => 1, QN(qnname_sz, -2) => 1]
  end
  return 3
end

ITensors.val(::ValName"Up", ::SiteType"S=1") = 1
ITensors.val(::ValName"Z0", ::SiteType"S=1") = 2
ITensors.val(::ValName"Dn", ::SiteType"S=1") = 3

ITensors.val(::ValName"↑", st::SiteType"S=1") = 1
ITensors.val(::ValName"0", st::SiteType"S=1") = 2
ITensors.val(::ValName"↓", st::SiteType"S=1") = 3

ITensors.val(::ValName"Z+", ::SiteType"S=1") = 1
# -- Z0 is already defined above --
ITensors.val(::ValName"Z-", ::SiteType"S=1") = 3

ITensors.state(::StateName"Up", ::SiteType"S=1") = [1.0, 0.0, 0.0]
ITensors.state(::StateName"Z0", ::SiteType"S=1") = [0.0, 1.0, 0.0]
ITensors.state(::StateName"Dn", ::SiteType"S=1") = [0.0, 0.0, 1.0]

ITensors.state(::StateName"↑", st::SiteType"S=1") = [1.0, 0.0, 0.0]
ITensors.state(::StateName"0", st::SiteType"S=1") = [0.0, 1.0, 0.0]
ITensors.state(::StateName"↓", st::SiteType"S=1") = [0.0, 0.0, 1.0]

ITensors.state(::StateName"Z+", st::SiteType"S=1") = [1.0, 0.0, 0.0]
# -- Z0 is already defined above --
ITensors.state(::StateName"Z-", st::SiteType"S=1") = [0.0, 0.0, 1.0]

ITensors.state(::StateName"X+", ::SiteType"S=1") = [1 / 2, 1 / sqrt(2), 1 / 2]
ITensors.state(::StateName"X0", ::SiteType"S=1") = [-1 / sqrt(2), 0, 1 / sqrt(2)]
ITensors.state(::StateName"X-", ::SiteType"S=1") = [1 / 2, -1 / sqrt(2), 1 / 2]

ITensors.state(::StateName"Y+", ::SiteType"S=1") = [-1 / 2, -im / sqrt(2), 1 / 2]
ITensors.state(::StateName"Y0", ::SiteType"S=1") = [1 / sqrt(2), 0, 1 / sqrt(2)]
ITensors.state(::StateName"Y-", ::SiteType"S=1") = [-1 / 2, im / sqrt(2), 1 / 2]

function ITensors.op!(Op::ITensor, ::OpName"Sz", ::SiteType"S=1", s::Index)
  Op[s' => 1, s => 1] = +1.0
  return Op[s' => 3, s => 3] = -1.0
end

function ITensors.op!(Op::ITensor, ::OpName"Sᶻ", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("Sz"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"S+", ::SiteType"S=1", s::Index)
  Op[s' => 2, s => 3] = sqrt(2)
  return Op[s' => 1, s => 2] = sqrt(2)
end

function ITensors.op!(Op::ITensor, ::OpName"S⁺", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S+"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Splus", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S+"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sp", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S+"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"S-", ::SiteType"S=1", s::Index)
  Op[s' => 3, s => 2] = sqrt(2)
  return Op[s' => 2, s => 1] = sqrt(2)
end

function ITensors.op!(Op::ITensor, ::OpName"S⁻", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S-"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sminus", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S-"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sm", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("S-"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sx", ::SiteType"S=1", s::Index)
  Op[s' => 2, s => 1] = 1 / sqrt(2)
  Op[s' => 1, s => 2] = 1 / sqrt(2)
  Op[s' => 3, s => 2] = 1 / sqrt(2)
  return Op[s' => 2, s => 3] = 1 / sqrt(2)
end

function ITensors.op!(Op::ITensor, ::OpName"Sˣ", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("Sx"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"iSy", ::SiteType"S=1", s::Index)
  Op[s' => 2, s => 1] = -1 / sqrt(2)
  Op[s' => 1, s => 2] = +1 / sqrt(2)
  Op[s' => 3, s => 2] = -1 / sqrt(2)
  return Op[s' => 2, s => 3] = +1 / sqrt(2)
end

function ITensors.op!(Op::ITensor, ::OpName"iSʸ", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("iSy"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sy", ::SiteType"S=1", s::Index)
  complex!(Op)
  Op[s' => 2, s => 1] = +1im / sqrt(2)
  Op[s' => 1, s => 2] = -1im / sqrt(2)
  Op[s' => 3, s => 2] = +1im / sqrt(2)
  return Op[s' => 2, s => 3] = -1im / sqrt(2)
end

function ITensors.op!(Op::ITensor, ::OpName"Sʸ", t::SiteType"S=1", s::Index)
  return op!(Op, OpName("Sy"), t, s)
end

function ITensors.op!(Op::ITensor, ::OpName"Sz2", ::SiteType"S=1", s::Index)
  Op[s' => 1, s => 1] = +1.0
  return Op[s' => 3, s => 3] = +1.0
end

function ITensors.op!(Op::ITensor, ::OpName"Sx2", ::SiteType"S=1", s::Index)
  Op[s' => 1, s => 1] = 0.5
  Op[s' => 3, s => 1] = 0.5
  Op[s' => 2, s => 2] = 1.0
  Op[s' => 1, s => 3] = 0.5
  return Op[s' => 3, s => 3] = 0.5
end

function ITensors.op!(Op::ITensor, ::OpName"Sy2", ::SiteType"S=1", s::Index)
  Op[s' => 1, s => 1] = +0.5
  Op[s' => 3, s => 1] = -0.5
  Op[s' => 2, s => 2] = +1.0
  Op[s' => 1, s => 3] = -0.5
  return Op[s' => 3, s => 3] = +0.5
end

ITensors.space(::SiteType"SpinOne"; kwargs...) = space(SiteType("S=1"); kwargs...)

ITensors.state(name::StateName, ::SiteType"SpinOne") = state(name, SiteType("S=1"))
ITensors.val(name::ValName, ::SiteType"SpinOne") = val(name, SiteType("S=1"))

function ITensors.op!(Op::ITensor, o::OpName, ::SiteType"SpinOne", s::Index)
  return op!(Op, o, SiteType("S=1"), s)
end
