
"""
    space(::SiteType"S=1/2";
          conserve_qns = false,
          conserve_sz = conserve_qns,
          conserve_szparity = false,
          qnname_sz = "Sz",
          qnname_szparity = "SzParity")

Create the Hilbert space for a site of type "S=1/2".

Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(
  ::SiteType"S=1/2";
  conserve_qns=false,
  conserve_sz=conserve_qns,
  conserve_szparity=false,
  qnname_sz="Sz",
  qnname_szparity="SzParity",
)
  if conserve_sz && conserve_szparity
    return [
      QN((qnname_sz, +1), (qnname_szparity, 1, 2)) => 1,
      QN((qnname_sz, -1), (qnname_szparity, 0, 2)) => 1,
    ]
  elseif conserve_sz
    return [QN(qnname_sz, +1) => 1, QN(qnname_sz, -1) => 1]
  elseif conserve_szparity
    return [QN(qnname_szparity, 1, 2) => 1, QN(qnname_szparity, 0, 2) => 1]
  end
  return 2
end

ITensors.val(::ValName"Up", ::SiteType"S=1/2") = 1
ITensors.val(::ValName"Dn", ::SiteType"S=1/2") = 2

ITensors.val(::ValName"↑", st::SiteType"S=1/2") = 1
ITensors.val(::ValName"↓", st::SiteType"S=1/2") = 2

ITensors.val(::ValName"Z+", ::SiteType"S=1/2") = 1
ITensors.val(::ValName"Z-", ::SiteType"S=1/2") = 2

ITensors.state(::StateName"Up", ::SiteType"S=1/2") = [1.0, 0.0]
ITensors.state(::StateName"Dn", ::SiteType"S=1/2") = [0.0, 1.0]

ITensors.state(::StateName"↑", st::SiteType"S=1/2") = [1.0, 0.0]
ITensors.state(::StateName"↓", st::SiteType"S=1/2") = [0.0, 1.0]

ITensors.state(::StateName"Z+", st::SiteType"S=1/2") = [1.0, 0.0]
ITensors.state(::StateName"Z-", st::SiteType"S=1/2") = [0.0, 1.0]

ITensors.state(::StateName"X+", st::SiteType"S=1/2") = [1 / sqrt(2), 1 / sqrt(2)]
ITensors.state(::StateName"X-", st::SiteType"S=1/2") = [1 / sqrt(2), -1 / sqrt(2)]

ITensors.state(::StateName"Y+", st::SiteType"S=1/2") = [1 / sqrt(2), im / sqrt(2)]
ITensors.state(::StateName"Y-", st::SiteType"S=1/2") = [1 / sqrt(2), -im / sqrt(2)]

ITensors.op(::OpName"Z", ::SiteType"S=1/2") = [
  1 0
  0 -1
]

ITensors.op(::OpName"Sz", ::SiteType"S=1/2") = [
  0.5 0.0
  0.0 -0.5
]

ITensors.op(::OpName"Sᶻ", t::SiteType"S=1/2") = op(OpName("Sz"), t)

ITensors.op(::OpName"S+", ::SiteType"S=1/2") = [
  0 1
  0 0
]

ITensors.op(::OpName"S⁺", t::SiteType"S=1/2") = op(OpName("S+"), t)

ITensors.op(::OpName"Splus", t::SiteType"S=1/2") = op(OpName("S+"), t)

ITensors.op(::OpName"S-", ::SiteType"S=1/2") = [
  0 0
  1 0
]

ITensors.op(::OpName"S⁻", t::SiteType"S=1/2") = op(OpName("S-"), t)

ITensors.op(::OpName"Sminus", t::SiteType"S=1/2") = op(OpName("S-"), t)

ITensors.op(::OpName"X", ::SiteType"S=1/2") = [
  0 1
  1 0
]

ITensors.op(::OpName"Sx", ::SiteType"S=1/2") = [
  0.0 0.5
  0.5 0.0
]

ITensors.op(::OpName"Sˣ", t::SiteType"S=1/2") = op(OpName("Sx"), t)

ITensors.op(::OpName"iY", ::SiteType"S=1/2") = [
  0 1
  -1 0
]

ITensors.op(::OpName"iSy", ::SiteType"S=1/2") = [
  0.0 0.5
  -0.5 0.0
]

ITensors.op(::OpName"iSʸ", t::SiteType"S=1/2") = op(OpName("iSy"), t)

ITensors.op(::OpName"Y", ::SiteType"S=1/2") = [
  0.0 -1.0im
  1.0im 0.0
]

ITensors.op(::OpName"Sy", ::SiteType"S=1/2") = [
  0.0 -0.5im
  0.5im 0.0
]

ITensors.op(::OpName"Sʸ", t::SiteType"S=1/2") = op(OpName("Sy"), t)

ITensors.op(::OpName"S2", ::SiteType"S=1/2") = [
  0.75 0.0
  0.0 0.75
]

ITensors.op(::OpName"S²", t::SiteType"S=1/2") = op(OpName("S2"), t)

ITensors.op(::OpName"ProjUp", ::SiteType"S=1/2") = [
  1 0
  0 0
]

ITensors.op(::OpName"projUp", t::SiteType"S=1/2") = op(OpName("ProjUp"), t)

ITensors.op(::OpName"ProjDn", ::SiteType"S=1/2") = [
  0 0
  0 1
]

ITensors.op(::OpName"projDn", t::SiteType"S=1/2") = op(OpName("ProjDn"), t)

# Support the tag "SpinHalf" as equivalent to "S=1/2"

space(::SiteType"SpinHalf"; kwargs...) = space(SiteType("S=1/2"); kwargs...)

val(name::ValName, ::SiteType"SpinHalf") = val(name, SiteType("S=1/2"))

ITensors.state(name::StateName, ::SiteType"SpinHalf") = state(name, SiteType("S=1/2"))

function ITensors.op(o::OpName, ::SiteType"SpinHalf"; kwargs...)
  return op(o, SiteType("S=1/2"); kwargs...)
end

# Support the tag "S=½" as equivalent to "S=1/2"

space(::SiteType"S=½"; kwargs...) = space(SiteType("S=1/2"); kwargs...)

val(name::ValName, ::SiteType"S=½") = val(name, SiteType("S=1/2"))

ITensors.state(name::StateName, ::SiteType"S=½") = state(name, SiteType("S=1/2"))

ITensors.op(o::OpName, ::SiteType"S=½"; kwargs...) = op(o, SiteType("S=1/2"); kwargs...)
