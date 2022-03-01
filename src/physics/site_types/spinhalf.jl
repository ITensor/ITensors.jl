
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

# Use Qubit  definition of any operator/state 
# called using S=1/2 SiteType
function ITensors.val(vn::ValName, ::SiteType"S=1/2"; kwargs...)
  return val(vn, SiteType("Qubit"); kwargs...)
end

function ITensors.state(sn::StateName, ::SiteType"S=1/2"; kwargs...)
  return state(sn, SiteType("Qubit"); kwargs...)
end

ITensors.op(o::OpName, ::SiteType"S=1/2"; kwargs...) = op(o, SiteType("Qubit"); kwargs...)

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
