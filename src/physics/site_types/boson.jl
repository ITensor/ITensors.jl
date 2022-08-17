
alias(::SiteType"Boson") = SiteType"Qudit"()

"""
    space(::SiteType"Boson";
          dim = 2,
          conserve_qns = false,
          conserve_number = false,
          qnname_number = "Number")

Create the Hilbert space for a site of type "Boson".

Optionally specify the conserved symmetries and their quantum number labels.
"""
ITensors.space(st::SiteType"Boson"; kwargs...) = space(alias(st); kwargs...)

ITensors.val(vn::ValName, st::SiteType"Boson") = val(vn, alias(st))

function ITensors.state(sn::StateName, st::SiteType"Boson", s::Index; kwargs...)
  return state(sn, alias(st), s; kwargs...)
end

function ITensors.op(on::OpName, st::SiteType"Boson", ds::Int...; kwargs...)
  return op(on, alias(st), ds...; kwargs...)
end

function ITensors.op(
  on::OpName, st::SiteType"Boson", s1::Index, s_tail::Index...; kwargs...
)
  rs = reverse((s1, s_tail...))
  ds = dim.(rs)
  opmat = op(on, st, ds...; kwargs...)
  return itensor(opmat, prime.(rs)..., dag.(rs)...)
end
