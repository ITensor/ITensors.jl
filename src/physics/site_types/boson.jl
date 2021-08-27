
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

ITensors.state(sn::StateName, st::SiteType"Boson", s::Index) = state(sn, alias(st), s)

ITensors.op(on::OpName, st::SiteType"Boson", s::Index) = op(on, alias(st), s)
