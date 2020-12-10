
# indexset.jl
@deprecate store(is::IndexSet) data(is)

# itensor.jl
@deprecate commonindex(args...; kwargs...) commonind(args...; kwargs...)
@deprecate findindex(args...; kwargs...) firstind(args...; kwargs...)
@deprecate findinds(args...; kwargs...) inds(args...; kwargs...)
@deprecate linkindex(args...; kwargs...) linkind(args...; kwargs...)
@deprecate matmul(A::ITensor, B::ITensor) product(A, B)
@deprecate replaceindex!(args...; kwargs...) replaceind!(args...; kwargs...)
@deprecate siteindex(args...; kwargs...) siteind(args...; kwargs...)
@deprecate uniqueindex(args...; kwargs...) uniqueind(args...; kwargs...)


# mps/abstractmps.jl
@deprecate orthoCenter(args...; kwargs...) orthocenter(args...; kwargs...)
@deprecate store(m::AbstractMPS) data(m)
@deprecate replacesites!(args...; kwargs...) ITensors.replace_siteinds!(args...; kwargs...)
@deprecate applyMPO(args...; kwargs...) contract(args...; kwargs...)
@deprecate applympo(args...; kwargs...) contract(args...; kwargs...)
@deprecate errorMPOprod(args...; kwargs...) error_contract(args...; kwargs...)
@deprecate error_mpoprod(args...; kwargs...) error_contract(args...; kwargs...)
@deprecate error_mul(args...; kwargs...) error_contract(args...; kwargs...)
@deprecate multMPO(args...; kwargs...) contract(args...; kwargs...)
@deprecate sum(A::AbstractMPS, B::AbstractMPS; kwargs...) add(A, B; kwargs...)
@deprecate multmpo(args...; kwargs...) contract(args...; kwargs...)
@deprecate set_leftlim!(args...; kwargs...) ITensors.setleftlim!(args...; kwargs...)
@deprecate set_rightlim!(args...; kwargs...) ITensors.setrightlim!(args...; kwargs...)
@deprecate tensors(args...; kwargs...) ITensors.data(args...; kwargs...)
@deprecate primelinks!(args...; kwargs...) ITensors.prime_linkinds!(args...; kwargs...)
@deprecate simlinks!(args...; kwargs...) ITensors.sim_linkinds!(args...; kwargs...)
@deprecate mul(A::AbstractMPS, B::AbstractMPS; kwargs...) contract(A, B; kwargs...)

# physics/autompo.jl
@deprecate toMPO(args...; kwargs...) MPO(args...; kwargs...)

# qn/qn.jl
@deprecate store(qn::QN) data(qn)

