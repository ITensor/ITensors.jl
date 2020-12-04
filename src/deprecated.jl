
# itensor.jl
@deprecate addblock! insertblock!

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

