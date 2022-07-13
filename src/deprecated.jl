
# global_variables.jl
@deprecate disable_combine_contract!() ITensors.disable_combine_contract()
@deprecate disable_tblis!() ITensors.disable_tblis()
@deprecate disable_warn_order!() ITensors.disable_warn_order()
@deprecate enable_combine_contract!() ITensors.enable_combine_contract()
@deprecate enable_tblis!() ITensors.enable_tblis()
@deprecate reset_warn_order!() ITensors.reset_warn_order()
@deprecate set_warn_order!(N) ITensors.set_warn_order(N)
@deprecate use_combine_contract() ITensors.using_combine_contract()
@deprecate use_debug_checks() ITensors.using_debug_checks()

# index.jl
@deprecate getindex(i::Index, n::Int) (i => n)

# indexset.jl
@deprecate store(is::IndexSet) data(is)
@deprecate firstintersect(is...; kwargs...) getfirst(intersect(is...); kwargs...)
@deprecate firstsetdiff(is...; kwargs...) getfirst(setdiff(is...); kwargs...)

# itensor.jl
@deprecate commonindex(args...; kwargs...) commonind(args...; kwargs...)
@deprecate emptyITensor(::Type{Any}) emptyITensor()
@deprecate findindex(args...; kwargs...) firstind(args...; kwargs...)
@deprecate findinds(args...; kwargs...) inds(args...; kwargs...)
@deprecate linkindex(args...; kwargs...) linkind(args...; kwargs...)
@deprecate matmul(A::ITensor, B::ITensor) product(A, B)
@deprecate replaceindex!(args...; kwargs...) replaceind!(args...; kwargs...)
@deprecate siteindex(args...; kwargs...) siteind(args...; kwargs...)
@deprecate store(A::ITensor) storage(A)
@deprecate setstore!(T::ITensor, st) setstorage!(T, st) false
@deprecate setstore(T::ITensor, st) setstorage(T, st) false
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

# mps/mpo.jl
@deprecate MPO(A::MPS; kwargs...) outer(A', A; kwargs...)

# mps/mps.jl
@deprecate randomMPS(sites::Vector{<:Index}, linkdims::Integer) randomMPS(
  sites; linkdims=linkdims
)
@deprecate randomMPS(ElType::Type, sites::Vector{<:Index}, linkdims::Integer) randomMPS(
  ElType, sites; linkdims=linkdims
)
@deprecate randomMPS(sites::Vector{<:Index}, state, linkdims::Integer) randomMPS(
  sites, state; linkdims=linkdims
)

# physics/autompo.jl
@deprecate toMPO(args...; kwargs...) MPO(args...; kwargs...)

# qn/qn.jl
@deprecate store(qn::QN) data(qn)
