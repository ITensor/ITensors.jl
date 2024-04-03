
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

# physics/autompo.jl
@deprecate toMPO(args...; kwargs...) MPO(args...; kwargs...)

# qn/qn.jl
@deprecate store(qn::QN) data(qn)
