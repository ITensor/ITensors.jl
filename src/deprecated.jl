# global_variables.jl
@deprecate disable_tblis!() ITensors.disable_tblis()
@deprecate disable_warn_order!() ITensors.disable_warn_order()
@deprecate enable_tblis!() ITensors.enable_tblis()
@deprecate reset_warn_order!() ITensors.reset_warn_order()
@deprecate set_warn_order!(N) ITensors.set_warn_order(N)
@deprecate use_debug_checks() ITensors.using_debug_checks()

# index.jl
@deprecate getindex(i::Index, n::Int) (i => n)

# indexset.jl
@deprecate store(is::IndexSet) data(is)
@deprecate firstintersect(is...; kwargs...) getfirst(intersect(is...); kwargs...)
@deprecate firstsetdiff(is...; kwargs...) getfirst(setdiff(is...); kwargs...)

# itensor.jl
@deprecate commonindex(args...; kwargs...) commonind(args...; kwargs...)
@deprecate diagITensor(args...; kwargs...) diag_itensor(args...; kwargs...)
@deprecate emptyITensor(::Type{Any}) emptyITensor()
@deprecate findindex(args...; kwargs...) firstind(args...; kwargs...)
@deprecate findinds(args...; kwargs...) inds(args...; kwargs...)
@deprecate linkindex(args...; kwargs...) linkind(args...; kwargs...)
@deprecate matmul(A::ITensor, B::ITensor) product(A, B)
@deprecate randomITensor(args...; kwargs...) random_itensor(args...; kwargs...)
@deprecate replaceindex!(args...; kwargs...) replaceind!(args...; kwargs...)
@deprecate siteindex(args...; kwargs...) siteind(args...; kwargs...)
@deprecate store(A::ITensor) storage(A)
@deprecate setstore!(T::ITensor, st) setstorage!(T, st) false
@deprecate setstore(T::ITensor, st) setstorage(T, st) false
@deprecate uniqueindex(args...; kwargs...) uniqueind(args...; kwargs...)

# qn/qn.jl
@deprecate store(qn::QN) data(qn)
