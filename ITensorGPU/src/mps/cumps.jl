# cu(ψ::Union{MPS,MPO}) = map(cu, ψ)
# cpu(ψ::Union{MPS,MPO}) = map(cpu, ψ)

cuMPS(ψ::MPS) = cu(ψ)
cuMPS(args...; kwargs...) = cu(MPS(args...; kwargs...))
randomCuMPS(args...; kwargs...) = cu(randomMPS(args...; kwargs...))

# For backwards compatibility
productCuMPS(args...; kwargs...) = cuMPS(args...; kwargs...)

cuMPO(M::MPO) = cu(M)
cuMPO(args...; kwargs...) = cu(MPO(args...; kwargs...))
randomCuMPO(args...; kwargs...) = cu(randomMPO(args...; kwargs...))
