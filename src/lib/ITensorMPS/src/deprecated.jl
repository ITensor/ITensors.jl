# mps/abstractmps.jl
@deprecate orthoCenter(args...; kwargs...) orthocenter(args...; kwargs...)
@deprecate store(m::AbstractMPS) data(m) false
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
@deprecate randomMPO(args...; kwargs...) random_mpo(args...; kwargs...)

# mps/mps.jl
@deprecate randomMPS(args...; kwargs...) random_mps(args...; kwargs...)

# Deprecated syntax for specifying link dimensions.
@deprecate randomMPS(elt::Type{<:Number}, sites::Vector{<:Index}, state, linkdims::Integer) random_mps(
  elt, sites, state; linkdims
)
@deprecate randomMPS(elt::Type{<:Number}, sites::Vector{<:Index}, linkdims::Integer) random_mps(
  elt, sites; linkdims
)
@deprecate randomMPS(sites::Vector{<:Index}, state, linkdims::Integer) random_mps(
  sites, state; linkdims
)
@deprecate randomMPS(sites::Vector{<:Index}, linkdims::Integer) random_mps(sites; linkdims)

# Pass throughs of old name to new name:

unique_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer) = siteinds(uniqueind, A, B, j)

unique_siteinds(A::AbstractMPS, B::AbstractMPS, j::Integer) = siteinds(uniqueinds, A, B, j)

unique_siteinds(A::AbstractMPS, B::AbstractMPS) = siteinds(uniqueind, A, B)

common_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer) = siteinds(commonind, A, B, j)

common_siteinds(A::AbstractMPS, B::AbstractMPS, j::Integer) = siteinds(commoninds, A, B, j)

common_siteinds(A::AbstractMPS, B::AbstractMPS) = siteinds(commoninds, A, B)

firstsiteind(M::AbstractMPS, j::Integer; kwargs...) = siteind(first, M, j; kwargs...)

map_linkinds!(f::Function, M::AbstractMPS) = map!(f, linkinds, M)

map_linkinds(f::Function, M::AbstractMPS) = map(f, linkinds, M)

function map_common_siteinds!(f::Function, M1::AbstractMPS, M2::AbstractMPS)
  return map!(f, siteinds, commoninds, M1, M2)
end

function map_common_siteinds(f::Function, M1::AbstractMPS, M2::AbstractMPS)
  return map(f, siteinds, commoninds, M1, M2)
end

function map_unique_siteinds!(f::Function, M1::AbstractMPS, M2::AbstractMPS)
  return map!(f, siteinds, uniqueinds, M1, M2)
end

function map_unique_siteinds(f::Function, M1::AbstractMPS, M2::AbstractMPS)
  return map(f, siteinds, uniqueinds, M1, M2)
end

for fname in
    (:sim, :prime, :setprime, :noprime, :addtags, :removetags, :replacetags, :settags)
  @eval begin
    function $(Symbol(fname, :_linkinds))(M::AbstractMPS, args...; kwargs...)
      return map(i -> $fname(i, args...; kwargs...), linkinds, M)
    end
    function $(Symbol(fname, :_linkinds!))(M::AbstractMPS, args...; kwargs...)
      return map!(i -> $fname(i, args...; kwargs...), linkinds, M)
    end
    function $(Symbol(fname, :_common_siteinds))(
      M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...
    )
      return map(i -> $fname(i, args...; kwargs...), siteinds, commoninds, M1, M2)
    end
    function $(Symbol(fname, :_common_siteinds!))(
      M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...
    )
      return map!(i -> $fname(i, args...; kwargs...), siteinds, commoninds, M1, M2)
    end
    function $(Symbol(fname, :_unique_siteinds))(
      M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...
    )
      return map(i -> $fname(i, args...; kwargs...), siteinds, uniqueinds, M1, M2)
    end
    function $(Symbol(fname, :_unique_siteinds!))(
      M1::AbstractMPS, M2::AbstractMPS, args...; kwargs...
    )
      return map!(i -> $fname(i, args...; kwargs...), siteinds, uniqueinds, M1, M2)
    end
  end
end
