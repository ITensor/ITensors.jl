# TODO: Deprecate.
const AutoMPO = OpSum

"""
    add!(opsum::OpSum,
         op1::String, i1::Int)

    add!(opsum::OpSum,
         coef::Number,
         op1::String, i1::Int)

    add!(opsum::OpSum,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    add!(opsum::OpSum,
         coef::Number,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    +(opsum:OpSum, term::Tuple)

Add a single- or multi-site operator
term to the OpSum `opsum`. Each operator
is specified by a name (String) and a
site number (Int). The second version
accepts a real or complex coefficient.

The `+` operator version of this function
accepts a tuple with entries either
(String,Int,String,Int,...) or
(Number,String,Int,String,Int,...)
where these tuple values are the same
as valid inputs to the `add!` function.
For inputting a very large number of
terms (tuples) to an OpSum, consider
using the broadcasted operator `.+=`
which avoids reallocating the OpSum
after each addition.

# Examples
```julia
opsum = OpSum()

add!(opsum,"Sz",2,"Sz",3)

opsum += ("Sz",3,"Sz",4)

opsum += (0.5,"S+",4,"S-",5)

opsum .+= (0.5,"S+",5,"S-",6)
```
"""
function add!(os::OpSum, o::Scaled{C,Prod{Op}}) where {C}
  push!(terms(os), o)
  return os
end
add!(os::OpSum, o::Op) = add!(os, Prod{Op}() * o)
add!(os::OpSum, o::Scaled{C,Op}) where {C} = add!(os, Prod{Op}() * o)
add!(os::OpSum, o::Prod{Op}) = add!(os, one(Float64) * o)
add!(os::OpSum, o::Tuple) = add!(os, Ops.op_term(o))
add!(os::OpSum, a1::String, args...) = add!(os, (a1, args...))
add!(os::OpSum, a1::Number, args...) = add!(os, (a1, args...))
subtract!(os::OpSum, o::Tuple) = add!(os, -Ops.op_term(o))

function isfermionic(t::Vector{Op}, sites)
  p = +1
  for op in t
    if has_fermion_string(name(op), sites[site(op)])
      p *= -1
    end
  end
  return (p == -1)
end

#
# Abuse broadcasting syntax for in-place addition:
#
# os .+= ("Sz",1)
# os .-= ("Sz",1)
#
# TODO: Deprecate this syntax?
#

struct OpSumStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:OpSum}) = OpSumStyle()

struct OpSumAddTermStyle <: Broadcast.BroadcastStyle end

Base.broadcastable(os::OpSum) = os

Base.BroadcastStyle(::OpSumStyle, ::Broadcast.Style{Tuple}) = OpSumAddTermStyle()

Broadcast.instantiate(bc::Broadcast.Broadcasted{OpSumAddTermStyle}) = bc

function Base.copyto!(os, bc::Broadcast.Broadcasted{OpSumAddTermStyle,<:Any,typeof(+)})
  add!(os, bc.args[2])
  return os
end

function Base.copyto!(os, bc::Broadcast.Broadcasted{OpSumAddTermStyle,<:Any,typeof(-)})
  subtract!(os, bc.args[2])
  return os
end

# XXX: Create a new function name for this.
isempty(op_qn::Pair{Vector{Op},QN}) = isempty(op_qn.first)

# the key type is Prod{Op} for the dense case
# and is Pair{Prod{Op},QN} for the QN conserving case
function posInLink!(linkmap::Dict{K,Int}, k::K)::Int where {K}
  isempty(k) && return -1
  pos = get(linkmap, k, -1)
  if pos == -1
    pos = length(linkmap) + 1
    linkmap[k] = pos
  end
  return pos
end

# TODO: Define as `C`. Rename `coefficient_type`.
function determineValType(terms::Vector{Scaled{C,Prod{Op}}}) where {C}
  for t in terms
    (!isreal(coefficient(t))) && return ComplexF64
  end
  return Float64
end

function computeSiteProd(sites, ops::Prod{Op})::ITensor
  i = only(site(ops[1]))
  T = op(sites[i], which_op(ops[1]); params(ops[1])...)
  for j in 2:length(ops)
    (only(site(ops[j])) != i) && error("Mismatch of site number in computeSiteProd")
    opj = op(sites[i], which_op(ops[j]); params(ops[j])...)
    T = product(T, opj)
  end
  return T
end

function remove_dups!(v::Vector{T}) where {T}
  N = length(v)
  (N == 0) && return nothing
  sort!(v)
  n = 1
  u = 2
  while u <= N
    while u < N && v[u] == v[n]
      u += 1
    end
    if v[u] != v[n]
      v[n + 1] = v[u]
      n += 1
    end
    u += 1
  end
  resize!(v, n)
  return nothing
end #remove_dups!

function sorteachterm(os::OpSum, sites)
  os = copy(os)
  isless_site(o1::Op, o2::Op) = site(o1) < site(o2)
  N = length(sites)
  for n in eachindex(os)
    t = os[n]
    Nt = length(t)

    if maximum(ITensors.sites(t)) > length(sites)
      error(
        "The OpSum contains a term $t that extends beyond the number of sites $(length(sites)).",
      )
    end

    prevsite = N + 1 #keep track of whether we are switching
    #to a new site to make sure F string
    #is only placed at most once for each site

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    perm = Vector{Int}(undef, Nt)
    sortperm!(perm, terms(t); alg=InsertionSort, lt=isless_site)

    t = coefficient(t) * Prod(terms(t)[perm])

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    parity = +1
    for n in Nt:-1:1
      currsite = site(t[n])
      fermionic = has_fermion_string(which_op(t[n]), sites[only(site(t[n]))])
      if !using_auto_fermion() && (parity == -1) && (currsite < prevsite)
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        terms(t)[n] = Op("$(which_op(t[n])) * F", only(site(t[n])))
      end
      prevsite = currsite

      if fermionic
        parity = -parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoMPO")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)
    # and account for anti-commuting, fermionic operators
    # during above sort; put resulting sign into coef
    t *= parity_sign(perm)
    terms(os)[n] = t
  end
  return os
end

function sortmergeterms(os::OpSum{C}) where {C}
  os_sorted_terms = sort(terms(os))
  os = Sum(os_sorted_terms)
  # Merge (add) terms with same operators
  merge_os_data = Scaled{C,Prod{Op}}[]
  last_term = copy(os[1])
  last_term_coef = coefficient(last_term)
  for n in 2:length(os)
    if argument(os[n]) == argument(last_term)
      last_term_coef += coefficient(os[n])
      last_term = last_term_coef * argument(last_term)
    else
      push!(merge_os_data, last_term)
      last_term = os[n]
      last_term_coef = coefficient(last_term)
    end
  end
  push!(merge_os_data, last_term)
  os = Sum(merge_os_data)
  return os
end

"""
    MPO(os::OpSum, sites::Vector{<:Index}; splitblocks=true, kwargs...)
    MPO(eltype::Type{<:Number}, os::OpSum, sites::Vector{<:Index}; splitblocks=true, kwargs...)

Convert an OpSum object `os` to an
MPO, with indices given by `sites`. The
resulting MPO will have the indices
`sites[1], sites[1]', sites[2], sites[2]'`
etc. The conversion is done by an algorithm
that compresses the MPO resulting from adding
the OpSum terms together, often achieving
the minimum possible bond dimension.

Optionally specify the desired element type
of the output MPO by passing the type
as the first argument.

The keyword argument `splitblocks` controls
the sparsity of the resulting MPO.
With the default `splitblocks=true`, the link
indices of the MPO are split into blocks of
dimension 1, potentially making the MPO more sparse.

With the `splitblocks=false`, the blocks
of the link dimensions are packed as much as
possible according to common quantum numbers,
making larger blocks. Before ITensors 0.3.19,
this was the default output, but we have found
that in general MPOs output with `splitblocks=true`
lead to better performance in algorithms like
DMRG.

# Examples

```julia
os = OpSum()
os += "Sz",1,"Sz",2
os += "Sz",2,"Sz",3
os += "Sz",3,"Sz",4

sites = siteinds("S=1/2",4)
H = MPO(os,sites)
H = MPO(Float32,os,sites)
H = MPO(os,sites; splitblocks=false)
```
"""
function MPO(os::OpSum, sites::Vector{<:Index}; splitblocks=true, kwargs...)::MPO
  length(terms(os)) == 0 && error("OpSum has no terms")

  os = deepcopy(os)
  os = sorteachterm(os, sites)
  os = sortmergeterms(os)

  if hasqns(sites[1])
    return qn_svdMPO(os, sites; kwargs...)
  end
  M = svdMPO(os, sites; kwargs...)
  if splitblocks
    M = ITensors.splitblocks(linkinds, M)
  end
  return M
end

function MPO(eltype::Type{<:Number}, os::OpSum, sites::Vector{<:Index}; kwargs...)
  return NDTensors.convert_scalartype(eltype, MPO(os, sites; kwargs...))
end

# Conversion from other formats
function MPO(o::Op, s::Vector{<:Index}; kwargs...)
  return MPO(OpSum{Float64}() + o, s; kwargs...)
end

function MPO(o::Scaled{C,Op}, s::Vector{<:Index}; kwargs...) where {C}
  return MPO(OpSum{C}() + o, s; kwargs...)
end

function MPO(o::Sum{Op}, s::Vector{<:Index}; kwargs...)
  return MPO(OpSum{Float64}() + o, s; kwargs...)
end

function MPO(o::Prod{Op}, s::Vector{<:Index}; kwargs...)
  return MPO(OpSum{Float64}() + o, s; kwargs...)
end

function MPO(o::Scaled{C,Prod{Op}}, s::Vector{<:Index}; kwargs...) where {C}
  return MPO(OpSum{C}() + o, s; kwargs...)
end

function MPO(o::Sum{Scaled{C,Op}}, s::Vector{<:Index}; kwargs...) where {C}
  return MPO(OpSum{C}() + o, s; kwargs...)
end
