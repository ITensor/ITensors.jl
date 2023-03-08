
@eval struct SiteType{T}
  (f::Type{<:SiteType})() = $(Expr(:new, :f))
end

# Note that the complicated definition of
# SiteType above is a workaround for performance
# issues when creating parameterized types
# in Julia 1.4 and 1.5-beta. Ideally we
# can just use the following in the future:
# struct SiteType{T}
# end

"""
SiteType is a parameterized type which allows
making Index tags into Julia types. Use cases
include overloading functions such as `op`,
`siteinds`, and `state` which generate custom
operators, Index arrays, and IndexVals associated
with Index objects having a certain tag.

To make a SiteType type, you can use the string
macro notation: `SiteType"MyTag"`

To make a SiteType value or object, you can use
the notation: `SiteType("MyTag")`

There are currently a few built-in site types
recognized by `ITensors.jl`. The system is easily extensible
by users. To add new operators to an existing site type,
or to create new site types, you can follow the instructions
[here](https://itensor.github.io/ITensors.jl/stable/examples/Physics.html).

The current built-in site types are:

- `SiteType"S=1/2"` (or `SiteType"S=½"`)
- `SiteType"S=1"`
- `SiteType"Qubit"`
- `SiteType"Qudit"`
- `SiteType"Boson"`
- `SiteType"Fermion"`
- `SiteType"tJ"`
- `SiteType"Electron"`

# Examples

Tags on indices get turned into SiteTypes internally, and then
we search for overloads of functions like `op` and `siteind`.
For example:

```julia
julia> s = siteind("S=1/2")
(dim=2|id=862|"S=1/2,Site")

julia> @show op("Sz", s);
op(s, "Sz") = ITensor ord=2
Dim 1: (dim=2|id=862|"S=1/2,Site")'
Dim 2: (dim=2|id=862|"S=1/2,Site")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.5   0.0
 0.0  -0.5

julia> @show op("Sx", s);
op(s, "Sx") = ITensor ord=2
Dim 1: (dim=2|id=862|"S=1/2,Site")'
Dim 2: (dim=2|id=862|"S=1/2,Site")
NDTensors.Dense{Float64,Array{Float64,1}}
 2×2
 0.0  0.5
 0.5  0.0

julia> @show op("Sy", s);
op(s, "Sy") = ITensor ord=2
Dim 1: (dim=2|id=862|"S=1/2,Site")'
Dim 2: (dim=2|id=862|"S=1/2,Site")
NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}}
 2×2
 0.0 + 0.0im  -0.0 - 0.5im
 0.0 + 0.5im   0.0 + 0.0im

julia> s = siteind("Electron")
(dim=4|id=734|"Electron,Site")

julia> @show op("Nup", s);
op(s, "Nup") = ITensor ord=2
Dim 1: (dim=4|id=734|"Electron,Site")'
Dim 2: (dim=4|id=734|"Electron,Site")
NDTensors.Dense{Float64,Array{Float64,1}}
 4×4
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0
```

Many operators are available, for example:

- `SiteType"S=1/2"`: `"Sz"`, `"Sx"`, `"Sy"`, `"S+"`, `"S-"`, ...
- `SiteType"Electron"`: `"Nup"`, `"Ndn"`, `"Nupdn"`, `"Ntot"`, `"Cup"`,
   `"Cdagup"`, `"Cdn"`, `"Cdagdn"`, `"Sz"`, `"Sx"`, `"Sy"`, `"S+"`, `"S-"`, ...
- ...

You can view the source code for the internal SiteType definitions
and operators that are defined [here](https://github.com/ITensor/ITensors.jl/tree/main/src/physics/site_types).
"""
SiteType(s::AbstractString) = SiteType{Tag(s)}()

SiteType(t::Integer) = SiteType{Tag(t)}()
SiteType(t::Tag) = SiteType{t}()

tag(::SiteType{T}) where {T} = T

macro SiteType_str(s)
  return SiteType{Tag(s)}
end

# Keep TagType defined for backwards
# compatibility; will be deprecated later
const TagType = SiteType
macro TagType_str(s)
  return TagType{Tag(s)}
end

#---------------------------------------
#
# op system
#
#---------------------------------------

@eval struct OpName{Name}
  (f::Type{<:OpName})() = $(Expr(:new, :f))
end

# Note that the complicated definition of
# OpName above is a workaround for performance
# issues when creating parameterized types
# in Julia 1.4 and 1.5-beta. Ideally we
# can just use the following in the future:
# struct OpName{Name}
# end

"""
OpName is a parameterized type which allows
making strings into Julia types for the purpose
of representing operator names.
The main use of OpName is overloading the
`ITensors.op!` method which generates operators
for indices with certain tags such as "S=1/2".

To make a OpName type, you can use the string
macro notation: `OpName"MyTag"`.

To make an OpName value or object, you can use
the notation: `OpName("myop")`
"""
OpName(s::AbstractString) = OpName{Symbol(s)}()
OpName(s::Symbol) = OpName{s}()
name(::OpName{N}) where {N} = N

macro OpName_str(s)
  return OpName{Symbol(s)}
end

# Default implementations of op and op!
op(::OpName; kwargs...) = nothing
op(::OpName, ::SiteType; kwargs...) = nothing
op(::OpName, ::SiteType, ::Index...; kwargs...) = nothing
function op(
  ::OpName, ::SiteType, ::SiteType, sitetypes_inds::Union{SiteType,Index}...; kwargs...
)
  return nothing
end
op!(::ITensor, ::OpName, ::SiteType, ::Index...; kwargs...) = nothing
function op!(
  ::ITensor,
  ::OpName,
  ::SiteType,
  ::SiteType,
  sitetypes_inds::Union{SiteType,Index}...;
  kwargs...,
)
  return nothing
end

# Deprecated version, for backwards compatibility
op(::SiteType, ::Index, ::AbstractString; kwargs...) = nothing

function _sitetypes(ts::TagSet)
  Ntags = length(ts)
  return SiteType[SiteType(data(ts)[n]) for n in 1:Ntags]
end

_sitetypes(i::Index) = _sitetypes(tags(i))

"""
    op(opname::String, s::Index; kwargs...)

Return an ITensor corresponding to the operator
named `opname` for the Index `s`. The operator
is constructed by calling an overload of either
the `op` or `op!` methods which take a `SiteType`
argument that corresponds to one of the tags of
the Index `s` and an `OpName"opname"` argument
that corresponds to the input operator name.

Operator names can be combined using the `"*"`
symbol, for example `"S+*S-"` or `"Sz*Sz*Sz"`.
The result is an ITensor made by forming each operator
then contracting them together in a way corresponding
to the usual operator product or matrix multiplication.

The `op` system is used by the OpSum
system to convert operator names into ITensors,
and can be used directly such as for applying
operators to MPS.

# Example

```julia
s = Index(2, "Site,S=1/2")
Sz = op("Sz", s)
```

To see all of the operator names defined for the site types included with
ITensor, please view the [source code](https://github.com/ITensor/ITensors.jl/tree/main/src/physics/site_types)
for each site type. Note that some site types such as "S=1/2" and "Qubit"
are aliases for each other and share operator definitions.
"""
function op(name::AbstractString, s::Index...; adjoint::Bool=false, kwargs...)
  name = strip(name)
  # TODO: filter out only commons tags
  # if there are multiple indices
  commontags_s = commontags(s...)

  # first we handle the + and - algebra, which requires a space between ops to avoid clashing
  name_split = nothing
  @ignore_derivatives name_split = String.(split(name, " "))
  oplocs = findall(x -> x ∈ ("+", "-"), name_split)

  if !isempty(oplocs)
    @ignore_derivatives !isempty(kwargs) &&
      error("Lazy algebra on parametric gates not allowed")

    # the string representation of algebra ops: ex ["+", "-", "+"]
    labels = name_split[oplocs]
    # assign coefficients to each term: ex [+1, -1, +1]
    coeffs = [1, [(-1)^Int(label == "-") for label in labels]...]

    # grad the name of each operator block separated by an algebra op, and do so by
    # making sure blank spaces between opnames are kept when building the new block.
    start, opnames = 0, String[]
    for oploc in oplocs
      finish = oploc
      opnames = vcat(
        opnames, [prod([name_split[k] * " " for k in (start + 1):(finish - 1)])]
      )
      start = oploc
    end
    opnames = vcat(
      opnames, [prod([name_split[k] * " " for k in (start + 1):length(name_split)])]
    )

    # build the vector of blocks and sum
    op_list = [
      coeff * (op(opname, s...; kwargs...)) for (coeff, opname) in zip(coeffs, opnames)
    ]
    return sum(op_list)
  end

  # the the multiplication come after
  oploc = findfirst("*", name)
  if !isnothing(oploc)
    op1, op2 = nothing, nothing
    @ignore_derivatives begin
      op1 = name[1:prevind(name, oploc.start)]
      op2 = name[nextind(name, oploc.start):end]
      if !(op1[end] == ' ' && op2[1] == ' ')
        @warn "($op1*$op2) composite op definition `A*B` deprecated: please use `A * B` instead (with spaces)"
      end
    end
    return product(op(op1, s...; kwargs...), op(op2, s...; kwargs...))
  end

  common_stypes = _sitetypes(commontags_s)
  @ignore_derivatives push!(common_stypes, SiteType("Generic"))
  opn = OpName(name)

  #
  # Try calling a function of the form:
  #    op(::OpName, ::SiteType, ::Index...; kwargs...)
  #
  for st in common_stypes
    res = op(opn, st, s...; kwargs...)
    if !isnothing(res)
      adjoint && return swapprime(dag(res), 0 => 1)
      return res
    end
  end

  #
  # Try calling a function of the form:
  #    op(::OpName; kwargs...)
  #  for backward compatibility with previous
  #  gate system in PastaQ.jl
  #
  op_mat = op(opn; kwargs...)
  if !isnothing(op_mat)
    rs = reverse(s)
    res = itensor(op_mat, prime.(rs)..., ITensors.dag.(rs)...)
    adjoint && return swapprime(dag(res), 0 => 1)
    return res
  end
  #
  # otherwise try calling a function of the form:
  #    op(::OpName, ::SiteType; kwargs...)
  # which returns a Julia matrix
  #
  for st in common_stypes
    op_mat = op(opn, st; kwargs...)
    if !isnothing(op_mat)
      rs = reverse(s)
      #return itensor(op_mat, prime.(rs)..., ITensors.dag.(rs)...)
      res = itensor(op_mat, prime.(rs)..., ITensors.dag.(rs)...)
      adjoint && return swapprime(dag(res), 0 => 1)
      return res
    end
  end

  # otherwise try calling a function of the form:
  #    op!(::ITensor, ::OpName, ::SiteType, ::Index...; kwargs...)
  #
  Op = ITensor(prime.(s)..., ITensors.dag.(s)...)
  for st in common_stypes
    op!(Op, opn, st, s...; kwargs...)
    if !isempty(Op)
      adjoint && return swapprime(dag(Op), 0 => 1)
      return Op
    end
  end

  if length(s) > 1
    # No overloads for common tags found. It might be a
    # case of making an operator with mixed site types,
    # searching for overloads like:
    #   op(::OpName,
    #      ::SiteType...,
    #      ::Index...;
    #      kwargs...)
    #   op!(::ITensor, ::OpName,
    #       ::SiteType...,
    #       ::Index...;
    #       kwargs...)
    stypes = _sitetypes.(s)

    for st in Iterators.product(stypes...)
      res = op(opn, st..., s...; kwargs...)
      if !isnothing(res)
        adjoint && return swapprime(dag(res), 0 => 1)
        return res
      end
    end

    Op = ITensor(prime.(s)..., ITensors.dag.(s)...)
    for st in Iterators.product(stypes...)
      op!(Op, opn, st..., s...; kwargs...)
      if !isempty(Op)
        adjoint && return swapprime(dag(Op), 0 => 1)
        return Op
      end
    end

    throw(
      ArgumentError(
        "Overload of \"op\" or \"op!\" functions not found for operator name \"$name\" and Index tags: $(tags.(s)).",
      ),
    )
  end

  #
  # otherwise try calling a function of the form:
  #   op(::SiteType, ::Index, ::AbstractString)
  #
  # (Note: this version is for backwards compatibility
  #  after version 0.1.10, and may be eventually
  #  deprecated)
  #
  for st in common_stypes
    res = op(st, s[1], name; kwargs...)
    if !isnothing(res)
      adjoint && return dag(res)
      return res
    end
  end

  return throw(
    ArgumentError(
      "Overload of \"op\" or \"op!\" functions not found for operator name \"$name\" and Index tags: $(tags.(s)).",
    ),
  )
end

op(name::AbstractString; kwargs...) = error("Must input indices when creating an `op`.")

"""
    op(X::AbstractArray, s::Index...)
    op(M::Matrix, s::Index...)

Given a matrix M and a set of indices s,t,... 
return an operator ITensor with matrix elements given
by M and indices s, s', t, t'

# Example

```julia
julia> s = siteind("S=1/2")
(dim=2|id=575|"S=1/2,Site")

julia> Sz = op([1/2 0; 0 -1/2],s)
ITensor ord=2 (dim=2|id=575|"S=1/2,Site")' (dim=2|id=575|"S=1/2,Site")
NDTensors.Dense{Float64, Vector{Float64}}

julia> @show Sz
Sz = ITensor ord=2
Dim 1: (dim=2|id=575|"S=1/2,Site")'
Dim 2: (dim=2|id=575|"S=1/2,Site")
NDTensors.Dense{Float64, Vector{Float64}}
 2×2
 0.5   0.0
 0.0  -0.5
ITensor ord=2 (dim=2|id=575|"S=1/2,Site")' (dim=2|id=575|"S=1/2,Site")
NDTensors.Dense{Float64, Vector{Float64}}
```
"""
op(X::AbstractArray, s::Index...) = itensor(X, prime.([s...]), dag.([s...]))

op(opname, s::Vector{<:Index}; kwargs...) = op(opname, s...; kwargs...)

op(s::Vector{<:Index}, opname; kwargs...) = op(opname, s...; kwargs...)

# For backwards compatibility, version of `op`
# taking the arguments in the other order:
op(s::Index, opname; kwargs...) = op(opname, s; kwargs...)

# To ease calling of other op overloads,
# allow passing a string as the op name
op(opname::AbstractString, t::SiteType; kwargs...) = op(OpName(opname), t; kwargs...)

"""
    op(opname::String,sites::Vector{<:Index},n::Int; kwargs...)

Return an ITensor corresponding to the operator
named `opname` for the n'th Index in the array
`sites`.

# Example

```julia
s = siteinds("S=1/2", 4)
Sz2 = op("Sz", s, 2)
```
"""
function op(opname, s::Vector{<:Index}, ns::NTuple{N,Integer}; kwargs...) where {N}
  return op(opname, ntuple(n -> s[ns[n]], Val(N))...; kwargs...)
end

function op(opname, s::Vector{<:Index}, ns::Vararg{Integer}; kwargs...)
  return op(opname, s, ns; kwargs...)
end

function op(s::Vector{<:Index}, opname, ns::Tuple{Vararg{Integer}}; kwargs...)
  return op(opname, s, ns...; kwargs...)
end

function op(s::Vector{<:Index}, opname, ns::Integer...; kwargs...)
  return op(opname, s, ns; kwargs...)
end

function op(s::Vector{<:Index}, opname, ns::Tuple{Vararg{Integer}}, kwargs::NamedTuple)
  return op(opname, s, ns; kwargs...)
end

function op(s::Vector{<:Index}, opname, ns::Integer, kwargs::NamedTuple)
  return op(opname, s, (ns,); kwargs...)
end

op(s::Vector{<:Index}, o::Tuple) = op(s, o...)

op(o::Tuple, s::Vector{<:Index}) = op(s, o...)

op(f::Function, args...; kwargs...) = f(op(args...; kwargs...))

function op(
  s::Vector{<:Index},
  f::Function,
  opname::AbstractString,
  ns::Tuple{Vararg{Integer}};
  kwargs...,
)
  return f(op(opname, s, ns...; kwargs...))
end

function op(
  s::Vector{<:Index}, f::Function, opname::AbstractString, ns::Integer...; kwargs...
)
  return f(op(opname, s, ns; kwargs...))
end

# Here, Ref is used to not broadcast over the vector of indices
# TODO: consider overloading broadcast for `op` with the example
# here: https://discourse.julialang.org/t/how-to-broadcast-over-only-certain-function-arguments/19274/5
# so that `Ref` isn't needed.
ops(s::Vector{<:Index}, os::AbstractArray) = [op(oₙ, s) for oₙ in os]
ops(os::AbstractVector, s::Vector{<:Index}) = [op(oₙ, s) for oₙ in os]

@doc """
    ops(s::Vector{<:Index}, os::Vector)
    ops(os::Vector, s::Vector{<:Index})

Given a list of operators, create ITensors using the collection
of indices.

# Examples

```julia
s = siteinds("Qubit", 4)
os = [("H", 1), ("X", 2), ("CX", 2, 4)]
# gates = ops(s, os)
gates = ops(os, s)
```
""" ops(::Vector{<:Index}, ::AbstractArray)

#---------------------------------------
#
# state system
#
#---------------------------------------

@eval struct StateName{Name}
  (f::Type{<:StateName})() = $(Expr(:new, :f))
end

StateName(s::AbstractString) = StateName{SmallString(s)}()
StateName(s::SmallString) = StateName{s}()
name(::StateName{N}) where {N} = N

macro StateName_str(s)
  return StateName{SmallString(s)}
end

state(::StateName, ::SiteType; kwargs...) = nothing
state(::StateName, ::SiteType, ::Index; kwargs...) = nothing
state!(::ITensor, ::StateName, ::SiteType, ::Index; kwargs...) = nothing

# Syntax `state("Up", Index(2, "S=1/2"))`
state(sn::String, i::Index; kwargs...) = state(i, sn; kwargs...)

"""
    state(s::Index, name::String; kwargs...)

Return an ITensor corresponding to the state
named `name` for the Index `s`. The returned
ITensor will have `s` as its only index.

The terminology here is based on the idea of a
single-site state or wavefunction in physics.

The `state` function is implemented for various
Index tags by overloading either the
`state` or `state!` methods which take a `SiteType`
argument corresponding to one of the tags of
the Index `s` and an `StateName"name"` argument
that corresponds to the input state name.

The `state` system is used by the MPS type
to construct product-state MPS and for other purposes.

# Example

```julia
s = Index(2, "Site,S=1/2")
sup = state(s,"Up")
sdn = state(s,"Dn")
sxp = state(s,"X+")
sxm = state(s,"X-")
```
"""
function state(s::Index, name::AbstractString; kwargs...)::ITensor
  stypes = _sitetypes(s)
  sname = StateName(name)

  # Try calling state(::StateName"Name",::SiteType"Tag",s::Index; kwargs...)
  for st in stypes
    v = state(sname, st, s; kwargs...)
    if !isnothing(v)
      if v isa ITensor
        return v
      else
        # TODO: deprecate, only for backwards compatibility.
        return itensor(v, s)
      end
    end
  end

  # Try calling state!(::ITensor,::StateName"Name",::SiteType"Tag",s::Index;kwargs...)
  T = ITensor(s)
  for st in stypes
    state!(T, sname, st, s; kwargs...)
    !isempty(T) && return T
  end

  #
  # otherwise try calling a function of the form:
  #    state(::StateName"Name", ::SiteType"Tag"; kwargs...)
  # which returns a Julia vector
  #
  for st in stypes
    v = state(sname, st; kwargs...)
    !isnothing(v) && return itensor(v, s)
  end

  return throw(
    ArgumentError(
      "Overload of \"state\" or \"state!\" functions not found for state name \"$name\" and Index tags $(tags(s))",
    ),
  )
end

state(s::Index, n::Integer) = onehot(s => n)

state(sset::Vector{<:Index}, j::Integer, st; kwargs...) = state(sset[j], st; kwargs...)

#---------------------------------------
#
# val system
#
#---------------------------------------

@eval struct ValName{Name}
  (f::Type{<:ValName})() = $(Expr(:new, :f))
end

ValName(s::AbstractString) = ValName{SmallString(s)}()
ValName(s::SmallString) = ValName{s}()
name(::ValName{N}) where {N} = N

macro ValName_str(s)
  return ValName{SmallString(s)}
end

val(::ValName, ::SiteType) = nothing
val(::AbstractString, ::SiteType) = nothing

"""
    val(s::Index, name::String)

Return an integer corresponding to the `name`
of a certain value the Index `s` can take.
In other words, the `val` function maps strings
to specific integer values within the range `1:dim(s)`.

The `val` function is implemented for various
Index tags by overloading methods named `val`
which take a `SiteType` argument corresponding to
one of the tags of the Index `s` and an `ValName"name"`
argument that corresponds to the input name.

# Example

```julia
s = Index(2, "Site,S=1/2")
val(s,"Up") == 1
val(s,"Dn") == 2

s = Index(2, "Site,Fermion")
val(s,"Emp") == 1
val(s,"Occ") == 2
```
"""
function val(s::Index, name::AbstractString)::Int
  stypes = _sitetypes(s)
  sname = ValName(name)

  # Try calling val(::StateName"Name",::SiteType"Tag",)
  for st in stypes
    res = val(sname, st)
    !isnothing(res) && return res
  end

  return throw(
    ArgumentError("Overload of \"val\" function not found for Index tags $(tags(s))")
  )
end

val(s::Index, n::Integer) = n

val(sset::Vector{<:Index}, j::Integer, st) = val(sset[j], st)

#---------------------------------------
#
# siteind system
#
#---------------------------------------

space(st::SiteType; kwargs...) = nothing

space(st::SiteType, n::Int; kwargs...) = space(st; kwargs...)

function space_error_message(st::SiteType)
  return "Overload of \"space\",\"siteind\", or \"siteinds\" functions not found for Index tag: $(tag(st))"
end

function siteind(st::SiteType; addtags="", kwargs...)
  sp = space(st; kwargs...)
  isnothing(sp) && return nothing
  return Index(sp, "Site, $(tag(st)), $addtags")
end

function siteind(st::SiteType, n; kwargs...)
  s = siteind(st; kwargs...)
  !isnothing(s) && return addtags(s, "n=$n")
  sp = space(st, n; kwargs...)
  isnothing(sp) && error(space_error_message(st))
  return Index(sp, "Site, $(tag(st)), n=$n")
end

siteind(tag::String; kwargs...) = siteind(SiteType(tag); kwargs...)

siteind(tag::String, n; kwargs...) = siteind(SiteType(tag), n; kwargs...)

# Special case of `siteind` where integer (dim) provided
# instead of a tag string
#siteind(d::Integer, n::Integer; kwargs...) = Index(d, "Site,n=$n")
function siteind(d::Integer, n::Integer; addtags="", kwargs...)
  return Index(d, "Site,n=$n, $addtags")
end

#---------------------------------------
#
# siteinds system
#
#---------------------------------------

siteinds(::SiteType, N; kwargs...) = nothing

"""
    siteinds(tag::String, N::Integer; kwargs...)

Create an array of `N` physical site indices of type `tag`.
Keyword arguments can be used to specify quantum number conservation,
see the `space` function corresponding to the site type `tag` for
supported keyword arguments.

# Example

```julia
N = 10
s = siteinds("S=1/2", N; conserve_qns=true)
```
"""
function siteinds(tag::String, N::Integer; kwargs...)
  st = SiteType(tag)

  si = siteinds(st, N; kwargs...)
  if !isnothing(si)
    return si
  end

  return [siteind(st, j; kwargs...) for j in 1:N]
end

"""
    siteinds(f::Function, N::Integer; kwargs...)

Create an array of `N` physical site indices where the site type at site `n` is given
by `f(n)` (`f` should return a string).
"""
function siteinds(f::Function, N::Integer; kwargs...)
  return [siteind(f(n), n; kwargs...) for n in 1:N]
end

# Special case of `siteinds` where integer (dim)
# provided instead of a tag string
"""
    siteinds(d::Integer, N::Integer; kwargs...)

Create an array of `N` site indices, each of dimension `d`.

# Keywords
- `addtags::String`: additional tags to be added to all indices
"""
function siteinds(d::Integer, N::Integer; kwargs...)
  return [siteind(d, n; kwargs...) for n in 1:N]
end

#---------------------------------------
#
# has_fermion_string system
#
#---------------------------------------

has_fermion_string(operator::AbstractArray{<:Number}, s::Index; kwargs...)::Bool = false

has_fermion_string(::OpName, ::SiteType) = nothing

function has_fermion_string(opname::AbstractString, s::Index; kwargs...)::Bool
  opname = strip(opname)

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst(isequal('*'), opname)
  if !isnothing(starpos)
    op1 = opname[1:prevind(opname, starpos)]
    op2 = opname[nextind(opname, starpos):end]
    return xor(has_fermion_string(op1, s; kwargs...), has_fermion_string(op2, s; kwargs...))
  end

  Ntags = length(tags(s))
  stypes = _sitetypes(s)
  opn = OpName(opname)
  for st in stypes
    res = has_fermion_string(opn, st)
    !isnothing(res) && return res
  end
  return false
end
