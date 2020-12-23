
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
you can follow the instructions [here](http://itensor.org/docs.cgi?vers=julia&page=formulas/sitetype_extending).
To create new site types, you can follow the instructions
[here](https://itensor.org/docs.cgi?vers=julia&page=formulas/sitetype_basic) and 
[here](https://itensor.org/docs.cgi?vers=julia&page=formulas/sitetype_qns).

The current built-in site types are:

- `SiteType"S=1/2"` (or `SiteType"S=½"`)
- `SiteType"S=1"`
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
- `SiteType"Electron"`: `"Nup"`, `"Ndn"`, `"Nupdn"`, `"Ntot"`, `"Cup"`, `"Cdagup"`, `"Cdn"`, `"Cdagup"`, `"Sz"`, `"Sx"`, `"Sy"`, `"S+"`, `"S-"`, ...
- ...

You can view the source code for the internal SiteType definitions
and operators that are defined [here](https://github.com/ITensor/ITensors.jl/tree/master/src/physics/site_types).
"""
SiteType(s::AbstractString) = SiteType{Tag(s)}()

SiteType(t::Tag) = SiteType{t}()

tag(::SiteType{T}) where {T} = T

macro SiteType_str(s)
  SiteType{Tag(s)}
end

# Keep TagType defined for backwards
# compatibility; will be deprecated later
const TagType = SiteType
macro TagType_str(s)
  TagType{Tag(s)}
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
  OpName{Symbol(s)}
end

# Default implementations of op and op!
op(::OpName, ::SiteType, ::Index...; kwargs...) = nothing
op(::OpName, ::SiteType, ::SiteType,
   sitetypes_inds::Union{SiteType, Index}...; kwargs...) = nothing
op!(::ITensor, ::OpName, ::SiteType, ::Index...; kwargs...) = nothing 
op!(::ITensor, ::OpName, ::SiteType, ::SiteType,
    sitetypes_inds::Union{SiteType, Index}...; kwargs...) = nothing 

# Deprecated version, for backwards compatibility
op(::SiteType, ::Index, ::AbstractString; kwargs...) = nothing

function _sitetypes(ts::TagSet)
  Ntags = length(ts)
  return SiteType[SiteType(ts[n]) for n in 1:Ntags]
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

The `op` system is used by the AutoMPO
system to convert operator names into ITensors,
and can be used directly such as for applying
operators to MPS.

# Example
```julia
s = Index(2, "Site,S=1/2")
Sz = op("Sz", s)
```
"""
function op(name::AbstractString,
            s::Index...;
            kwargs...)

  name = strip(name)

  # TODO: filter out only commons tags
  # if there are multiple indices
  commontags_s = commontags(s...)

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst("*", name)
  if !isnothing(starpos)
    op1 = name[1:starpos.start-1]
    op2 = name[starpos.start+1:end]
    return product(op(op1, s...; kwargs...),
                   op(op2, s...; kwargs...))
  end

  common_stypes  = _sitetypes(commontags_s)
  push!(common_stypes,SiteType("Generic"))
  opn = OpName(name)

  #
  # Try calling a function of the form:
  #    op(::OpName, ::SiteType, ::Index; kwargs...)
  #
  for st in common_stypes
    res = op(opn, st, s...; kwargs...)
    if !isnothing(res)
      return res
    end
  end

  # otherwise try calling a function of the form:
  #    op!(::ITensor, ::OpName, ::SiteType, ::Index; kwargs...)
  #
  Op = emptyITensor(prime.(s)..., dag.(s)...)
  for st in common_stypes
    op!(Op, opn, st, s...; kwargs...)
    if !isempty(Op)
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
        return res
      end
    end

    Op = emptyITensor(prime.(s)..., dag.(s)...)
    for st in Iterators.product(stypes...)
      op!(Op, opn, st..., s...; kwargs...)
      if !isempty(Op)
        return Op
      end
    end
    error("Older op interface does not support multiple indices with mixed site types. You may want to overload `op(::OpName, ::SiteType..., ::Index...)` or `op!(::ITensor, ::OpName, ::SiteType..., ::Index...) for the operator \"$name\" and Index tags $(tags.(s)).")
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
      return res
    end
  end

  throw(ArgumentError("Overload of \"op\" or \"op!\" functions not found for operator name \"$name\" and Index tags: $(commontags_s))"))
end

# For backwards compatibility, version of `op`
# taking the arguments in the other order:
op(s::Index,
   opname::AbstractString;
   kwargs...) = op(opname, s; kwargs...)


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
op(opname::AbstractString, s::Vector{<:Index},
   ns::Vararg{Int, N}; kwargs...) where {N} =
  op(opname, ntuple(n -> s[ns[n]], Val(N))...; kwargs...)

op(s::Vector{ <: Index}, opname::AbstractString,
   ns::Int...; kwargs...) =
  op(opname, s, ns...; kwargs...)

op(s::Vector{ <: Index}, opname::AbstractString,
   ns::Tuple{Vararg{Int}}, kwargs::NamedTuple) =
  op(opname, s, ns...; kwargs...)

op(s::Vector{ <: Index}, opname::AbstractString,
   ns::Int, kwargs::NamedTuple) =
  op(opname, s, ns; kwargs...)

# This version helps with call like `op.(Ref(s), os)` where `os`
# is a vector of tuples.
op(s::Vector{ <: Index}, os::Tuple{String, Vararg}) =
  op(s, os...)

# Here, Ref is used to not broadcast over the vector of indices
# TODO: consider overloading broadcast for `op` with the example
# here: https://discourse.julialang.org/t/how-to-broadcast-over-only-certain-function-arguments/19274/5
# so that `Ref` isn't needed.
ops(s::Vector{ <: Index},
    os::AbstractArray{ <: Tuple{String, Vararg}}) =
  op.(Ref(s), os)

ops(os::Vector{ <: Tuple{String, Vararg}}, s::Vector{ <: Index}) =
  op.(Ref(s), os)

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
  StateName{SmallString(s)}
end

state(::SiteType, ::StateName) = nothing
state(::SiteType, ::AbstractString) = nothing

function state(s::Index,
               name::AbstractString)::IndexVal
  stypes  = _sitetypes(s)
  sname = StateName(name)

  # Try calling state(::SiteType"Tag",::StateName"Name")
  for st in stypes
    res = state(st,sname)
    !isnothing(res) && return s(res)
  end

  # Try calling state(::SiteType"Tag","Name")
  for st in stypes
    res = state(st,name)
    !isnothing(res) && return s(res)
  end

  throw(ArgumentError("Overload of \"state\" function not found for Index tags $(tags(s))"))
end

state(s::Index,n::Integer) = s[n]

state(sset::Vector{<:Index},j::Integer,st) = state(sset[j],st)

#---------------------------------------
#
# siteind system
#
#---------------------------------------

space(st::SiteType; kwargs...) = nothing

space(st::SiteType, n::Int; kwargs...) =
  space(st; kwargs...)

function space_error_message(st::SiteType)
  return "Overload of \"space\",\"siteind\", or \"siteinds\" functions not found for Index tag: $(tag(st))"
end

function siteind(st::SiteType; addtags = "", kwargs...) 
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

siteind(tag::String; kwargs...) =
  siteind(SiteType(tag); kwargs...)

siteind(tag::String, n; kwargs...) =
  siteind(SiteType(tag), n; kwargs...)

# Special case of `siteind` where integer (dim) provided
# instead of a tag string
siteind(d::Integer,n::Integer; kwargs...) = Index(d,"Site,n=$n")

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
"""
function siteinds(tag::String,
                  N::Integer; kwargs...)
  st = SiteType(tag)

  si = siteinds(st,N;kwargs...)
  if !isnothing(si)
    return si
  end

  return [siteind(st,j; kwargs...) for j=1:N]
end

"""
  siteinds(f::Function, N::Integer; kwargs...)

Create an array of `N` physical site indices where the site type at site `n` is given
by `f(n)` (`f` should return a string).
"""
function siteinds(f::Function,
                  N::Integer; kwargs...)
  [siteind(f(n),n; kwargs...) for n=1:N]
end

# Special case of `siteinds` where integer (dim)
# provided instead of a tag string
function siteinds(d::Integer,
                  N::Integer; kwargs...)
  return [siteind(d,n; kwargs...) for n=1:N]
end

#---------------------------------------
#
# has_fermion_string system
#
#---------------------------------------

has_fermion_string(::OpName, ::SiteType) = nothing

function has_fermion_string(opname::AbstractString,
                            s::Index;
                            kwargs...)::Bool
  opname = strip(opname)

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst("*", opname)
  if !isnothing(starpos)
    op1 = opname[1:starpos.start-1]
    op2 = opname[starpos.start+1:end]
    return xor(has_fermion_string(op1,s; kwargs...),
               has_fermion_string(op2,s; kwargs...))
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
