
#"""
#SiteType is a parameterized type which allows
#making Index tags into Julia types. Use cases
#include overloading functions such as `op`,
#`siteinds`, and `state` which generate custom
#operators, Index arrays, and IndexVals associated
#with Index objects having a certain tag.
#
#To make a SiteType type, you can use the string
#macro notation: `SiteType"MyTag"`
#
#To make an SiteType value or object, you can use
#the notation: `SiteType("MyTag")`
#"""
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

#"""
#OpName is a parameterized type which allows
#making strings into Julia types for the purpose
#of representing operator names.
#The main use of OpName is overloading the 
#`ITensors.op!` method which generates operators 
#for indices with certain tags such as "S=1/2".
#
#To make a OpName type, you can use the string
#macro notation: `OpName"MyTag"`. 
#
#To make an OpName value or object, you can use
#the notation: `OpName("myop")`
#"""
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

OpName(s::AbstractString) = OpName{SmallString(s)}()
OpName(s::SmallString) = OpName{s}()
name(::OpName{N}) where {N} = N

macro OpName_str(s)
  OpName{SmallString(s)}
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

Operator names can be combined using the "*"
symbol, for example "S+*S-" or "Sz*Sz*Sz". 
The result is an ITensor made by forming each operator 
then contracting them together in a way corresponding
to the usual operator product or matrix multiplication.

The `op` system is used by the AutoMPO
system to convert operator names into ITensors,
and can be used directly such as for applying
operators to MPS.

# Example
```julia
s = Index(2,"Site,S=1/2")
Sz = op("Sz",s)
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
"""
op(opname::AbstractString,
   s::Vector{<:Index},
   ns::Vararg{Int, N};
   kwargs...) where {N} =
  op(opname, ntuple(n -> s[ns[n]], Val(N))...; kwargs...)

op(s::Vector{<:Index},
   opname::AbstractString,
   ns::Int...;
   kwargs...) =
  op(opname, s, ns...; kwargs...)

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

space(st::SiteType; kwargs...) = throw(MethodError("Overload of \"space\",\"siteind\", or \"siteinds\" functions not found for Index tag: $(tag(st))"))

function siteind(st::SiteType; addtags="", kwargs...) 
  sp = space(st;kwargs...)
  return Index(sp,"Site,$(tag(st)),$addtags")
end

siteind(st::SiteType, n; kwargs...) = addtags(siteind(st; kwargs...),"n=$n")

siteind(tag::String; kwargs...) = siteind(SiteType(tag);kwargs...)

siteind(tag::String,n; kwargs...) = siteind(SiteType(tag),n;kwargs...)

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
