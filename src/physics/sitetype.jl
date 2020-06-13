
"""
SiteType is a parameterized type which allows
making Index tags into Julia types. Use cases
include overloading functions such as `op`,
`siteinds`, and `state` which generate custom
operators, Index arrays, and IndexVals associated
with Index objects having a certain tag.

To make a SiteType type, you can use the string
macro notation: `SiteType"MyTag"`

To make an SiteType value or object, you can use
the notation: `SiteType("MyTag")`
"""
struct SiteType{T}
end

SiteType(s::Tag) = SiteType{s}()
SiteType(s::AbstractString) = SiteType(Tag(s))

macro SiteType_str(s)
  SiteType{Tag(s)}
end

# Keep TagType defined for backwards
# compatibility; will be deprecated later
const TagType = SiteType
macro TagType_str(s)
  TagType{Tag(s)}
end

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
struct OpName{Name}
end

OpName(s::SmallString) = OpName{s}()
OpName(s::AbstractString) = OpName(SmallString(s))

macro OpName_str(s)
  OpName{SmallString(s)}
end


"""
    op(opname::String,s::Index; kwargs...)

Return an ITensor corresponding to the operator
named `opname` for the Index `s`. The operator
is constructed by calling an overload of either
the `op` or `op!` methods which take a `SiteType`
argument that corresponds to one of the tags of
the Index `s`.

The `op` system is used by the AutoMPO
system to convert operator names into ITensors,
and can be used directly such as for applying
operators to MPS.

# Example
```julia
s = Index(2,"S=1/2,Site")
Sz = op("Sz",s)
```
"""
function op(opname::AbstractString,
            s::Index;
            kwargs...)

  opname = strip(opname)

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst("*",opname)
  if !isnothing(starpos)
    op1 = opname[1:starpos.start-1]
    op2 = opname[starpos.start+1:end]
    return product(op(op1,s;kwargs...),op(op2,s;kwargs...))
  end

  #
  # Try calling a function of the form:
  #    op(::SiteType,::OpName,::Index;kwargs...)
  #
  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op,Tuple{SiteType{tagn},OpName{SmallString(opname)},Index})
      return op(SiteType(tagn),OpName(opname),s;kwargs...)
    end
  end

  # otherwise try calling a function of the form:
  #    op!(::ITensor,::SiteType,::OpName,::Index;kwargs...)
  #
  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op!,Tuple{ITensor,SiteType{tagn},OpName{SmallString(opname)},Index})
      Op = emptyITensor(s',dag(s))
      op!(Op,SiteType(tagn),OpName(opname),s;kwargs...)
      return Op
    end
  end

  #
  # otherwise try calling a function of the form:
  #   op(::SiteType,::Index,::AbstractString)
  #
  # (Note: this version is for backwards compatibility
  #  after version 0.1.10, and may be eventually
  #  deprecated)
  #

  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op,Tuple{SiteType{tagn},Index,AbstractString})
      return op(SiteType(tagn),s,opname;kwargs...)
    end
  end

  throw(ArgumentError("Overload of \"op\" or \"op!\" functions not found for operator name \"$opname\" and Index tags $(tags(s))"))
end

# For backwards compatibility, version of `op`
# taking the arguments in the other order:
op(s::Index,opname::AbstractString;kwargs...) = op(opname,s;kwargs...)


"""
  op(opname::String,sites::Vector{<:Index},n::Int; kwargs...)

Return an ITensor corresponding to the operator
named `opname` for the n'th Index in the array 
`sites`.
"""
function op(opname::AbstractString,
            s::Vector{<:Index},
            n::Int;
            kwargs...)::ITensor
  return op(s[n],opname;kwargs...)
end

op(s::Vector{<:Index},
   opname::AbstractString,
   n::Int;
   kwargs...) = op(opname,s,n;kwargs...)


state(s::Index,n::Integer) = s[n]

function state(s::Index,
               str::String)::IndexVal
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    SType = SiteType{tags(s)[n]}
    if hasmethod(state,Tuple{SType,AbstractString})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    throw(ArgumentError("Overload of \"state\" function not found for Index tags $(tags(s))"))
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"state\""))
  end
  st = SiteType(tags(s)[use_tag])
  sn = state(st,str)
  return s[sn]
end

function state(sset::Vector{<:Index},
               j::Integer,
               st)::IndexVal
  return state(sset[j],st)
end

function siteinds(d::Integer,
                  N::Integer; kwargs...)
  return [Index(d,"Site,n=$n") for n=1:N]
end

function siteinds(str::String,
                  N::Integer; kwargs...)
  SType = SiteType{Tag(str)}
  if !hasmethod(siteinds,Tuple{SType,Int})
    throw(ArgumentError("Overload of \"siteinds\" function not found for tag type \"$str\""))
  end
  return siteinds(SType(),N; kwargs...)
end

function has_fermion_string(s::Index,
                            opname::AbstractString;
                            kwargs...)::Bool
  opname = strip(opname)
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    SType = SiteType{tags(s)[n]}
    if hasmethod(has_fermion_string,Tuple{SType,Index,AbstractString})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    return false
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"has_fermion_string\""))
  end
  st = SiteType(tags(s)[use_tag])
  return has_fermion_string(st,s,opname;kwargs...)
end
