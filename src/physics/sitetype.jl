
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


function call_op(opname::AbstractString,
                 s::Index;
                 kwargs...)

  #
  # Try calling a function of the form:
  #    op(::SiteType,::OpName,::Index;kwargs...)
  #
  usetag = Tag()
  nfound = 0
  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op,Tuple{SiteType{tagn},OpName{SmallString(opname)},Index})
      usetag = tagn
      nfound += 1
    end
  end
  if nfound == 1
    return op(SiteType(usetag),OpName(opname),s;kwargs...)
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"ITensors.op\""))
  end

  # otherwise, nfound == 0, so
  #
  # Try calling a function of the form:
  #    op!(::ITensor,::SiteType,::OpName,::Index;kwargs...)
  #
  usetag = Tag()
  nfound = 0
  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op!,Tuple{ITensor,SiteType{tagn},OpName{SmallString(opname)},Index})
      usetag = tagn
      nfound += 1
    end
  end
  if nfound == 1
    Op = emptyITensor(s',dag(s))
    op!(Op,SiteType(usetag),OpName(opname),s;kwargs...)
    return Op
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"ITensors.op!\""))
  end

  # otherwise, nfound == 0, so
  #
  # Try calling a function of the form:
  #   op(::SiteType,::Index,::AbstractString)
  #
  # (Note: this version is for backwards compatibility
  #  after version 0.1.10, and may be eventually
  #  deprecated)
  #

  usetag = Tag()
  nfound = 0
  for n=1:length(tags(s))
    tagn = tags(s)[n]
    if hasmethod(op,Tuple{SiteType{tagn},Index,AbstractString})
      usetag = tagn
      nfound += 1
    end
  end
  if nfound == 1
    return op(SiteType(usetag),s,opname;kwargs...)
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"ITensors.op\""))
  end

  throw(ArgumentError("Overload of \"op\" or \"op!\" functions not found for operator name \"$opname\" and Index tags $(tags(s))"))
end


function op(opname::AbstractString,
            s::Index;
            kwargs...)::ITensor

  opname = strip(opname)

  if opname == "Id"
    Op = emptyITensor(dag(s),s')
    for n=1:dim(s)
      Op[dag(s)(n),s'(n)] = 1.0
    end
    return Op
  end

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst("*",opname)
  if !isnothing(starpos)
    op1 = opname[1:starpos.start-1]
    op2 = opname[starpos.start+1:end]
    return product(op(s,op1;kwargs...),op(s,op2;kwargs...))
  end

  return call_op(opname,s;kwargs...)
end

# For backwards compatibility, version of `op`
# taking the arguments in the other order:
op(s::Index,opname::AbstractString;kwargs...) = op(opname,s;kwargs...)

# Version of `op` taking an array of indices
# and an integer of which Index to use
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
