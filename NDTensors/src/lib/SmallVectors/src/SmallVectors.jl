module SmallVectors
using StaticArrays

export AbstractSmallVector,
  SmallVector,
  MSmallVector,
  SubSmallVector,
  FastCopy,
  InsertStyle,
  IsInsertable,
  NotInsertable,
  insert,
  delete,
  thaw,
  freeze,
  maxlength,
  unionsortedunique,
  unionsortedunique!,
  setdiffsortedunique,
  setdiffsortedunique!,
  intersectsortedunique,
  intersectsortedunique!,
  symdiffsortedunique,
  symdiffsortedunique!,
  thaw_type

struct NotImplemented <: Exception
  msg::String
end
NotImplemented() = NotImplemented("Not implemented.")

include("BaseExt/insertstyle.jl")
include("BaseExt/thawfreeze.jl")
include("BaseExt/sort.jl")
include("BaseExt/sortedunique.jl")
include("abstractarray/insert.jl")
include("abstractsmallvector/abstractsmallvector.jl")
include("abstractsmallvector/deque.jl")
include("msmallvector/msmallvector.jl")
include("smallvector/smallvector.jl")
include("smallvector/insertstyle.jl")
include("msmallvector/thawfreeze.jl")
include("smallvector/thawfreeze.jl")
include("subsmallvector/subsmallvector.jl")
end
