module SmallVectors
using StaticArrays

export AbstractSmallVector, SmallVector, MSmallVector, SubSmallVector, FastCopy, InsertStyle, insert, delete

struct NotImplemented <: Exception
  msg::String
end
NotImplemented() = NotImplemented("Not implemented.")

include("BaseExt/insertstyle.jl")
include("BaseExt/thawfreeze.jl")
## include("DictionariesExt/insert.jl")
## include("SmallVectorsDictionariesExt/interface.jl")
include("abstractarray/insert.jl")
## include("abstractarray/isinsertable.jl")
include("abstractsmallvector/abstractsmallvector.jl")
include("abstractsmallvector/deque.jl")
include("msmallvector/msmallvector.jl")
## include("msmallvector/insertstyle.jl")
include("smallvector/smallvector.jl")
include("smallvector/insertstyle.jl")
include("msmallvector/thawfreeze.jl")
include("smallvector/thawfreeze.jl")
include("subsmallvector/subsmallvector.jl")
end
