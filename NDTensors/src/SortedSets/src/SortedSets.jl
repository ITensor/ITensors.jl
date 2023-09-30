module SortedSets
using Dictionaries
using ..SmallVectors

using Base: @propagate_inbounds
using Base.Order: Ordering, Forward
using Random

export AbstractWrappedIndices, SortedSet

include("DictionariesExt/insert.jl")
include("DictionariesExt/isinsertable.jl")
include("abstractwrappedset.jl")
include("SmallVectorsDictionariesExt/interface.jl")
include("sortedset.jl")

end
