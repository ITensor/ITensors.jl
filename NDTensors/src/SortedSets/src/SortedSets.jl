module SortedSets
using Compat
using Dictionaries
using Random
using ..SmallVectors

using Base: @propagate_inbounds
using Base.Order: Ordering, Forward, ord, lt

export AbstractWrappedIndices, SortedSet, SmallSet, MSmallSet

include("BaseExt/sorted.jl")
include("DictionariesExt/insert.jl")
include("DictionariesExt/isinsertable.jl")
include("abstractset.jl")
include("abstractwrappedset.jl")
include("SmallVectorsDictionariesExt/interface.jl")
include("sortedset.jl")
include("SortedSetsSmallVectorsExt/smallset.jl")

end
