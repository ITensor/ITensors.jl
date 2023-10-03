module SortedSets
using Dictionaries
using ..SmallVectors

using Base: @propagate_inbounds
using Base.Order: Ordering, Forward, ord, lt

using Random

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
