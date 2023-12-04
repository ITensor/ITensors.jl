module Sectors

using ..SortedSets
import ..SortedSets: AbstractSmallSet
using InlineStrings
using HalfIntegers

import LinearAlgebra: norm
import Base: *, ==, getindex, intersect, union, setdiff, symdiff, iterate, length, show
#import NDTensors: ⊗
#import NDTensors.SortedSets: AbstractSmallSet

export ⊗, ⊕, U, SU, SUd, SUz, Z, Fib, Ising, Sector, nactive, show

include("category.jl")
include("category_definitions.jl")
include("label.jl")
include("sector.jl")

end
