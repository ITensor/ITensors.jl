module Sectors

using ..SortedSets
import ..SortedSets: AbstractSmallSet

include("abstractcategory.jl")
include("group_definitions.jl")
include("category_definitions.jl")

include("namedtuple_operations.jl")
include("sector.jl")

end
