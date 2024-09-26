module Sectors

using BlockArrays: blocklengths
using HalfIntegers: Half, HalfInteger, half, twice

using NDTensors.LabelledNumbers:
  LabelledInteger, label, label_type, labelled, unlabel, unlabel_type
using NDTensors.GradedAxes:
  GradedAxes,
  blocklabels,
  dual,
  fuse_blocklengths,
  fusion_product,
  gradedrange,
  tensor_product

include("symmetry_style.jl")
include("abstractcategory.jl")
include("category_definitions/fib.jl")
include("category_definitions/ising.jl")
include("category_definitions/o2.jl")
include("category_definitions/su.jl")
include("category_definitions/su2k.jl")
include("category_definitions/u1.jl")
include("category_definitions/zn.jl")

include("namedtuple_operations.jl")
include("category_product.jl")

end
