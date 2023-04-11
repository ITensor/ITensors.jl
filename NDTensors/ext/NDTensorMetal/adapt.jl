#
# Used to adapt `EmptyStorage` types
#

@inline NDTensors.mtl(xs; unified::Bool=false) = NDTensors.adapt_structure(MtlArray, xs)
