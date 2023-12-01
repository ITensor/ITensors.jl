module SparseArrayInterface
include("sparsearrayinterface/densearray.jl")
include("sparsearrayinterface/interface.jl")
include("sparsearrayinterface/interface_optional.jl")
include("sparsearrayinterface/indexing.jl")
include("sparsearrayinterface/base.jl")
include("sparsearrayinterface/map.jl")
include("sparsearrayinterface/copyto.jl")
include("sparsearrayinterface/broadcast.jl")
include("sparsearrayinterface/conversion.jl")
include("sparsearrayinterface/wrappers.jl")
include("sparsearrayinterface/zero.jl")
include("sparsearrayinterface/SparseArrayInterfaceLinearAlgebraExt.jl")
include("abstractsparsearray/abstractsparsearray.jl")
include("abstractsparsearray/sparsearrayinterface.jl")
include("abstractsparsearray/base.jl")
include("abstractsparsearray/broadcast.jl")
include("abstractsparsearray/map.jl")
include("abstractsparsearray/baseinterface.jl")
include("abstractsparsearray/convert.jl")
include("abstractsparsearray/SparseArrayInterfaceSparseArraysExt.jl")
include("abstractsparsearray/SparseArrayInterfaceLinearAlgebraExt.jl")
end
