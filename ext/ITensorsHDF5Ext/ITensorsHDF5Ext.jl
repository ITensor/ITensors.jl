module ITensorsHDF5Ext

# HDF5.jl has no do-block form of `open_group`, `create_group`, or `open_dataset`,
# so without this the handles opened below stay alive until their finalizers run.
function closeafter(f, obj)
    try
        return f(obj)
    finally
        close(obj)
    end
end

include("index.jl")
include("itensor.jl")
include("qnindex.jl")
include("indexset.jl")
include("qn.jl")
include("tagset.jl")
end
