module NDTensorsHDF5Ext

# HDF5.jl has no do-block form of `open_group`, `create_group`, or `open_dataset`,
# so without this the handles opened below stay alive until their finalizers run.
function closeafter(f, obj)
    try
        return f(obj)
    finally
        close(obj)
    end
end

include("blocksparse.jl")
include("dense.jl")
include("diag.jl")
include("empty.jl")

end # module NDTensorsHDF5Ext
