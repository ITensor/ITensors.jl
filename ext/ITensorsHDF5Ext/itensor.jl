using HDF5: HDF5, attributes, create_group, open_group, read, read_attribute, write
using ITensors: ITensor, inds, itensor, storage
using NDTensors:
    NDTensors, BlockSparse, Combiner, Dense, Diag, DiagBlockSparse, EmptyStorage

function HDF5.write(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, T::ITensor; kwargs...
    )
    return closeafter(create_group(parent, name)) do g
        attributes(g)["type"] = "ITensor"
        attributes(g)["version"] = 1
        write(g, "inds", inds(T))
        return write(g, "storage", storage(T); kwargs...)
    end
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{ITensor}; kwargs...
    )
    return closeafter(open_group(parent, name)) do g
        if read_attribute(g, "type") != "ITensor"
            error("HDF5 group or file does not contain ITensor data")
        end
        inds = read(g, "inds", Vector{<:Index})

        # check input file for key name of ITensor data
        # ITensors.jl <= v0.1.x uses `store` as key
        # whereas ITensors.jl >= v0.2.x uses `storage` as key
        for key in ["storage", "store"]
            if haskey(g, key)
                stypestr = closeafter(open_group(g, key)) do sg
                    return read_attribute(sg, "type")
                end
                stype = eval(Meta.parse(stypestr))
                storage = read(g, key, stype; kwargs...)
                return itensor(storage, inds)
            end
        end
        return error("HDF5 file: $(g) does not contain correct ITensor data.\nNeither key
             `store` nor `storage` could be found.")
    end
end
