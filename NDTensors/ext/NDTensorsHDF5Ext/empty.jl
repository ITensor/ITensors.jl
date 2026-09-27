using HDF5: HDF5, attributes, create_group, open_group, read_attribute
using NDTensors: EmptyStorage

# XXX: this seems a bit strange and fragile?
# Takes the type very literally.
# Trailing `kwargs` are used to capture chunking/compression options,
# which are ignored for EmptyStorage.
function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{StoreT}; kwargs...
    ) where {StoreT <: EmptyStorage}
    return closeafter(open_group(parent, name)) do g
        typestr = string(StoreT)
        if read_attribute(g, "type") != typestr
            error("HDF5 group or file does not contain $typestr data")
        end
        return StoreT()
    end
end

# Trailing `kwargs` are used to capture chunking/compression options,
# which are ignored for EmptyStorage.
function HDF5.write(
        parent::Union{HDF5.File, HDF5.Group}, name::String, ::StoreT; kwargs...
    ) where {StoreT <: EmptyStorage}
    return closeafter(create_group(parent, name)) do g
        attributes(g)["type"] = string(StoreT)
        return attributes(g)["version"] = 1
    end
end
