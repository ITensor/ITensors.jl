using HDF5: HDF5, attributes, create_group, open_group, read, write
using NDTensors: EmptyStorage

# XXX: this seems a bit strange and fragile?
# Takes the type very literally.
function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{StoreT}
    ) where {StoreT <: EmptyStorage}
    g = open_group(parent, name)
    typestr = string(StoreT)
    if read(attributes(g)["type"]) != typestr
        error("HDF5 group or file does not contain $typestr data")
    end
    return StoreT()
end

function HDF5.write(
        parent::Union{HDF5.File, HDF5.Group}, name::String, ::StoreT
    ) where {StoreT <: EmptyStorage}
    g = create_group(parent, name)
    attributes(g)["type"] = string(StoreT)
    return attributes(g)["version"] = 1
end
