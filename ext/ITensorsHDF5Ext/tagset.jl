using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors.TagSets: TagSet, tagstring

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, T::TagSet)
    g = create_group(parent, name)
    attributes(g)["type"] = "TagSet"
    attributes(g)["version"] = 1
    return write(g, "tags", tagstring(T))
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{TagSet}
    )
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "TagSet"
        error("HDF5 group '$name' does not contain TagSet data")
    end
    tstring = read(g, "tags")
    return TagSet(tstring)
end
