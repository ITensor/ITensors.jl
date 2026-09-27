using HDF5: HDF5, attributes, create_group, open_group, read, read_attribute, write
using ITensors.TagSets: TagSet, tagstring

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, T::TagSet)
    return closeafter(create_group(parent, name)) do g
        attributes(g)["type"] = "TagSet"
        attributes(g)["version"] = 1
        return write(g, "tags", tagstring(T))
    end
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{TagSet}
    )
    return closeafter(open_group(parent, name)) do g
        if read_attribute(g, "type") != "TagSet"
            error("HDF5 group '$name' does not contain TagSet data")
        end
        tstring = read(g, "tags")
        return TagSet(tstring)
    end
end
