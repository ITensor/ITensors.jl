using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: Arrow, dim, dir, id, Index, plev, QNBlocks, space, tags, TagSet

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, I::Index)
    g = create_group(parent, name)
    attributes(g)["type"] = "Index"
    attributes(g)["version"] = 1
    write(g, "id", id(I))
    write(g, "dim", dim(I))
    write(g, "dir", Int(dir(I)))
    write(g, "tags", tags(I))
    write(g, "plev", plev(I))
    return if typeof(space(I)) == Int
        attributes(g)["space_type"] = "Int"
    elseif typeof(space(I)) == QNBlocks
        attributes(g)["space_type"] = "QNBlocks"
        write(g, "space", space(I))
    else
        error("Index space type not recognized")
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{Index})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "Index"
        error("HDF5 group or file does not contain Index data")
    end
    id = read(g, "id")
    dim = read(g, "dim")
    dir = Arrow(read(g, "dir"))
    tags = read(g, "tags", TagSet)
    plev = read(g, "plev")
    space_type = "Int"
    if haskey(attributes(g), "space_type")
        space_type = read(attributes(g)["space_type"])
    end
    if space_type == "Int"
        space = dim
    elseif space_type == "QNBlocks"
        space = read(g, "space", QNBlocks)
    end
    return Index(id, space, dir, tags, plev)
end
