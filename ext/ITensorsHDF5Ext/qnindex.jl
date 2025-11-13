using HDF5: HDF5, attributes, create_group, open_group, read, write
using ITensors: dims, QNBlock, QNBlocks

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, B::QNBlocks)
    g = create_group(parent, name)
    attributes(g)["type"] = "QNBlocks"
    attributes(g)["version"] = 1
    write(g, "length", length(B))
    dims = [block[2] for block in B]
    write(g, "dims", dims)
    for n in 1:length(B)
        write(g, "QN[$n]", B[n][1])
    end
    return nothing
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{QNBlocks}
    )
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "QNBlocks"
        error("HDF5 group or file does not contain QNBlocks data")
    end
    N = read(g, "length")
    dims = read(g, "dims")
    B = QNBlocks(undef, N)
    for n in 1:length(B)
        B[n] = QNBlock(read(g, "QN[$n]", QN), dims[n])
    end
    return B
end
