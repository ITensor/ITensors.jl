using HDF5: HDF5, attributes, create_group, open_group, read, write
using NDTensors: data, Block, blockoffsets, BlockOffsets, BlockSparse

# Helper function for HDF5 write/read of BlockSparse
function offsets_to_array(boff::BlockOffsets{N}) where {N}
    nblocks = length(boff)
    asize = (N + 1) * nblocks
    n = 1
    a = Vector{Int}(undef, asize)
    for bo in pairs(boff)
        for j in 1:N
            a[n] = bo[1][j]
            n += 1
        end
        a[n] = bo[2]
        n += 1
    end
    return a
end

# Helper function for HDF5 write/read of BlockSparse
function array_to_offsets(a, N::Int)
    asize = length(a)
    nblocks = div(asize, N + 1)
    boff = BlockOffsets{N}()
    j = 0
    for b in 1:nblocks
        insert!(boff, Block(ntuple(i -> (a[j + i]), N)), a[j + N + 1])
        j += (N + 1)
    end
    return boff
end

function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::String, B::BlockSparse)
    g = create_group(parent, name)
    attributes(g)["type"] = "BlockSparse{$(eltype(B))}"
    attributes(g)["version"] = 1
    return if eltype(B) != Nothing
        write(g, "ndims", ndims(B))
        write(g, "data", data(B))
        off_array = offsets_to_array(blockoffsets(B))
        write(g, "offsets", off_array)
    end
end

function HDF5.read(
        parent::Union{HDF5.File, HDF5.Group}, name::AbstractString, ::Type{Store}
    ) where {Store <: BlockSparse}
    g = open_group(parent, name)
    ElT = eltype(Store)
    typestr = "BlockSparse{$ElT}"
    if read(attributes(g)["type"]) != typestr
        error("HDF5 group or file does not contain $typestr data")
    end
    N = read(g, "ndims")
    off_array = read(g, "offsets")
    boff = array_to_offsets(off_array, N)
    # Attribute __complex__ is attached to the "data" dataset
    # by the h5 library used by C++ version of ITensor:
    if haskey(attributes(g["data"]), "__complex__")
        M = read(g, "data")
        nelt = size(M, 1) * size(M, 2)
        data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
    else
        data = read(g, "data")
    end
    return BlockSparse(data, boff)
end
