@eval module $(gensym())
using GPUArraysCore: @allowscalar
using NDTensors:
    NDTensors,
    Block,
    BlockOffsets,
    BlockSparse,
    BlockSparseTensor,
    Combiner,
    Dense,
    DenseTensor,
    contract,
    dim,
    dims,
    tensor
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: devices_list, is_supported_eltype
using Test: @testset, @test, @test_throws

# Testing generic block indices
struct Index{Space}
    space::Space
end
NDTensors.dim(i::Index) = sum(b -> last(b), i.space)
NDTensors.nblocks(i::Index) = length(i.space)
NDTensors.blockdim(i::Index, block::Integer) = last(i.space[block])
function NDTensors.outer(i1::Index, i2::Index)
    return Index(
        vec(
            map(Iterators.product(i1.space, i2.space)) do (b1, b2)
                return first(b1) + first(b2) => last(b1) * last(b2)
            end,
        )
    )
end
NDTensors.permuteblocks(i::Index, perm::Vector{Int}) = Index(i.space[perm])

struct QN end
Base.:+(q1::QN, q2::QN) = QN()

@testset "CombinerTensor basic functionality" begin
    @testset "test device: $dev, eltype: $elt" for dev in devices_list(copy(ARGS)),
            elt in (Float64, Float32)

        if !is_supported_eltype(dev, elt)
            continue
        end
        @testset "Dense * Combiner" begin
            d = 2
            input_tensor_inds = (d, d, d)
            combiner_tensor_inds = (d^2, d, d)
            output_tensor_inds = (d, d^2)

            input_tensor = dev(tensor(Dense(randn(elt, input_tensor_inds)), input_tensor_inds))
            combiner_tensor = dev(tensor(Combiner([1], [1]), combiner_tensor_inds))

            output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
            @test output_tensor isa DenseTensor
            @test dims(output_tensor) == output_tensor_inds
            @allowscalar for i in 1:length(input_tensor)
                @test input_tensor[i] == output_tensor[i]
            end

            # Test uncombining
            new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
            @test NDTensors.cpu(new_input_tensor) == NDTensors.cpu(input_tensor)

            # Catch invalid combining
            input_tensor_inds = (d,)
            input_tensor = dev(tensor(Dense(randn(elt, input_tensor_inds)), input_tensor_inds))
            combiner_tensor = dev(tensor(Combiner([1], [1]), combiner_tensor_inds))
            @test_throws Any contract(input_tensor, (-1,), combiner_tensor, (1, -1, -2))
        end

        ind_constructors = (dim -> [dim], dim -> Index([QN() => dim]))
        #TODO cu doesn't work with blocksparse yet
        @testset "BlockSparse * Combiner" for ind_constructor in ind_constructors
            d = 2
            i, j, k = map(ind_constructor, (d, d, d))
            c = ind_constructor(d^2)

            input_tensor_inds = (i, j, k)
            combiner_tensor_inds = (c, j, k)
            output_tensor_inds = (c, i)

            input_tensor = dev(
                tensor(
                    BlockSparse(
                        randn(elt, dim(input_tensor_inds)), BlockOffsets{3}([Block(1, 1, 1)], [0])
                    ),
                    input_tensor_inds,
                ),
            )
            combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)

            output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
            @test output_tensor isa BlockSparseTensor
            @test dims(output_tensor) == dims(output_tensor_inds)
            output_tensor = permutedims(output_tensor, (2, 1))
            @allowscalar for i in 1:length(input_tensor)
                @test input_tensor[i] == output_tensor[i]
            end

            # Test uncombining. Broken for inds that are not `Index`.
            new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
            new_input_tensor = permutedims(new_input_tensor, (3, 1, 2))
            @test NDTensors.cpu(new_input_tensor) == NDTensors.cpu(input_tensor)

            # Catch invalid combining
            invalid_input_tensor_inds = (k,)
            invalid_input_tensor = dev(
                tensor(
                    BlockSparse(
                        randn(elt, dim(invalid_input_tensor_inds)), BlockOffsets{1}([Block(1)], [0])
                    ),
                    invalid_input_tensor_inds,
                ),
            )
            combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)
            @test_throws Any contract(invalid_input_tensor, (-1,), combiner_tensor, (1, 2, -1))
        end
    end
end
end
