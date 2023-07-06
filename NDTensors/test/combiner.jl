using NDTensors
using LinearAlgebra
using Test

# Testing generic block indices
using ITensors: QN, Index

if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  using Metal
end

@testset "CombinerTensor basic functionality" begin
  include("device_list.jl")
  devs = devices_list(copy(ARGS))
  @testset "test device: $dev" for dev in devs
    @testset "Dense * Combiner" begin
      d = 2
      input_tensor_inds = (d, d, d)
      combiner_tensor_inds = (d^2, d, d)
      output_tensor_inds = (d, d^2)

      input_tensor = dev(tensor(Dense(randn(input_tensor_inds)), input_tensor_inds))
      combiner_tensor = dev(tensor(Combiner([1], [1]), combiner_tensor_inds))

      output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
      @test output_tensor isa DenseTensor
      @test dims(output_tensor) == output_tensor_inds
      for i in 1:length(input_tensor)
        @test input_tensor[i] == output_tensor[i]
      end

      # Test uncombining
      new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
      @test new_input_tensor == input_tensor

      # Catch invalid combining
      input_tensor_inds = (d,)
      input_tensor = dev(tensor(Dense(randn(input_tensor_inds)), input_tensor_inds))
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
            randn(dim(input_tensor_inds)), BlockOffsets{3}([Block(1, 1, 1)], [0])
          ),
          input_tensor_inds,
        ),
      )
      combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)

      output_tensor = contract(input_tensor, (1, -1, -2), combiner_tensor, (2, -1, -2))
      @test output_tensor isa BlockSparseTensor
      @test dims(output_tensor) == dims(output_tensor_inds)
      output_tensor = permutedims(output_tensor, (2, 1))
      for i in 1:length(input_tensor)
        @test input_tensor[i] == output_tensor[i]
      end

      # Test uncombining. Broken for inds that are not `Index`.
      new_input_tensor = contract(output_tensor, (1, -1), combiner_tensor, (-1, 2, 3))
      new_input_tensor = permutedims(new_input_tensor, (3, 1, 2))
      @test new_input_tensor == input_tensor

      # Catch invalid combining
      invalid_input_tensor_inds = (k,)
      invalid_input_tensor = dev(
        tensor(
          BlockSparse(
            randn(dim(invalid_input_tensor_inds)), BlockOffsets{1}([Block(1)], [0])
          ),
          invalid_input_tensor_inds,
        ),
      )
      combiner_tensor = tensor(Combiner([1], [1]), combiner_tensor_inds)
      @test_throws Any contract(invalid_input_tensor, (-1,), combiner_tensor, (1, 2, -1))
    end
  end
end
