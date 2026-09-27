@eval module $(gensym())
using HDF5: HDF5, h5open, read, write
using NDTensors:
    BlockSparse, Diag, blockoffsets, randomBlockSparseTensor, randomTensor, storage, tensor
using Test: @test, @testset

@testset "HDF5 read and write" begin
    nopen(file) = HDF5.API.h5f_get_obj_count(file, HDF5.API.H5F_OBJ_ALL)
    stores = [
        "dense_real" => storage(randomTensor((3, 4))),
        "dense_complex" => storage(randomTensor(ComplexF64, (3, 4))),
        "diag_real" => storage(tensor(Diag(randn(4)), (4, 4))),
        "diag_complex" => storage(tensor(Diag(randn(ComplexF64, 4)), (4, 4))),
        "blocksparse_real" =>
            storage(randomBlockSparseTensor([(1, 2), (2, 1)], ([2, 3], [4, 5]))),
        "blocksparse_complex" => storage(
            randomBlockSparseTensor(ComplexF64, [(1, 2), (2, 1)], ([2, 3], [4, 5]))
        ),
    ]
    mktempdir() do dir
        fn = joinpath(dir, "data.h5")
        # Collect unrelated finalizers so they can't skew the open-handle counts below.
        GC.gc()
        for (name, S) in stores
            h5open(fn, "w") do fo
                write(fo, name, S)
                @test nopen(fo) == 1
            end
            h5open(fn, "r") do fi
                R = read(fi, name, typeof(S))
                @test typeof(R) == typeof(S)
                @test R ≈ S
                if S isa BlockSparse
                    @test blockoffsets(R) == blockoffsets(S)
                end
                @test nopen(fi) == 1
            end
        end
    end
end
end
