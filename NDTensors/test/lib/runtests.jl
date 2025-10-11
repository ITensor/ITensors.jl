@eval module $(gensym())
using NDTensors: NDTensors
using Test: @testset
@testset "Test NDTensors lib $lib" for lib in [
        "AMDGPUExtensions",
        "BackendSelection",
        "CUDAExtensions",
        "GPUArraysCoreExtensions",
        "MetalExtensions",
        "Expose",
    ]
    include(joinpath(pkgdir(NDTensors), "src", "lib", lib, "test", "runtests.jl"))
end
end
