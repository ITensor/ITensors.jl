function ITensors.sweepup(::Type{<:CuArray})
    GC.gc()
    CUDA.reclaim()
end