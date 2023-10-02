function ITensors.sweepup(::Type{<:CuArray})
  GC.gc()
  return CUDA.reclaim()
end
