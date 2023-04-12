using Preferences

function enable_cuda!(use_cuda::Bool)
  # Set it in our runtime values, as well as saving it to disk
  @set_preferences!("cuda_enabled" => use_cuda)
  @info("CUDA testing enabled")
end

const use_cuda = @load_preference("cuda_enabled", false)

function enable_metal!(use_mtl::Bool)
  @set_preferences!("metal_enabled" => use_mtl)
  @info("Metal testing enabled")
end

const use_mtl = @load_preference("metal_enabled", false)