using Preferences

function enable_cuda!(cuda_enabled::Bool)
  # Set it in our runtime values, as well as saving it to disk
  @set_preferences!("cuda_enabled" => cuda_enabled)
  @info("CUDA testing enabled")
end

const cuda_enabled = @load_preference("cuda_enabled", false)

function enable_metal!(metal_enabled::Bool)
  @set_preferences!("metal_enabled" => metal_enabled)
  @info("Metal testing enabled")
end

const metal_enabled = @load_preference("metal_enabled", false)
