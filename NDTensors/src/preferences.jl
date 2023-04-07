using Preferences

function set_cuda_backend(new_backend::String)
  if !(new_backend in ("ITensorGPU", "NDTensorCUDA"))
    throw(ArgumentError("Invalid CUDA backend: \"$(new_backend)\""))
  end

  #set CUDA tpe in runtime value as well as diagblocksparse
  @set_preferences!("cuda_backend" => new_backend)
  @info("New CUDA backend set; restart your Julia session for this change to take effect!")
end

const cuda_backend = @load_preference("cuda_backend", "NDTensorCUDA")
