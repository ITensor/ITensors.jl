module SmallVectors
  using StaticArrays

  include("StaticArraysExt.jl")
  using .StaticArraysExt

  export SmallVector, MSmallVector, SubSmallVector

  struct NotImplemented <: Exception
    msg::String
  end
  NotImplemented() = NotImplemented("Not implemented.")

  struct BufferDimensionMismatch <: Exception
    msg::String
  end

  include("abstractsmallvector/abstractsmallvector.jl")
  include("abstractsmallvector/deque.jl")
  include("msmallvector/msmallvector.jl")
  include("smallvector/smallvector.jl")
  include("subsmallvector/subsmallvector.jl")
end
