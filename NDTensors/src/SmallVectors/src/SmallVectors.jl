module SmallVectors
  using StaticArrays

  export SmallVector, MSmallVector, SubSmallVector

  struct NotImplemented <: Exception
    msg::String
  end
  NotImplemented() = NotImplemented("Not implemented.")

  include("abstractsmallvector/abstractsmallvector.jl")
  include("abstractsmallvector/deque.jl")
  include("msmallvector/msmallvector.jl")
  include("smallvector/smallvector.jl")
  include("subsmallvector/subsmallvector.jl")
end
