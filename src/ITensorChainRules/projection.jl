function ChainRulesCore.ProjectTo(x::ITensor)
  return ProjectTo{ITensor}(; element=ProjectTo(zero(eltype(x))))
end

function (project::ProjectTo{ITensor})(dx::ITensor)
  S = eltype(dx)
  T = ChainRulesCore.project_type(project.element)
  dy = S <: T ? dx : map(project.element, dx)
  return dy
end
