
# Warn only once, using the message `msg`.
# `funcsym` is a symbol that determines if the warning has been
# called before (so there is only one warning per `funcsym`).
function warn_once(msg, funcsym; force=true, stacktrace=true)
  if stacktrace
    io = IOBuffer()
    Base.show_backtrace(io, backtrace())
    backtrace_string = String(take!(io))
    backtrace_string *= "\n"
    msg *= backtrace_string
  end
  Base.depwarn(msg, funcsym; force)
  return nothing
end

# Directory helper functions (useful for
# running examples)
src_dir() = dirname(pathof(@__MODULE__))
pkg_dir() = joinpath(src_dir(), "..")
examples_dir() = joinpath(pkg_dir(), "examples")

# Determine version and uuid of the package
function _parse_project_toml(field::String)
  return Pkg.TOML.parsefile(joinpath(pkg_dir(), "Project.toml"))[field]
end
version() = VersionNumber(_parse_project_toml("version"))
uuid() = Base.UUID(_parse_project_toml("uuid"))
