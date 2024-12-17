using SafeTestsets: @safetestset
using Suppressor: Suppressor

# check for filtered groups
# either via `--group=ALL` or through ENV["GROUP"]
const pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = uppercase(
  if isnothing(arg_id)
    get(ENV, "GROUP", "ALL")
  else
    only(match(pat, ARGS[arg_id]).captures)
  end,
)

"match files of the form `test_*.jl`, but exclude `*setup*.jl`"
istestfile(fn) =
  endswith(fn, ".jl") && startswith(basename(fn), "test_") && !contains(fn, "setup")
"match files of the form `*.jl`, but exclude `*_notest.jl` and `*setup*.jl`"
isexamplefile(fn) =
  endswith(fn, ".jl") && !endswith(fn, "_notest.jl") && !contains(fn, "setup")

@time begin
  # tests in groups based on folder structure
  for testgroup in filter(isdir, readdir(@__DIR__))
    if GROUP == "ALL" || GROUP == uppercase(testgroup)
      for file in filter(istestfile, readdir(joinpath(@__DIR__, testgroup); join=true))
        @eval @safetestset $file begin
          include($file)
        end
      end
    end
  end

  # single files in top folder
  for file in filter(istestfile, readdir(@__DIR__))
    (file == basename(@__FILE__)) && continue # exclude this file to avoid infinite recursion
    @eval @safetestset $file begin
      include($file)
    end
  end

  # test examples
  examplepath = joinpath(@__DIR__, "..", "examples")
  for (root, _, files) in walkdir(examplepath)
    contains(chopprefix(root, @__DIR__), "setup") && continue
    for file in filter(isexamplefile, files)
      filename = joinpath(root, file)
      @eval begin
        @safetestset $file begin
          $(Expr(
            :macrocall,
            GlobalRef(Suppressor, Symbol("@suppress")),
            LineNumberNode(@__LINE__, @__FILE__),
            :(include($filename)),
          ))
        end
      end
    end
  end
end
