
"""
    ITensors.compile(; path = "~/.julia/sysimages",
                       filename = "sys_itensors.so")

Compile ITensors.jl with [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).

This will take some time, perhaps a few minutes. This will create the file `sys_itensors.so`, by default in the directory `~/.julia/sysimages/`. This is a version of Julia that is compiled with ITensors. After running this command, if you start Julia with:
```
~ julia --sysimage ~/.julia/sysimages/sys_itensors.so
```
and you should see the startup time and JIT compilation time are substantially improved when you are using ITensors.

You can create an alias like:
```
~ alias julia_itensors="julia --sysimage ~/.julia/sysimages/sys_itensors.so"
```
(which you can put in your `~/.bashrc`, `~/.zshrc`, etc.). Then you can start Julia with a version of ITensors installed with the command:
```
~ julia_itensors
```

Note that if you update ITensors to a new version, for example with `Pkg.update("ITensors")`, you will need to run this command again to recompile the new version of ITensors.
"""
function compile(; path::AbstractString = joinpath(ENV["HOME"],
                                                   ".julia",
                                                   "sysimages"),
                   filename::AbstractString = "sys_itensors.so")
  if !isdir(path)
    println("""The directory "$path" doesn't exist yet, creating it now.""")
    println()
    mkdir(path)
  end
  path_filename = joinpath(path, filename)
  println("""Creating the system image containing the compiled version of ITensors at "$path_filename". This may take a few minutes.""")
  create_sysimage(:ITensors;
                  sysimage_path = path_filename,
                  precompile_execution_file = "precompile_itensors.jl")
  println("""

          The system image containing the compiled version of ITensors is located at "$path_filename". This is a version of Julia that is compiled with ITensors.

          You can now start Julia with:
          ```
          ~ julia --sysimage $path_filename
          ```
          and you should see that the startup times and JIT compilation times are substantially improved when you are using ITensors.

          You can create an alias like:
          ```
          ~ alias julia_itensors="julia --sysimage $path_filename"
          ```
          (which you can put in your `~/.bashrc`, `~/.zshrc`, etc.). Then you can start Julia with a version of ITensors installed with the command:
          ```
          ~ julia_itensors
          ```
          """)
end
