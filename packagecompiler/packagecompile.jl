try
  @eval using PackageCompiler
catch e
  using Pkg
  Pkg.add("PackageCompiler")
  @eval using PackageCompiler
end
sysim_dir = "$(ENV["HOME"])/.julia/sysimages"
!isdir(sysim_dir) && mkdir(sysim_dir)
using PackageCompiler
create_sysimage(:ITensors,
                sysimage_path="$sysim_dir/sys_itensors.so",
                precompile_execution_file="precompile_itensors.jl")
