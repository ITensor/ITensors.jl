# ITensorVisualizationBase

This is a package for visualizing tensor networks. The main purpose is to use it with the [ITensors.jl](https://github.com/ITensor/ITensors.jl) package to view and debug tensor network contractions, for example:
```julia
using ITensors
using ITensorVisualizationBase

i = Index(2, "i")
j = Index(10, "j")
k = Index(40, "k")
l = Index(40, "l")
m = Index(40, "m")
A = randomITensor(i, j, k)
B = randomITensor(i, j, l, m)
C = randomITensor(k, l)
ABC = @visualize A * B * C
```
This will execute the contraction and output 
![alt text](assets/ITensorVisualization_A_B_C_unicode_notags.png)

You can show the visualization with tags with:
```julia
ABC = @visualize A * B * C edge_labels=(tags=true,)
```
![alt text](assets/ITensorVisualization_A_B_C_unicode_tags.png)

In order to output a more sophisticated interactive visualization,
load a Makie backend and specify you want Makie as your backend:
```julia
using GLMakie

ITensorVisualizationBase.set_backend!("Makie")

ABC = @visualize A * B * C edge_labels=(tags=true,);
```
A window like the following should appear:
![alt text](assets/ITensorVisualization_A_B_C.png)

The visualization makes an initial guess for the locations of the tensors (using [NetworkLayout.jl](https://github.com/JuliaGraphs/NetworkLayout.jl)), and then allows users to interactively move the tensors to better locations. You can move the tensors and external indices (the square and circle nodes of the network) by left clicking on a node and dragging it to a new location.  You can also right click and drag to translate the entire diagram, and scroll to zoom in and out.

In addition, you can visualize multiple steps of a contraction as follows:
```julia
julia> AB = @visualize fig A * B edge_labels=(tags=true,);

julia> ABC = @visualize! fig[1, 2] AB * C edge_labels=(tags=true,);

julia> fig
```
![alt text](assets/ITensorVisualization_A_B_C_sequence.png)
