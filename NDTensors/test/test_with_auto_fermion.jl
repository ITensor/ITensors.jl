@eval module $(gensym())
using NDTensors: NDTensors
using Test: @test, @test_throws, @testset

@testset "NDTensors.with_auto_fermion" begin
    @testset "starting state $start" for start in (false, true)
        start ? NDTensors.enable_auto_fermion() : NDTensors.disable_auto_fermion()
        @test NDTensors.using_auto_fermion() == start

        @testset "default enables auto-fermion" begin
            inside = Ref(false)
            NDTensors.with_auto_fermion() do
                return inside[] = NDTensors.using_auto_fermion()
            end
            @test inside[] == true
            @test NDTensors.using_auto_fermion() == start
        end

        @testset "explicit enable=false disables auto-fermion" begin
            inside = Ref(true)
            NDTensors.with_auto_fermion(false) do
                return inside[] = NDTensors.using_auto_fermion()
            end
            @test inside[] == false
            @test NDTensors.using_auto_fermion() == start
        end

        @testset "return value is propagated" begin
            @test NDTensors.with_auto_fermion(() -> 42) == 42
            @test NDTensors.with_auto_fermion(() -> 42, false) == 42
            @test NDTensors.using_auto_fermion() == start
        end

        @testset "restores previous state after exception" begin
            @test_throws ErrorException NDTensors.with_auto_fermion() do
                return error("boom")
            end
            @test NDTensors.using_auto_fermion() == start
            @test_throws ErrorException NDTensors.with_auto_fermion(false) do
                return error("boom")
            end
            @test NDTensors.using_auto_fermion() == start
        end

        @testset "nested scopes" begin
            inner_state = Ref{Bool}(start)
            NDTensors.with_auto_fermion(true) do
                @test NDTensors.using_auto_fermion() == true
                NDTensors.with_auto_fermion(false) do
                    return inner_state[] = NDTensors.using_auto_fermion()
                end
                @test NDTensors.using_auto_fermion() == true
            end
            @test inner_state[] == false
            @test NDTensors.using_auto_fermion() == start
        end

        @testset "nested scope restores after inner exception" begin
            NDTensors.with_auto_fermion(true) do
                @test NDTensors.using_auto_fermion() == true
                @test_throws ErrorException NDTensors.with_auto_fermion(false) do
                    return error("boom")
                end
                @test NDTensors.using_auto_fermion() == true
            end
            @test NDTensors.using_auto_fermion() == start
        end
    end

    # Leave the auto-fermion state as we found it for the rest of the suite.
    NDTensors.disable_auto_fermion()
end
end
