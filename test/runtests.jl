using Test
using MaxwellFDTD

function runtests()
    exename = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir = pwd()

    printstyled("Testing MaxwellFDTD.jl\n"; bold=true, color=:white)
    try
        run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test1D.jl"))`)
        run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test2D.jl"))`)
        #run(`$exename -O3 --startup-file=no $(joinpath(testdir, "test3D.jl"))`)
    catch e
        printstyled("Error in tests: $(e)\n"; bold=true, color=:red)
        stacktrace()
        printstyled("Test failed\n"; bold=true, color=:red)
        return 1
    end
    return 0
end

exit(runtests())