using DifferentialEquations
using OrdinaryDiffEq:
    ODEProblem,
    solve,
    ImplicitEuler,
    ImplicitMidpoint,
    Midpoint,
    Euler,
    SSPRK33,
    NLNewton,
    NLAnderson
using LinearSolve: KrylovJL_GMRES
using ModelingToolkit
using BenchmarkTools
using SparsityDetection, SparseArrays

function volumetric_liquid_fraction(ϑ_l::FT, ν_eff::FT) where {FT}
    if ϑ_l < ν_eff
        θ_l = ϑ_l
    else
        θ_l = ν_eff
    end
    return θ_l
end

function matric_potential(α::FT, n::FT, m::FT, S::FT) where {FT}
    ψ_m = -((S^(-FT(1) / m) - FT(1)) * α^(-n))^(FT(1) / n)
    return ψ_m
end

function effective_saturation(porosity::FT, ϑ_l::FT, θr::FT) where {FT}
    ϑ_l_safe = max(ϑ_l, θr + eps(FT))
    S_l::FT = (ϑ_l_safe - θr) / (porosity - θr)
    return S_l
end

function pressure_head(
    α::FT,
    n::FT,
    m::FT,
    θ_r::FT,
    ϑ_l::FT,
    ν_eff::FT,
    S_s::FT,
) where {FT}
    S_l_eff = effective_saturation(ν_eff, ϑ_l, θ_r)
    if S_l_eff <= FT(1.0)
        ψ = matric_potential(α,n,m, S_l_eff)
    else
        ψ = (ϑ_l - ν_eff) / S_s
    end
    return ψ
end

function hydraulic_conductivity(
    Ksat::FT,
    m::FT,
    S::FT
) where {FT}
    if S < FT(1)
        K = sqrt(S) * (FT(1) - (FT(1) - S^(FT(1) / m))^m)^FT(2)
    else
        K = FT(1)
    end
    return K* Ksat
end

const ft = Float64;

ν = ft(0.495);
Ksat = ft(0.0443 / 3600 / 100); # m/s
S_s = ft(1e-3); #inverse meters
vg_n = ft(1.4);
vg_α = ft(2.6); # inverse meters
vg_m = ft(1) - ft(1) / vg_n;
θ_r = ft(0.124);

zmax = ft(0);
zmin = ft(-1.5);
const n = 150;
const Δz = (zmax-zmin)/n
const z = Array((zmin+Δz/ft(2.0)):Δz:(zmax-Δz/ft(2.0)))


p = [ν,vg_α, vg_n, vg_m, Ksat, S_s, θ_r];

t0 = ft(0);
tf = ft(60*60);
θ0 =  ft(0.24) .+ zeros(n)

function rhs_flux!(dY, Y, p, t)
    ν,
    vg_α,
    vg_n,
    vg_m,
    Ksat,
    S_s,
    θ_r = p
    K =  hydraulic_conductivity.(Ksat,vg_m,
                                 effective_saturation.(ν, Y, θ_r))
    ψ = pressure_head.(vg_α,vg_n,vg_m, θ_r,Y, ν, S_s) .+ z
    
    top = -0.5*(Ksat+K[end])*(0.0-ψ[end])/Δz
    @inbounds for i in 1:1:n
        ip1, im1 = i+1, i-1
        Fi_ph::ft = ip1 == n+1 ? ft.(top) : ft(-2.0)/(ft(1.0)/K[ip1]+ft(1.0)/K[i])*(ψ[ip1]-ψ[i])/Δz
        Fi_mh::ft = im1  == 0 ? ft.(-K[1]) : ft(-2.0)/(ft(1.0)/K[i]+ft(1.0)/K[im1])*(ψ[i]-ψ[im1])/Δz
        dY[i] = -ft(1.0)/Δz*(Fi_ph-Fi_mh)
    end

    
end

prob = ODEProblem(rhs_flux!, θ0, (t0, tf), p);
truth = solve(prob, dt = 1.0,SSPRK33(),save_every_step =false)
u = truth.u[end]


i = Array(1:1:n)
iu = Array(1:1:n-1)
il = Array(2:1:n)
i1 = vcat(i,iu,il)
i2 = vcat(i,il,iu)
jac_sparsity = sparse(i1,i2, 1.0)
f = ODEFunction(rhs_flux!;jac_prototype=jac_sparsity);
prob_sparse = ODEProblem(f, θ0, (t0,tf), p);

#first, time a step of each
explicit_integrator = init(prob,Euler();dt = 1.0, adaptive = false, save_every_step = false, abstol = 1e-1, reltol = 1e-1);
@btime step!(explicit_integrator)
#  23.322 μs (6 allocations: 6.70 KiB)

#atol, rtol dont affect allocations or time in meaningful way. 18x difference explicit to implicit. Converges in one Newton step?
@btime step!(init(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}); dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1));
#  458.935 μs (1583 allocations: 491.44 KiB)
@btime step!(init(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}); dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1));
#  7.526 ms (776 allocations: 1.25 MiB) - prob_sparse helps despite more allocations
@btime step!(init(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()); dt = 1.0, save_every_step = false, adaptive = false));
#   319.132 μs (219 allocations: 221.59 KiB)#much fewer allocations compared with above
# 14x difference
# documentation says this doesnt require a sparsity pattern. why was it using autodiff? why does it not error if we give it a sparsity pattern??

@btime step!(init(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLAnderson(; κ=1//100, max_iter=10, max_history=5, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)); dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1));
#   108.417 μs (91 allocations: 137.95 KiB)...5x!


#_____________________________________________



@btime solve(prob, Euler(), dt= 1.0, save_every_step = false, adaptive = false);
#   88.004 ms (21632 allocations: 23.66 MiB)
# newton does not converge unless we increase tolerances. Again, once we are in convergent region, value of abstol, reltol doesnt change result
@btime solve(prob_sparse, ImplicitEuler(autodiff=false,diff_type=Val{:central}), dt= 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1);
#    981.047 ms (349260 allocations: 696.87 MiB)
## -> need ~11x larger timesteps. consistent with a single step, close enough? 18 v 11?

@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central},linsolve=KrylovJL_GMRES()), dt= 1.0, save_every_step = false, adaptive = false);
#  595.623 ms (235249 allocations: 197.68 MiB) # only 7x larger steps needed!
@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLAnderson(; κ=1//100, max_iter=10, max_history=5, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1);
#  197.603 ms (30767 allocations: 35.46 MiB) ## 2.2x slower only!
# True picard
@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLAnderson(; κ=1//100, max_iter=10, max_history=0, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt = 1.0, save_every_step = false, adaptive = false, abstol = 1e-1, reltol = 1e-1);
#  198.821 ms (30762 allocations: 35.45 MiB) the same?
# True picard, one iter
@btime solve(prob, ImplicitEuler(autodiff=false,diff_type=Val{:central}, nlsolve = NLAnderson(; κ=1//100, max_iter=1, max_history=0, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt = 1.0, save_every_step = false, adaptive = false, abstol = 100.0, reltol = 100.0);
#  176.126 ms (28884 allocations: 33.02 MiB)

@btime solve(prob, ImplicitEuler(nlsolve = NLAnderson(; κ=1//100, max_iter=1, max_history=5, aa_start=1, droptol=nothing, fast_convergence_cutoff=1//5)), dt = 1.0, save_every_step = false, adaptive = false, abstol = 100, reltol = 100);
# a little faster   175.805 ms (28878 allocations: 33.02 MiB)

#*** = not tried yet

## Need to solve nonlinear Ax = b for x=y^{n+1}.
# Method 1. Construct Jacobian and solve via Newton's method. This requires solving a linear system Ax =b each
# Newton iteration. This is the default.
# ***Can also construct J via autodiff.
# ***We can also try other linear solvers besides the default here - tridiagoal?
## Can we give a linear solver to nlsolve?
# ***We can also precondition the linear system.
# Method 2. Newton Krylov. Each Newton step, solve with GMRES. also avoids constructing Jacobian?
# *** We can also precondition here. Autodiff seems to be an option too
# Method 3: Fixed point iteration. Anderson acceleration.
# ***Preconditioners dont make sense here, I think.
# Method 4: Can we pass in an approx Jacobian? for Newton?


# preconditioners for Newton, NewtonKrylov #http://linearsolve.sciml.ai/dev/basics/Preconditioners/
# different linear solves (diagonal matric = J) #http://linearsolve.sciml.ai/dev/solvers/solvers/

##Do we need to give "tol" to explicit? why?
