export wall_distance!

"""
Poisson Equation:
y = ± Σⱼ₌₁,₃ ∂ϕ∂xⱼ² + Σⱼ₌₁,₃ ∂ϕ∂xⱼ² + 2ϕ
 - PG Tucker 2003
"""
function wall_distance!(model, config)
    @info "Calculating wall distance..."

    mesh = model.domain

    y = model.turbulence.y
    (; solvers, schemes, runtime, hardware) = config
    iterations = solvers.y.itmax
    
    phi = ScalarField(mesh)
    phif = FaceScalarField(mesh)
    initialise!(phif, 1.0)
    initialise!(phi, 0.0)

    phi_eqn = (
        - Laplacian{schemes.y.laplacian}(phif, phi) 
        == Source(ConstantScalar(1.0))
    ) → ScalarEquation(phi)

    @reset phi_eqn.preconditioner = set_preconditioner(
        solvers.y.preconditioner, phi_eqn, phi.BCs, config)

    @reset phi_eqn.solver = solvers.y.solver(_A(phi_eqn), _b(phi_eqn))

    TF = _get_float(mesh)

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    grad!(phiGrad, phif, phi, phi.BCs, zero(TF), config) # assuming time=0

    for iteration ∈ 1:iterations
        discretise!(phi_eqn, phi, config)
        apply_boundary_conditions!(phi_eqn, phi.BCs, nothing, 0.0, config)

        update_preconditioner!(phi_eqn.preconditioner, mesh, config)
        implicit_relaxation!(phi_eqn, phi.values, solvers.y.relax, nothing, config)
        phi_res = solve_system!(phi_eqn, solvers.y, phi, nothing, config)

        if phi_res < solvers.y.convergence 
            @info "Wall distance converged in $iteration iterations ($phi_res)"
            break
        elseif iteration == iterations
            @info "Wall distance calculation did not converged ($phi_res)"
        end
    end
    
    grad!(phiGrad, phif, phi, phi.BCs, zero(TF), config) # assuming time=0
    normal_distance!(y, phi, phiGrad, config)

end

function normal_distance!(y, phi, phiGrad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    kernel! = _normal_distance!(backend, workgroup)
    kernel!(y, phi, phiGrad, ndrange = length(phi.values))
    # KernelAbstractions.synchronize(backend)
end

@kernel function _normal_distance!(y, phi, phiGrad)
    i = @index(Global)

    # y = ± Σⱼ₌₁,₃ (∂ϕ/∂xⱼ)² + Σⱼ₌₁,₃ (∂ϕ/∂xⱼ)² + 2ϕ
    tot = (phiGrad.result.x[i]^2 + phiGrad.result.y[i]^2 + phiGrad.result.z[i]^2)
    ymax = (-tot + (tot + 2*phi.values[i]))
    ymin = (+tot + (tot + 2*phi.values[i]))
    y.values[i] = ymax + ymin
end