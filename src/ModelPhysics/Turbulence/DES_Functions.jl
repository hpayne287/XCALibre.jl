#I am yet to test the functionality of these functions but am ever hopeful this is right :)

function compute_RANS_lengthscale(rans,mesh)
    (; k, omega) = rans
    L_RANS= sqrt.(k.values) ./ (omega.values .+ eps())
    return L_RANS
end


function compute_DES_lengthscale(rans, les, mesh, C_DES=0.65)
    L_RANS = compute_RANS_lengthscale(rans, mesh)
    Δ = ScalarField(mesh)
    delta!(Δ, mesh,les.config)
    return min.(L_RANS, C_DES * Δ)  
end

function compute_turbulent_viscosity(des::DES, model, mesh)
    L_DES = compute_DES_lengthscale(des.rans, des.les, mesh)
    S = compute_strain_rate_tensor(model)

    # Compute eddy viscosity ν_t
    ν_t = L_DES^2 * norm(S)

    return ν_t
end
