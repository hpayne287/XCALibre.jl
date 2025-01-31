export KωSmagorinsky

struct KωSmagorinsky{S1,S2,S3,F1,F2,F3,C} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
end
Adapt.@adapt_structure KωSmagorinsky

struct KωSmagorinskyModel{E1,E2,D,S}
    k_eqn::E1 
    ω_eqn::E2
    Δ::D 
    magS::S
end 
Adapt.@adapt_structure KωSmagorinskyModel


#Model API Constructor 
# DES{KωSmagorinsky}() = begin
#     coeffs = (β⁺=0.09, α1=0.52, β1=0.072, σk=0.5, σω=0.5, C=0.15)
#     ARG = typeof(coeffs)
#     DES{KωSmagorinsky,ARG}(coeffs)
# end 

DES{KωSmagorinsky}() = begin
    rans_model = RANS{KOmega}()  # Initialize RANS model
    les_model = LES{Smagorinsky}()    # Initialize LES model
    des_args = (C_DES=0.65, σk=0.5, σω=0.5) # DES coefficients
    X = typeof(rans_model)
    Y = typeof(les_model)
    ARG = typeof(des_args)
    DES{KωSmagorinsky, ARG}(rans_model, les_model, des_args)
end


# Function as constructor
(rans::DES{KωSmagorinsky,ARG})(mesh) where ARG = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = rans.args
    KωSmagorinsky(k,omega,nut,kf,omegaf,nutf,coeffs)
end

# function initialise(
#     turbulence::KωSmagorinsky, model::Physics{T,F,M,Tu,E,D,BI},mdotf,peqn,config
#     ) where {T,F,M,Tu,E,D,BI}

#     (; solvers, schemes,runtime) = config
#     mesh = model.domain
#     (; k, omega, nut) = turbulence
#     (; rho) = model.fluid
#     eqn = peqn.equation

#     # define fluxes and sources
#     mueffk = FaceScalarField(mesh)
#     mueffω = FaceScalarField(mesh)
#     Dkf = ScalarField(mesh)
#     Dωf = ScalarField(mesh)
#     Pk = ScalarField(mesh)
#     Pω = ScalarField(mesh)
    
#     k_eqn = (
#             Time{schemes.k.time}(rho, k)
#             + Divergence{schemes.k.divergence}(mdotf, k) 
#             - Laplacian{schemes.k.laplacian}(mueffk, k) 
#             + Si(Dkf,k) # Dkf = β⁺rho*omega
#             ==
#             Source(Pk)
#         ) → eqn
    
#     ω_eqn = (
#             Time{schemes.omega.time}(rho, omega)
#             + Divergence{schemes.omega.divergence}(mdotf, omega) 
#             - Laplacian{schemes.omega.laplacian}(mueffω, omega) 
#             + Si(Dωf,omega)  # Dωf = rho*β1*omega
#             ==
#             Source(Pω)
#     ) → eqn

#     # Set up preconditioners
#     @reset k_eqn.preconditioner = set_preconditioner(
#                 solvers.k.preconditioner, k_eqn, k.BCs, config)

#     # @reset ω_eqn.preconditioner = set_preconditioner(
#     #             solvers.omega.preconditioner, ω_eqn, omega.BCs, config)

#     @reset ω_eqn.preconditioner = k_eqn.preconditioner
    
#     # preallocating solvers
#     @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
#     @reset ω_eqn.solver = solvers.omega.solver(_A(ω_eqn), _b(ω_eqn))

#     magS = ScalarField(mesh)
#     Δ = ScalarField(mesh)

#     delta!(Δ,mesh,config)
#     @. Δ.values = Δ.values^2 
#     return KωSmagorinskyModel(k_eqn,ω_eqn,Δ,magS)
# end
function initialize!(turbulence::KωSmagorinsky, model::Physics{T,F,M,Tu,E,D,BI},mdotf,peqn,config) where {T,F,M,Tu,E,D,BI}
    
    # Initialize RANS model (for near-wall regions)
    initialize!(des.rans, mesh, config)

    # Initialize LES model (for free-stream)
    initialize!(des.les, mesh, config)

    # Compute initial DES length scale
    des_length_scale = compute_DES_lengthscale(des.rans, des.les, mesh)

    # Store the length scale inside the DES model (optional)
    des.args = merge(des.args, (L_DES = des_length_scale,))

    println("DES Model Initialized with RANS-LES blending")
end

function turbulence!(des::DES, model, mesh, config)
    ν_t = compute_turbulent_viscosity(des, model, mesh)
    apply_turbulence_model!(des, model, ν_t, config)
end

#Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI},VTKWriter,name
    ) where {T,F,M,Tu<:KωSmagorinsky,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("T", model.energy.T),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("omega", model.turbulence.omega),
            ("nut", model.turbulence.nut)
        )
    end
    write_vtk(name, model.domain, VTKWriter, args...)
end