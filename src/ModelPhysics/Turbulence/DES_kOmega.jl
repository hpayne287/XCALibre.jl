export KωSmagorinsky

struct KωSmagorinsky{S1,S2,S3,F1,F2,F3,C,M1,M2} <: AbstractDESModel
    k::S1
    omega::S2
    nut::S3
    kf::F1
    omegaf::F2
    nutf::F3
    coeffs::C
    rans::M1
    les::M2
end
Adapt.@adapt_structure KωSmagorinsky

#Model Constructor using a RANS and LES model
# Can be rewritten for K-ϵ model or another LES turbulence type
DES{KωSmagorinsky}(nu, mesh_dev, v) = begin
    # Construct RANS model
    rans_model = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=RANS{KOmega}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )
    # Construct LES model
    les_model = Physics(
        time=Transient(),
        fluid=Fluid{Incompressible}(nu=nu),
        turbulence=LES{Smagorinsky}(),
        energy=Energy{Isothermal}(),
        domain=mesh_dev
    )
    des_args = (C_DES=0.65, σk=0.5, σω=0.5, v=v) # DES coefficients
    ARG = typeof(des_args)
    DES{KωSmagorinsky,ARG}(rans_model, les_model, des_args)
end


# Functor as constructor
(des::DES{KωSmagorinsky,ARG})(mesh) where {ARG} = begin
    k = ScalarField(mesh)
    omega = ScalarField(mesh)
    nut = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    omegaf = FaceScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = des.args
    rans = des.rans
    les = des.les
    KωSmagorinsky(k, omega, nut, kf, omegaf, nutf, coeffs, rans, les)
end

function initialise(T::KωSmagorinsky, model::Physics, mdotf::FaceScalarField, peqn::ModelEquation, config)

    #This feels like the wrong way of storing access to the LES and RANS models but works for now
    rans = T.rans_model
    les = T.les_model
    velocity = T.args.v
    # Initialise RANS model 
    initialise!(rans.momentum.U, velocity) #Need to review what actually gets passed to the RANS and LES initialise methods


    # Initialise LES model 
    initialise(les.turbulence, les, mdotf, peqn, config)

    # Compute initial DES length scale
    des_length_scale = compute_DES_lengthscale(rans, les, mesh)

    # Store the length scale inside the DES model 
    des.args = merge(des.args, (L_DES=des_length_scale,))
end

function turbulence!(des::DES, model, mesh, config)
    ν_t = compute_turbulent_viscosity(des, model, mesh)
    apply_turbulence_model!(des, model, ν_t, config)
end

#Specialise VTK writer
function model2vtk(model::Physics{T,F,M,Tu,E,D,BI}, VTKWriter, name
) where {T,F,M,Tu<:KωSmagorinsky,E,D,BI}
    if typeof(model.fluid) <: AbstractCompressible
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