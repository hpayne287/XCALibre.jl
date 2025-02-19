# Hudson can you add your case test code here so I can help!

using Plots
using XCALibre

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2024-11-24-CircleMesh.unv"

mesh = UNV2D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=4)

mesh_dev = mesh

velocity = [4, 0.0, 0.0]
nu = 1e-3
Re = velocity[1] * 0.1 / nu
k_inlet = 1
ω_inlet = 1000

model = Physics(
    time=Steady(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=DES{MenterF1}(walls=(:wall1,:wall2,:wall3)),
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)


#region Define BC
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall2, [0.0, 0.0, 0.0]),
    Wall(:wall1, [0.0, 0.0, 0.0]),
    Wall(:wall3, [0.0, 0.0, 0.0]),
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:wall1, 0.0),
    Neumann(:wall2, 0.0),
    Neumann(:wall3, 0.0),
)
@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    KWallFunction(:wall1),
    KWallFunction(:wall2),
    KWallFunction(:wall3)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall1),
    OmegaWallFunction(:wall2),
    OmegaWallFunction(:wall3)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet / ω_inlet),
    Neumann(:outlet, 0.0),
    NutWallFunction(:wall1),
    NutWallFunction(:wall2),
    NutWallFunction(:wall3)
)

#endregion

schemes = (
    U=set_schemes(divergence=Linear),
    p=set_schemes(),
    k=set_schemes(gradient=Midpoint),
    omega=set_schemes(gradient=Midpoint),
    y = set_schemes(gradient=Midpoint)
)

solvers = (
    U=set_solver(
        model.momentum.U;
        solver=BicgstabSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    p=set_solver(
        model.momentum.p;
        solver=CgSolver,
        preconditioner=Jacobi(),
        convergence=1e-7,
        relax=0.7,
        rtol=1e-4,
        atol=1e-10
    ),
    y = set_solver(
        model.turbulence.y;
        solver      = CgSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.9,
        itmax = 2000
    ),
    k = set_solver(
        model.turbulence.k;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    ),
    omega = set_solver(
        model.turbulence.omega;
        solver      = BicgstabSolver, # BicgstabSolver, GmresSolver
        preconditioner = Jacobi(),
        convergence = 1e-8,
        relax       = 0.3,
        rtol = 1e-2,
    )
)

runtime = set_runtime(iterations=100, time_step=1, write_interval=1)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)

initialise!(model.momentum.U, velocity) 
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config);