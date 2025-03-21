# using Plots
using XCALibre
using Alert


# grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
# grid1 = "flatplate_2D_lowRe.unv"
# # grid1 = "bfs_unv_tet_10mm.unv"
# mesh_file = joinpath(grids_dir, grid1)

mesh_file = "C:/Users/Hudson/OneDrive - The University of Nottingham/Year 3/Individual Project/Code/Meshes/2025-03-21-JetFlow3D.unv"

mesh = UNV3D_mesh(mesh_file, scale=0.001)

# Select backend and setup hardware
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=32)

mesh_dev = mesh

velocity = [0.0, 0.0, -0.5]
noSlip = [0.0,0.0,0.0]
nu = 1e-3
Re = 23000
k_inlet = 1
ω_inlet = 1000

model = Physics(
    time=Steady(),
    fluid=Fluid{Incompressible}(nu=nu),
    turbulence=DES{Hybrid}(walls=(:wall,)), #walls=(:wall,) blendType=MenterF1()
    energy=Energy{Isothermal}(),
    domain=mesh_dev
)


#region Define BC
@assign! model momentum U (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    # Neumann(:top,0.0),
    # Neumann(:side,0.0),
    # Neumann(:bottom,0.0)
)

@assign! model momentum p (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Wall(:wall, 0.0),
    # Neumann(:top,0.0),
    # Neumann(:side,0.0),
    # Neumann(:bottom,0.0)
)
@assign! model turbulence k (
    Dirichlet(:inlet, k_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall,0.0),
    # Neumann(:top,0.0),
    # Neumann(:side,0.0),
    # Neumann(:bottom,0.0)
)

@assign! model turbulence omega (
    Dirichlet(:inlet, ω_inlet),
    Neumann(:outlet, 0.0),
    OmegaWallFunction(:wall),
    # Neumann(:top,0.0),
    # Neumann(:side,0.0),
    # Neumann(:bottom,0.0)
)

@assign! model turbulence nut (
    Dirichlet(:inlet, k_inlet / ω_inlet),
    Neumann(:outlet, 0.0),
    Dirichlet(:wall,0.0),
    # Neumann(:top,0.0),
    # Neumann(:side,0.0),
    # Neumann(:bottom,0.0)
)

#endregion

schemes = (
    U=set_schemes(divergence=Linear),
    p=set_schemes(),
    y = set_schemes(gradient=Midpoint),
    k=set_schemes(gradient=Midpoint),
    omega=set_schemes(gradient=Midpoint)
    
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

runtime = set_runtime(iterations=1000, time_step=1, write_interval=10)

config = Configuration(
    solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware)


zeroV = [0.0,0.0,0.0]
initialise!(model.momentum.U, zeroV) 
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.k, k_inlet)
initialise!(model.turbulence.omega, ω_inlet)
initialise!(model.turbulence.nut, k_inlet/ω_inlet)

residuals = run!(model, config);

alert("Simulation Done!!");