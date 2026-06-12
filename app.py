import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Reactor Simulator", layout="wide")
st.title("Educational Reactor Control Rod Simulator")

# =========================
# CORE PARAMETERS
# =========================
N = 30
dx = 1.0
D = 0.6

beta_i = np.array([0.000215,0.001424,0.001274,
                   0.002568,0.000748,0.000273])
lambda_i = np.array([0.0124,0.0305,0.111,
                     0.301,1.14,3.01])
beta = np.sum(beta_i)
Lambda = 1e-4

# =========================
# UI CONTROLS
# =========================
st.sidebar.header("Controls")

rho_base = st.sidebar.slider("Base Reactivity", -0.005, 0.005, 0.002, 0.0001)

num_rods = st.sidebar.slider("Number of Control Rods", 0, 10, 2)

rod_positions = []
for i in range(num_rods):
    rod_positions.append(
        st.sidebar.slider(f"Rod {i+1} Position", 0, N-1, N//2)
    )

rod_strength = st.sidebar.slider("Rod Strength", 0.0, 0.01, 0.003, 0.0001)

sigma = 2.0

# =========================
# INITIAL CONDITIONS
# =========================
phi0 = np.ones(N)

C0 = np.zeros((6, N))
for g in range(6):
    C0[g, :] = beta_i[g] * phi0 / (Lambda * lambda_i[g])

y0 = np.concatenate([phi0, C0.flatten()])

# =========================
# HELPER FUNCTIONS
# =========================
def unpack(y):
    phi = y[:N]
    C = y[N:].reshape((6, N))
    return phi, C

# =========================
# REACTIVITY FIELD (CONTROL RODS)
# =========================
def rho_field(i):
    rho = rho_base
    for rp in rod_positions:
        rho -= rod_strength * np.exp(-((i - rp)**2)/(2*sigma**2))
    return rho

# =========================
# REACTOR MODEL
# =========================
def reactor(t, y):
    phi, C = unpack(y)

    dphi = np.zeros(N)
    dC = np.zeros((6, N))

    for i in range(N):

        # diffusion
        lap = 0
        if i > 0:
            lap += phi[i-1]
        if i < N-1:
            lap += phi[i+1]
        lap -= 2 * phi[i]
        lap /= dx**2

        rho = rho_field(i)

        dphi[i] = D * lap + ((rho - beta)/Lambda)*phi[i] + np.sum(lambda_i * C[:, i])

        dC[:, i] = beta_i * phi[i] / Lambda - lambda_i * C[:, i]

    return np.concatenate([dphi, dC.flatten()])

# =========================
# SOLVER
# =========================
t_eval = np.linspace(0, 5, 300)

sol = solve_ivp(
    reactor,
    (0, 5),
    y0,
    t_eval=t_eval,
    method="Radau"
)
if not sol.success:
    st.error("Solver failed to converge.")
    st.stop()

phi_time = sol.y[:N, :]
phi = phi_time

# =========================
# VISUALIZATION SELECTOR
# =========================

view = st.selectbox(
    "Visualization Mode",
    [
        "Flux Plot",
        "Heat Map",
        "Animated Core",
        "Temperature"
    ]
)

# =========================
# STANDARD PLOTS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Flux Shape (Final State)")
    fig, ax = plt.subplots()
    ax.plot(phi[:, -1])
    ax.set_xlabel("Core Position")
    ax.set_ylabel("Flux")
    ax.grid()
    st.pyplot(fig)

with col2:
    st.subheader("Center Power vs Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(sol.t, phi[N//2])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Flux")
    ax2.grid()
    st.pyplot(fig2)

# =========================
# HEAT MAP
# =========================

if view == "Flux Plot":
    st.info("Showing standard flux and power plots above.")

elif view == "Heat Map":

    st.subheader("Neutron Flux Heat Map")

    heatmap = np.tile(phi[:, -1], (10, 1))

    fig, ax = plt.subplots()

    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(im, ax=ax)

    st.pyplot(fig)

# =========================
# ANIMATED CORE
# =========================

elif view == "Animated Core":

    st.subheader("Animated Reactor Core")

    time_idx = st.slider(
        "Simulation Time",
        0,
        len(sol.t)-1,
        len(sol.t)-1
    )

    current_flux = phi[:, time_idx]

    current_flux = np.nan_to_num(
        current_flux,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    core = current_flux.reshape(5, 6)

    fig, ax = plt.subplots()

    im = ax.imshow(
        core,
        aspect="equal",
        origin="lower"
    )

    plt.colorbar(
        im,
        ax=ax,
        label="Neutron Flux"
    )

    for rp in rod_positions:

        row = rp // 6
        col = rp % 6

        ax.scatter(
            col,
            row,
            marker="X",
            s=200
        )

    ax.set_title(
        f"Core State at t={sol.t[time_idx]:.2f} s"
    )

    st.pyplot(fig)

# =========================
# TEMPERATURE VIEW
# =========================

elif view == "Temperature":

    st.subheader("Core Temperature Map")

    time_idx = st.slider(
        "Simulation Time",
        0,
        len(sol.t)-1,
        len(sol.t)-1,
        key="temp_slider"
    )

    current_flux = np.maximum(
        phi[:, time_idx],
        0
    )

    current_flux = np.nan_to_num(
        current_flux,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    core = current_flux.reshape(5, 6)

    max_flux = np.max(core)

    if max_flux > 0:
        temperature = 300 + 500 * (
            core / max_flux
        )
    else:
        temperature = np.full_like(
            core,
            300
        )

    fig, ax = plt.subplots()

    im = ax.imshow(
        temperature,
        aspect="equal",
        origin="lower"
    )

    plt.colorbar(
        im,
        ax=ax,
        label="Temperature (K)"
    )

    for rp in rod_positions:

        row = rp // 6
        col = rp % 6

        ax.scatter(
            col,
            row,
            marker="X",
            s=200
        )

    ax.set_title(
        f"Estimated Core Temperature at t={sol.t[time_idx]:.2f} s"
    )

    st.pyplot(fig)

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Peak Temperature",
            f"{temperature.max():.1f} K"
        )

    with colB:
        st.metric(
            "Average Temperature",
            f"{temperature.mean():.1f} K"
        )

   
# =========================
# EXTRA INFO PANEL
# =========================
st.markdown("### System Info")
st.write(f"Rods inserted: {num_rods}")
st.write(f"Base reactivity: {rho_base}")
st.write("Model: 1D diffusion + 6-group delayed neutrons")
st.write("Model: 1D diffusion + 6-group delayed neutrons")
