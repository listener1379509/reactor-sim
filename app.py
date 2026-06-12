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

phi = sol.y[:N]

# =========================
# VISUALIZATION
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Flux Shape (Final State)")
    fig, ax = plt.subplots()
    ax.plot(phi[:, -1], color="orange")
    ax.set_xlabel("Core Position")
    ax.set_ylabel("Flux")
    ax.grid()
    st.pyplot(fig)

with col2:
    st.subheader("Center Power vs Time")
    fig2, ax2 = plt.subplots()
    ax2.plot(sol.t, phi[N//2], color="blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Flux")
    ax2.grid()
    st.pyplot(fig2)

# =========================
# EXTRA INFO PANEL
# =========================
st.markdown("### System Info")
st.write(f"Rods inserted: {num_rods}")
st.write(f"Base reactivity: {rho_base}")
st.write("Model: 1D diffusion + 6-group delayed neutrons")
