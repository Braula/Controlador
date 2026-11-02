"""Malha fechada: referência -> PD -> OpenSim -> medição -> PD ...

Cenário: o humano entrega só uma fração do torque normal (FRACAO_HUMANO).
O exo entrega o torque que falta (desejado - humano) e o PD corrige o
que sobrar. KP = KD = 0 deixa só a parte do torque que falta.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from referencia import carregar_referencia, interpolador
from controlador import controle_exo
from planta import montar_planta, ler_junta, aplicar_torque, avancar

# ---------------- CONFIGURAÇÃO ----------------
PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
ARQ_MODELO = os.path.join(PASTA, "multi_com_exo.osim")       # Gait2392 + exo (URDF)
ARQ_IK = os.path.join(PASTA, "IK_filtrado.mot")              # gerado pelo preprocessar.py
ARQ_ID = os.path.join(PASTA, "ID_filtrado.sto")              # ID do modelo SEM exo
ARQ_GRF = os.path.join(PASTA, "subject01_walk1_grf.xml")     # .mot na mesma pasta; None = sem chão

JUNTA = "knee_angle_l"
FRACAO_HUMANO = 0.1     # 1.0 = pessoa saudável; 0.0 = exo faz tudo
# Ganhos "chutados" pela dinâmica da perna: Kp = J*wn^2, Kd = 2*zeta*J*wn
# J = 0,8 kg.m^2 (canela + pé + exo em torno do joelho), wn = 2 Hz, zeta = 1
KP = 126.0              # N.m/rad
KD = 20.0               # N.m.s/rad
TAU_MAX = 120.0         # limite do motor, N.m
DT = 0.001              # período do controlador, s
T_FINAL = None          # None = até o fim da referência
# ----------------------------------------------


def simular(Kp, Kd, fracao, t_final=None):
    ref = carregar_referencia(ARQ_IK, ARQ_ID, JUNTA)
    th_d, dth_d = interpolador(ref, "theta_d"), interpolador(ref, "dtheta_d")
    tau_h = interpolador(ref, "tau_humano")

    modelo, estado, at_h, at_exo = montar_planta(ARQ_MODELO, ref, ARQ_GRF)
    t_final = t_final or float(ref["t"][-1])

    chaves = ("t", "theta_d", "theta", "tau_d", "tau_h", "tau_exo", "erro_tau")
    log = {k: [] for k in chaves}
    while estado.getTime() < t_final - 1e-9:
        t = estado.getTime()
        modelo.realizeVelocity(estado)
        theta, dtheta = ler_junta(modelo, estado, JUNTA)

        td = tau_h(t)                       # torque desejado: marcha normal (ID)
        th = fracao * td
        te = controle_exo(td, th, (th_d(t), dth_d(t)), (theta, dtheta), Kp, Kd, TAU_MAX)
        aplicar_torque(estado, at_h, at_exo, th, te)
        erro_tau = td - (th + te)           # desejado - (humano + robô)

        for k, v in zip(chaves, (t, th_d(t), theta, td, th, te, erro_tau)):
            log[k].append(v)
        estado = avancar(modelo, estado, DT)

    return {k: np.array(v) for k, v in log.items()}


def metricas(r):
    erro = np.rad2deg(r["theta_d"] - r["theta"])
    tot = np.abs(r["tau_h"]) + np.abs(r["tau_exo"])
    return {
        "RMSE (graus)": np.sqrt(np.mean(erro ** 2)),
        "erro máximo (graus)": np.max(np.abs(erro)),
        "pico torque exo (N.m)": np.max(np.abs(r["tau_exo"])),
        "parcela do exo (%)": 100 * np.sum(np.abs(r["tau_exo"])) / np.sum(tot),
        "RMSE torque (N.m)": np.sqrt(np.mean(r["erro_tau"] ** 2)),
        "erro máx. torque (N.m)": np.max(np.abs(r["erro_tau"])),
    }


def plotar(r, titulo):
    cores = {"ref": "#1f4e79", "sim": "#d95f02", "hum": "#1b9e77",
             "exo": "#7570b3", "tot": "#e7298a", "err": "#a6761d"}
    fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True, facecolor="white")
    ax[0].plot(r["t"], np.rad2deg(r["theta_d"]), color=cores["ref"], lw=2, label="referência")
    ax[0].plot(r["t"], np.rad2deg(r["theta"]), "--", color=cores["sim"], lw=2, label="simulado")
    ax[0].set_ylabel("ângulo (graus)")
    ax[1].plot(r["t"], np.rad2deg(r["theta_d"] - r["theta"]), color=cores["sim"])
    ax[1].set_ylabel("erro ângulo (graus)")
    ax[2].plot(r["t"], r["tau_d"], color=cores["ref"], lw=2, label="desejado (ID)")
    ax[2].plot(r["t"], r["tau_h"] + r["tau_exo"], "--", color=cores["tot"], lw=2, label="humano + exo")
    ax[2].plot(r["t"], r["tau_h"], color=cores["hum"], label="humano")
    ax[2].plot(r["t"], r["tau_exo"], color=cores["exo"], label="exo")
    ax[2].set_ylabel("torque (N.m)")
    ax[3].plot(r["t"], r["erro_tau"], color=cores["err"])
    ax[3].set_ylabel("erro torque (N.m)")
    ax[3].set_xlabel("tempo (s)")
    for a in ax:
        a.set_facecolor("white")
        a.grid(alpha=0.3)
    ax[0].legend(); ax[2].legend(ncol=2, fontsize=8)
    ax[0].set_title(titulo)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    r = simular(KP, KD, FRACAO_HUMANO, T_FINAL)
    for nome, valor in metricas(r).items():
        print(f"{nome:24s} {valor:8.3f}")
    plotar(r, f"PD Kp={KP:g} Kd={KD:g} | humano {FRACAO_HUMANO:.0%}")
    nome = f"resultado_Kp{KP:g}_Kd{KD:g}_humano{FRACAO_HUMANO*100:.0f}.png"
    plt.savefig(nome, dpi=150)
    print("gráfico salvo em:", nome)
    plt.show()