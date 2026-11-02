
"""Controlador do exoesqueleto: antecipação (torque que falta) + PD."""


def controle_pd(ref, saida, Kp, Kd):
    """tau = Kp*(theta_d - theta) + Kd*(dtheta_d - dtheta)

    ref   = (theta_d, dtheta_d)  -> referência no instante atual
    saida = (theta, dtheta)      -> medido na planta no instante atual
    """
    theta_d, dtheta_d = ref
    theta, dtheta = saida
    return Kp * (theta_d - theta) + Kd * (dtheta_d - dtheta)


def torque_falta(tau_desejado, tau_humano):
    """Torque que o humano não entrega: tau_desejado - tau_humano."""
    return tau_desejado - tau_humano


def controle_exo(tau_desejado, tau_humano, ref, saida, Kp, Kd, tau_max=None):
    """tau_exo = (tau_desejado - tau_humano) + PD, limitado a +-tau_max."""
    tau = torque_falta(tau_desejado, tau_humano) + controle_pd(ref, saida, Kp, Kd)
    if tau_max is not None:
        tau = max(-tau_max, min(tau_max, tau))
    return tau
 