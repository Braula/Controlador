"""Dados de referência: marcha humana normal, sem exoesqueleto (IK + ID)."""
import numpy as np
 
 
def ler_sto(caminho):
    """Lê .sto/.mot do OpenSim. Retorna (dict coluna -> array, em_graus)."""
    with open(caminho, "r") as f:
        linhas = f.read().splitlines()
    em_graus = False
    i = 0
    while "endheader" not in linhas[i].lower():
        if linhas[i].replace(" ", "").lower() == "indegrees=yes":
            em_graus = True
        i += 1
    nomes = linhas[i + 1].split()
    dados = np.array([l.split() for l in linhas[i + 2:] if l.strip()], dtype=float)
    return {n: dados[:, k] for k, n in enumerate(nomes)}, em_graus
 
 
def carregar_referencia(arquivo_ik, arquivo_id, junta="knee_angle_l"):
    """Retorna dict com:
    t, theta_d, dtheta_d  -> trajetória desejada da junta (rad, rad/s)
    tau_humano            -> torque do ID na junta, nos tempos do IK (N.m)
    coords                -> todas as coordenadas do IK em SI (rad ou m),
                             usadas para prescrever o resto do corpo
    """
    ik, em_graus = ler_sto(arquivo_ik)
    idd, _ = ler_sto(arquivo_id)
 
    # só o trecho em que existem os dois arquivos (fora dele o torque não existe)
    dentro = (ik["time"] >= idd["time"][0]) & (ik["time"] <= idd["time"][-1])
    t = ik["time"][dentro]
 
    coords = {}
    for nome, valores in ik.items():
        if nome == "time":
            continue
        valores = valores[dentro]
        # translações da pelve ficam em metros; o resto é ângulo
        eh_translacao = nome.endswith(("_tx", "_ty", "_tz"))
        coords[nome] = valores if (eh_translacao or not em_graus) else np.deg2rad(valores)
 
    theta_d = coords[junta]
    tau = np.interp(t, idd["time"], idd[junta + "_moment"])
 
    return {
        "t": t,
        "theta_d": theta_d,
        "dtheta_d": np.gradient(theta_d, t),
        "tau_humano": tau,
        "coords": coords,
        "junta": junta,
    }
 
 
def interpolador(ref, chave):
    """Função f(tempo) para consultar a referência em qualquer instante."""
    return lambda tempo: float(np.interp(tempo, ref["t"], ref[chave]))
 
