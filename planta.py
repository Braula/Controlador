"""Planta no OpenSim: modelo Gait2392 (+ exo do URDF, se já acoplado).

Todas as coordenadas seguem o IK (prescritas); só a junta controlada fica
livre e responde aos torques. Isso é o "fêmur prescrito": a coxa se move
como na marcha real e a perna (canela + pé + peça do exo) é integrada.
"""
import os
import opensim as osim

osim.Logger.setLevelString("warn")   # esconde o relatório do Manager a cada passo


def _remover_musculos(modelo):
    forcas = modelo.updForceSet()
    for i in reversed(range(forcas.getSize())):
        if osim.Actuator.safeDownCast(forcas.get(i)) is not None:
            forcas.remove(i)


def _prescrever(coord, tempos, valores):
    f = osim.SimmSpline()
    for t, v in zip(tempos, valores):
        f.addPoint(float(t), float(v))
    coord.setPrescribedFunction(f)
    coord.setDefaultIsPrescribed(True)


def _atuador(modelo, junta, nome):
    a = osim.CoordinateActuator(junta)
    a.setName(nome)
    a.setOptimalForce(1.0)
    a.setMinControl(-1e6)
    a.setMaxControl(1e6)
    modelo.addForce(a)
    return a


def montar_planta(arquivo_modelo, ref, arquivo_grf_xml=None):
    """Monta o modelo e retorna (modelo, estado, atuador_humano, atuador_exo).

    arquivo_grf_xml: forças de reação do solo. Sem elas a fase de apoio
    fica errada (o torque do ID foi calculado com o pé no chão).
    """
    junta = ref["junta"]
    modelo = osim.Model(arquivo_modelo)
    _remover_musculos(modelo)

    for i in range(modelo.getCoordinateSet().getSize()):
        c = modelo.updCoordinateSet().get(i)
        c.setDefaultLocked(False)
        c.setDefaultClamped(False)          # sem limite de amplitude
        nome = c.getName()
        if nome != junta and nome in ref["coords"]:
            _prescrever(c, ref["t"], ref["coords"][nome])

    if arquivo_grf_xml:
        cargas = osim.ExternalLoads(arquivo_grf_xml, True)
        # o .xml cita o .mot pelo nome; o caminho é resolvido pela pasta do .xml
        pasta = os.path.dirname(os.path.abspath(arquivo_grf_xml))
        cargas.setDataFileName(os.path.join(pasta, cargas.getDataFileName().strip()))
        modelo.addModelComponent(cargas)

    humano = _atuador(modelo, junta, "torque_humano")
    exo = _atuador(modelo, junta, "torque_exo")

    estado = modelo.initSystem()
    humano.overrideActuation(estado, True)
    exo.overrideActuation(estado, True)

    # condição inicial = primeiro instante da referência
    estado.setTime(float(ref["t"][0]))
    cs = modelo.getCoordinateSet()
    for nome, valores in ref["coords"].items():
        if cs.contains(nome):
            cs.get(nome).setValue(estado, float(valores[0]), False)
    c = cs.get(junta)
    c.setValue(estado, float(ref["theta_d"][0]))
    c.setSpeedValue(estado, float(ref["dtheta_d"][0]))
    return modelo, estado, humano, exo


def ler_junta(modelo, estado, junta):
    """Saída da planta: (theta, dtheta) da junta livre."""
    c = modelo.getCoordinateSet().get(junta)
    return c.getValue(estado), c.getSpeedValue(estado)


def aplicar_torque(estado, humano, exo, tau_humano, tau_exo):
    """Escreve os dois torques no estado (valem até o próximo passo)."""
    humano.setOverrideActuation(estado, float(tau_humano))
    exo.setOverrideActuation(estado, float(tau_exo))


def avancar(modelo, estado, dt, precisao=1e-5):
    """Integra a dinâmica direta por dt segundos e devolve o novo estado."""
    m = osim.Manager(modelo)
    m.setIntegratorAccuracy(precisao)
    m.initialize(estado)
    return osim.State(m.integrate(estado.getTime() + dt))