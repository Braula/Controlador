"""Pré-processamento da referência: IK filtrado + ID recalculado.

Por que existe: o IK vem dos marcadores e traz pequenos tremidos de medição.
O controlador usa a derivada do ângulo (velocidade), que amplifica esses
tremidos, e o Kd os transforma em picos de torque. Filtrar o IK remove isso.

Por que recalcular o ID: o torque de referência precisa ser coerente com os
ângulos de referência. Se só o IK for filtrado, o ID antigo deixa de bater
com ele e a simulação piora. Por isso o ID é calculado de novo no OpenSim,
a partir do IK já filtrado.

Etapas:
  1. Lê o IK original.
  2. Reamostra de ~60 para 200 pontos por segundo (interpolação linear).
  3. Filtra todas as coordenadas com passa-baixa de 6 Hz, sem atraso.
  4. Grava o IK filtrado (.mot).
  5. Roda o Inverse Dynamics do OpenSim com esse IK e o modelo SEM exo.
  6. Grava o ID filtrado (.sto).

Saída: IK_filtrado.mot e ID_filtrado.sto na PASTA. No simular.py, troque
ARQ_IK e ARQ_ID para esses dois arquivos.
"""
import os
import numpy as np
import opensim as osim

from referencia import ler_sto
from filtro import filtrar

osim.Logger.setLevelString("warn")

# ---------------- CONFIGURAÇÃO ----------------
PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")
ARQ_IK = os.path.join(PASTA, "IKcorrigido.mot")          # IK original
ARQ_MODELO = os.path.join(PASTA, "multi_sem_exo.osim")   # referência = marcha SEM exo
ARQ_GRF = os.path.join(PASTA, "subject01_walk1_grf.xml") # forças do solo
SAIDA_IK = os.path.join(PASTA, "IK_filtrado.mot")
SAIDA_ID = os.path.join(PASTA, "ID_filtrado.sto")

FS = 200.0   # pontos por segundo do IK filtrado
FC = 6.0     # corte do filtro, Hz (padrão da literatura para marcha, Winter)
# ----------------------------------------------


def filtrar_ik():
    """Etapas 1 a 4: reamostra e filtra todas as colunas do IK."""
    ik, em_graus = ler_sto(ARQ_IK)
    nomes = list(ik)                                   # 'time' + 23 coordenadas
    t = np.arange(ik["time"][0], ik["time"][-1] + 1e-9, 1 / FS)

    colunas = [t]
    for nome in nomes[1:]:
        reamostrado = np.interp(t, ik["time"], ik[nome])
        colunas.append(filtrar(reamostrado, FC, FS))
    dados = np.column_stack(colunas)

    # cabeçalho no formato que o OpenSim lê
    with open(SAIDA_IK, "w") as f:
        f.write("Coordinates\nversion=1\n")
        f.write(f"nRows={len(t)}\nnColumns={len(nomes)}\n")
        f.write(f"inDegrees={'yes' if em_graus else 'no'}\nendheader\n")
        f.write("\t".join(nomes) + "\n")
        for linha in dados:
            f.write("\t".join(f"{v:.8f}" for v in linha) + "\n")

    print(f"IK filtrado: {len(t)} pontos, {FS:g} Hz, corte {FC:g} Hz -> {SAIDA_IK}")
    return t[0], t[-1]


def recalcular_id(t0, t1):
    """Etapas 5 e 6: Inverse Dynamics com o IK filtrado."""
    # O .xml cita o .mot das forças só pelo nome. Gravamos uma cópia com o
    # caminho completo para o OpenSim achar o .mot de qualquer pasta.
    cargas = osim.ExternalLoads(ARQ_GRF, True)
    pasta_grf = os.path.dirname(os.path.abspath(ARQ_GRF))
    cargas.setDataFileName(os.path.join(pasta_grf, cargas.getDataFileName().strip()))
    grf_temp = os.path.join(PASTA, "grf_caminho_completo.xml")
    cargas.printToXML(grf_temp)

    # O modelo precisa ficar guardado numa variável enquanto a ferramenta roda.
    # Se for criado direto dentro de setModel(...), o Python o apaga da
    # memória e o OpenSim fecha com erro (segmentation fault).
    modelo = osim.Model(ARQ_MODELO)
    modelo.initSystem()

    ferramenta = osim.InverseDynamicsTool()
    ferramenta.setModel(modelo)
    ferramenta.setCoordinatesFileName(SAIDA_IK)
    ferramenta.setExternalLoadsFileName(grf_temp)
    ferramenta.setLowpassCutoffFrequency(-1)          # já filtrado: não filtra de novo
    excluir = osim.ArrayStr()
    excluir.append("Muscles")                         # ID calcula o torque líquido da junta
    ferramenta.setExcludedForces(excluir)
    ferramenta.setStartTime(t0)
    ferramenta.setEndTime(t1)
    ferramenta.setResultsDir(os.path.dirname(SAIDA_ID))
    ferramenta.setOutputGenForceFileName(os.path.basename(SAIDA_ID))
    ferramenta.run()

    os.remove(grf_temp)
    print(f"ID recalculado -> {SAIDA_ID}")


if __name__ == "__main__":
    t0, t1 = filtrar_ik()
    recalcular_id(t0, t1)