"""Filtro passa-baixa Butterworth de 2ª ordem, aplicado sem atraso.

Passa-baixa: deixa passar as variações lentas do sinal (o movimento da
marcha) e corta as rápidas (tremidos dos marcadores, erro de medição).

Butterworth: tipo de filtro que não deforma a parte que passa (resposta
plana até o corte). É o padrão em biomecânica.

Sem atraso: todo filtro atrasa o sinal. Aplicando uma vez do início para o
fim e outra do fim para o início, os atrasos se cancelam e a curva filtrada
fica alinhada no tempo com a original (o "filtfilt" do MATLAB).
"""
import numpy as np


def coeficientes(fc, fs):
    """Coeficientes (b, a) do Butterworth de 2ª ordem.

    fc: frequência de corte (Hz) -> variações mais rápidas que isso são cortadas
    fs: taxa de amostragem do sinal (Hz) -> pontos por segundo
    Obtidos pela transformação bilinear do filtro analógico.
    """
    w = np.tan(np.pi * fc / fs)
    a0 = 1 + np.sqrt(2) * w + w * w
    b = np.array([w * w, 2 * w * w, w * w]) / a0
    a = np.array([1.0, 2 * (w * w - 1) / a0, (1 - np.sqrt(2) * w + w * w) / a0])
    return b, a


def _uma_passada(b, a, x):
    """Aplica o filtro do início ao fim do vetor.
    Cada saída depende das 3 últimas entradas e das 2 últimas saídas:
    y[i] = b0*x[i] + b1*x[i-1] + b2*x[i-2] - a1*y[i-1] - a2*y[i-2]
    """
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = x[0]          # começa "parado" no primeiro valor
    for i, xi in enumerate(x):
        yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def filtrar(x, fc, fs):
    """Filtra o vetor x sem atraso.

    As bordas são estendidas por espelhamento antes de filtrar. Sem isso o
    filtro "estranha" o começo e o fim do sinal e distorce essas regiões.
    """
    x = np.asarray(x, dtype=float)
    n = min(len(x) - 1, int(3 * fs / fc))          # tamanho da extensão
    inicio = 2 * x[0] - x[n:0:-1]
    fim = 2 * x[-1] - x[-2:-n - 2:-1]
    estendido = np.concatenate([inicio, x, fim])

    b, a = coeficientes(fc, fs)
    ida = _uma_passada(b, a, estendido)
    volta = _uma_passada(b, a, ida[::-1])[::-1]
    return volta[n:n + len(x)]                      # remove a extensão