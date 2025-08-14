# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:31:38 2025

@author: valen
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # Para calcular la curva de Gauss
import math  # Para cálculos matemáticos

def calcular_SNR(voltaje, ruido):
    """Calcula la relación señal-ruido (SNR) y devuelve la SNR en dB."""
    potencia_senal = np.sqrt(np.mean(voltaje**2))
    potencia_ruido = np.sqrt(np.mean(ruido**2))
    SNR = potencia_senal / potencia_ruido
    SNR_dB = 10 * np.log10(SNR)
    return SNR_dB

def calcular_histograma_manual(voltaje, num_bins=50):
    """Calcula el histograma manualmente contando valores en intervalos."""
    min_val = min(voltaje)
    max_val = max(voltaje)
    bin_width = (max_val - min_val) / num_bins
    histograma = [0] * num_bins
    
    for valor in voltaje:
        bin_index = int((valor - min_val) / bin_width)
        if bin_index == num_bins:
            bin_index -= 1
        histograma[bin_index] += 1
    
    return histograma, min_val, max_val, bin_width, num_bins, max(histograma)

def calcular_pdf_manual(voltaje, num_bins, max_hist):
    """Calcula la función de densidad de probabilidad manualmente y la escala para ajustarla al histograma."""
    media = sum(voltaje) / len(voltaje)
    desviacion = math.sqrt(sum((x - media) ** 2 for x in voltaje) / len(voltaje))
    
    x_vals = np.linspace(min(voltaje), max(voltaje), 100)
    pdf_vals = []
    
    for x in x_vals:
        pdf = (1 / (desviacion * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - media) / desviacion) ** 2)
        pdf_vals.append(pdf)
    
    # Ajustar la escala de la PDF al histograma
    scale_factor = max_hist / max(pdf_vals)
    pdf_vals = [p * scale_factor for p in pdf_vals]
    
    return x_vals, pdf_vals

def graficar_histograma_y_pdf_manual(voltaje, Histograma_manual):
    """Grafica el histograma manual junto con la función de probabilidad (campana de Gauss) manual escalada."""
    histograma, min_val, max_val, bin_width, num_bins, max_hist = calcular_histograma_manual(voltaje)
    bins = np.linspace(min_val, max_val, len(histograma))
    x_vals, pdf_vals = calcular_pdf_manual(voltaje, num_bins, max_hist)
    
    plt.figure(figsize=(10, 6))
    plt.bar(bins, histograma, width=bin_width, color='skyblue', edgecolor='black', alpha=0.7, label='Histograma Manual')
    plt.plot(x_vals, pdf_vals, color='red', linewidth=2, label='Campana de Gauss (Manual)')
    plt.title(Histograma_manual)
    plt.xlabel('Voltaje (mV)')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.grid()
    plt.show()




def calcular_coeficiente_variacion(voltaje):
    """Calcula el coeficiente de variación."""
    media = np.mean(voltaje)
    desviacion = np.std(voltaje)
    coef_var = desviacion / media
    return coef_var

def graficar_señal(voltaje, titulo, color):
    """Función para graficar la señal"""
    plt.figure(figsize=(12, 6))
    plt.plot(voltaje, color=color, linewidth=1)
    plt.title(titulo)
    plt.xlabel('Muestras (ms)')
    plt.ylabel('Amplitud (mV)')
    plt.grid()
    plt.show()

def mostrar_histograma_con_gauss(voltaje, titulo):
    """Muestra el histograma de la señal con la curva de Gauss superpuesta"""
    media = np.mean(voltaje)
    desviacion = np.std(voltaje)
    
    # Crear histograma
    plt.figure(figsize=(10, 6))
    plt.hist(voltaje, bins=50, color='skyblue', edgecolor='black', density=True, label='Histograma')

    # Crear la curva de Gauss
    x = np.linspace(min(voltaje), max(voltaje), 1000)
    gauss = norm.pdf(x, loc=media, scale=desviacion)
    plt.plot(x, gauss, color='red', linewidth=2, label='Campana de Gauss')

    # Configuración de la gráfica
    plt.title(titulo)
    plt.xlabel('Voltaje (mV)')
    plt.ylabel('Densidad de Probabilidad')
    plt.legend()
    plt.grid()
    plt.show()

def custom_mean(data):
    ## a mano
    
    """Calcula la media de una lista de números."""
    datos=data.copy()
    datos.tolist()
    total=0.0
    contador=0
    for x in data:
      total+=x
      contador+=1
    return total/contador if contador != 0 else 0.0


def custom_std_dev(data):
    """Calcula la desviación estándar muestral (raíz cuadrada de la varianza)."""
    datos=data.copy()
    datos.tolist()
    mean = custom_mean(data)
    diferencia_raiz_suma = 0.0
    contador=0
    for x in data:
      diferencia_raiz_suma += (x - mean) ** 2
      contador+=1
    return (diferencia_raiz_suma / contador) ** 0.5 if contador != 0 else 0.0


def custom_coeff_variation(data):
    """Calcula el coeficiente de variación en porcentaje."""
    datos=data.copy()
    datos.tolist()
    mean = custom_mean(data)
    if mean == 0:
        return 0.0  
    return (custom_std_dev(data) / mean) * 100

signal = wfdb.rdrecord('C:/Users/valen/Downloads/drive01')

volaje_canal_1 = signal.p_signal[:, 1]


media = custom_mean(volaje_canal_1)
std_dev = custom_std_dev(volaje_canal_1)
cv = custom_coeff_variation(volaje_canal_1)

print(f"Canal: {signal.sig_name[1]}")
print(f"Media: {media:.4f} {volaje_canal_1}")
print(f"Desviación estándar: {std_dev:.4f} {signal.units[1]}")
print(f"Coef. Variación: {cv:.2f}%")


def menu_interactivo():
    """Menú interactivo para seleccionar el tipo de ruido"""
    signal = wfdb.rdrecord('C:/Users/valen/Downloads/drive01')
    voltaje_canal_1 = signal.p_signal[:, 1]
    print("Muestras cargadas.")

    # Calcular la media, desviación estándar y coeficiente de variación de la señal original
    media = np.mean(voltaje_canal_1)
    desviacion = np.std(voltaje_canal_1)
    coef_var = calcular_coeficiente_variacion(voltaje_canal_1)

    print(f"Media Programada : {media} V")
    print(f"Desviación estándar Programada: {desviacion} V")
    print(f"Coeficiente de variación Programada : {coef_var} % ")

    # Mostrar señal original
    graficar_señal(voltaje_canal_1, "Señal Original (Canal 1)", "blue")
    mostrar_histograma_con_gauss(voltaje_canal_1, "Histograma del Voltaje (Señal Original)")

    while True:
        print("\nSelecciona el tipo de ruido a agregar a la señal:")
        print("1. Ruido Gaussiano")
        print("2. Ruido de Impulso")
        print("3. Ruido Artefacto")
        print("4. Histograma Manual")
        print("5. Salir")
        
        opcion = input("Ingresa el número de tu opción: ")

        if opcion == "1":
            # Ruido Gaussiano
            ruido_gaussiano = np.random.normal(0, desviacion, voltaje_canal_1.shape)
            voltaje_contaminado = voltaje_canal_1 + ruido_gaussiano
            SNR_gaussiano = calcular_SNR(voltaje_canal_1, ruido_gaussiano)
            

            print(f"SNR del ruido Gaussiano: {SNR_gaussiano} dB")
            
            graficar_señal(voltaje_contaminado, "Señal con Ruido Gaussiano", "red")
            mostrar_histograma_con_gauss(voltaje_contaminado, "Histograma de la Señal con Ruido Gaussiano")

        elif opcion == "2":
            # Ruido de Impulso
            probabilidad_impulso = 0.02
            amplitud_impulso = 0.2
            ruido_impulso = np.random.choice([0.02, amplitud_impulso, -amplitud_impulso], size=voltaje_canal_1.shape, p=[1 - probabilidad_impulso, probabilidad_impulso / 2, probabilidad_impulso / 2])
            voltaje_contaminado_impulso = voltaje_canal_1 + ruido_impulso
            SNR_impulso = calcular_SNR(voltaje_canal_1, ruido_impulso)

            print(f"SNR del ruido de Impulso: {SNR_impulso} dB")

            graficar_señal(voltaje_contaminado_impulso, "Señal con Ruido de Impulso", "green")
            mostrar_histograma_con_gauss(voltaje_contaminado_impulso, "Histograma de la Señal con Ruido de Impulso")

        elif opcion == "3":
            # Ruido Artefacto
            frecuencia_artefacto = 50
            amplitud_artefacto = 0.05
            t = np.arange(len(voltaje_canal_1)) / signal.fs
            ruido_artefacto = amplitud_artefacto * np.sin(2 * np.pi * frecuencia_artefacto * t)
            voltaje_contaminado_artefacto = voltaje_canal_1 + ruido_artefacto
            SNR_artefacto = calcular_SNR(voltaje_canal_1, ruido_artefacto)

            print(f"SNR del ruido Artefacto: {SNR_artefacto} dB")

            graficar_señal(voltaje_contaminado_artefacto, "Señal con Ruido Artefacto", "purple")
            mostrar_histograma_con_gauss(voltaje_contaminado_artefacto, "Histograma de la Señal con Ruido Artefacto")
        elif opcion == "4":
            graficar_histograma_y_pdf_manual(voltaje_canal_1, "Histograma Manual  (Señal Original)")
            plt.show()
        elif opcion == "5":
            print("Saliendo...")
            
            plt.show()
            break
        else:
            print("Opción no válida, intenta nuevamente.")

if __name__ == "__main__":
    menu_interactivo()

