import numpy as np
import matplotlib.pyplot as plt

class SimplifiedDICE:
    def __init__(self, co2_emissions, temperature_pct_change, base_temperature=1.0, delta=0.002):
        """
        Inicializa el modelo simplificado de DICE utilizando datos de cambio porcentual de temperatura.

        :param co2_emissions: Serie temporal de emisiones de CO2.
        :param temperature_pct_change: Serie temporal del cambio porcentual de temperatura.
        :param base_temperature: Temperatura inicial en grados Celsius (por encima de los niveles preindustriales).
        :param delta: Parámetro que ajusta la sensibilidad del daño climático.
        """
        self.co2_emissions = co2_emissions
        self.temperature_pct_change = temperature_pct_change
        self.base_temperature = base_temperature
        self.delta = delta
        self.T = [base_temperature]  # Inicializa la temperatura con el valor base
        self.D = []  # Daño económico
    
    def convertir_temperatura(self, pct_change):
        """
        Convierte el cambio porcentual en temperatura absoluta.
        :param pct_change: Cambio porcentual de la temperatura.
        :return: Temperatura absoluta en grados Celsius.
        """
        return self.T[-1] * (1 + pct_change / 100)

    def dano_economico(self, T):
        """
        Calcula el daño económico como función de la temperatura.
        :param T: Temperatura global en el tiempo t.
        :return: Daño económico en % del PIB.
        """
        return T**2 / (1 + self.delta * T**2)
    
    def run_simulation(self, years):
        """
        Corre la simulación utilizando los datos de cambio porcentual de temperatura.
        
        :param years: Número de años para correr la simulación.
        """
        for year in range(1, years + 1):
            if year < len(self.temperature_pct_change):
                # Usar los datos reales de cambio porcentual de temperatura
                new_T = self.convertir_temperatura(self.temperature_pct_change[year])
            else:
                # Si no hay más datos, asumir un cambio porcentual constante
                new_T = self.T[-1] * 1.02  # Supongamos un aumento del 2%
            
            self.T.append(new_T)
            
            # Calcular el daño económico
            new_D = self.dano_economico(new_T)
            self.D.append(new_D)
    
    def plot_results(self):
        """
        Grafica los resultados de la simulación: temperatura y daño económico.
        """
        years = range(len(self.T))
        plt.plot(years, self.T, label='Temperatura (°C)', color='tab:red')
        plt.xlabel('Años')
        plt.ylabel('Temperatura (°C)')
        plt.title('Evolución de la Temperatura Global')
        plt.grid(True)
        plt.show()

        # Graficar el daño económico
        plt.plot(years[1:], self.D, label='Daño Económico (%)', color='tab:green')
        plt.xlabel('Años')
        plt.ylabel('Daño Económico (%)')
        plt.title('Impacto Económico del Cambio Climático')
        plt.grid(True)
        plt.show()
