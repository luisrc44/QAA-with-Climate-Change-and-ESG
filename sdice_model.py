import numpy as np
import matplotlib.pyplot as plt

class ScenarioDICE(SimplifiedDICE):
    def __init__(self, initial_temperature=1.0, initial_emissions=35, climate_sensitivity=3.0, 
                 abatement_cost=0.02, discount_rate=0.02):
        # Llamamos al constructor del modelo SimplifiedDICE original
        super().__init__(initial_temperature, initial_emissions, climate_sensitivity, discount_rate)
        self.abatement_cost = abatement_cost

    def generate_random_scenarios(self, num_scenarios):
        """
        Genera múltiples escenarios con valores aleatorios para los parámetros.
        
        :param num_scenarios: Número de escenarios a generar.
        :return: Una lista de diccionarios con los parámetros de los escenarios aleatorios.
        """
        scenarios = {}
        for i in range(num_scenarios):
            scenario_name = f"Escenario Aleatorio {i+1}"
            scenarios[scenario_name] = {
                "initial_temperature": 1.0,  # Temperatura inicial fija
                "initial_emissions": np.random.uniform(20, 50),  # Emisiones iniciales entre 20 y 50 GtC
                "climate_sensitivity": np.random.uniform(2.0, 4.5),  # Sensibilidad climática entre 2.0 y 4.5 grados
                "abatement_cost": np.random.uniform(0.01, 0.05),  # Costo de reducción entre 1% y 5%
                "discount_rate": np.random.uniform(0.02, 0.04)  # Tasa de descuento entre 2% y 4%
            }
        return scenarios

    def simulate_multiple_scenarios(self, scenarios, years=100):
        """
        Simula múltiples escenarios, cada uno con diferentes parámetros.
        
        :param scenarios: Una lista de diccionarios con los parámetros de los diferentes escenarios.
        :param years: Número de años para la simulación.
        :return: Resultados de las simulaciones en cada escenario.
        """
        results = {}
        
        for scenario_name, params in scenarios.items():
            # Configurar el escenario
            self.__init__(**params)  # Reiniciar el modelo con los parámetros del escenario
            temperatures, emissions, damages, economic_outputs = self.simulate_climate_impact(years)
            results[scenario_name] = {
                'temperatures': temperatures,
                'emissions': emissions,
                'damages': damages,
                'economic_outputs': economic_outputs
            }

        return results

    def plot_scenarios(self, results):
        """
        Genera gráficos comparativos para múltiples escenarios.
        
        :param results: Resultados de la simulación de escenarios múltiples.
        """
        years = np.arange(100)

        # Gráficos de temperatura por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['temperatures'], label=scenario_name)
        plt.title("Comparación de escenarios - Temperatura Global")
        plt.xlabel("Años")
        plt.ylabel("Temperatura (°C)")
        plt.legend()
        plt.grid()

        # Gráficos de emisiones por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['emissions'], label=scenario_name)
        plt.title("Comparación de escenarios - Emisiones de CO2")
        plt.xlabel("Años")
        plt.ylabel("Emisiones (GtC)")
        plt.legend()
        plt.grid()

        # Gráficos de daños económicos por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['damages'], label=scenario_name)
        plt.title("Comparación de escenarios - Daños Económicos")
        plt.xlabel("Años")
        plt.ylabel("Daños Económicos (%)")
        plt.legend()
        plt.grid()

        # Gráficos de producción económica por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['economic_outputs'], label=scenario_name)
        plt.title("Comparación de escenarios - Producción Económica")
        plt.xlabel("Años")
        plt.ylabel("Producción Económica (USD trillones)")
        plt.legend()
        plt.grid()

        plt.show()
