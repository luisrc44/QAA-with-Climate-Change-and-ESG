import numpy as np
import matplotlib.pyplot as plt

class SimplifiedDICE:
    def __init__(self, initial_temperature=1.0, initial_emissions=35, climate_sensitivity=3.0, discount_rate=0.02):
        """
        Inicializa el modelo DICE simplificado con parámetros iniciales.
        
        :param initial_temperature: Incremento inicial de la temperatura desde los niveles preindustriales (en grados Celsius).
        :param initial_emissions: Emisiones iniciales de CO2 (GtC por año).
        :param climate_sensitivity: Sensibilidad climática al CO2 (grados Celsius por duplicación de CO2).
        :param discount_rate: Tasa de descuento intertemporal para el cálculo económico.
        """
        self.temperature = initial_temperature
        self.emissions = initial_emissions
        self.initial_emissions = initial_emissions  # Corregir: almacenar las emisiones iniciales como un atributo
        self.climate_sensitivity = climate_sensitivity
        self.discount_rate = discount_rate

        # Coeficientes del modelo simplificado
        self.carbon_decay_rate = 0.1  # Tasa de decaimiento del carbono (aproximación)
        self.abatement_cost = 0.02  # Costo de reducción de emisiones
        self.damage_coefficient = 0.002  # Coeficiente de daño económico
        self.economic_output = 100  # Producto económico inicial (en trillones de dólares)

    def carbon_cycle(self, emissions):
        """
        Modela el ciclo del carbono: cómo las emisiones actuales afectan las concentraciones de CO2.
        
        :param emissions: Emisiones actuales de CO2 (GtC por año).
        :return: Nuevas concentraciones de CO2.
        """
        return emissions * (1 - self.carbon_decay_rate)

    def temperature_change(self, emissions):
        """
        Calcula el cambio en la temperatura global basado en las emisiones.
        
        :param emissions: Emisiones actuales de CO2 (GtC por año).
        :return: Cambio proyectado en la temperatura global (grados Celsius).
        """
        # Sensibilidad climática: aumento de temperatura por duplicación de CO2
        delta_temp = self.climate_sensitivity * np.log2(emissions / self.initial_emissions)
        self.temperature += delta_temp
        return self.temperature

    def economic_damage(self, temperature):
        """
        Calcula el impacto económico del cambio climático sobre la producción.
        
        :param temperature: Cambio en la temperatura global (grados Celsius).
        :return: Daño económico (% de la producción económica perdida).
        """
        return self.damage_coefficient * (temperature ** 2)

    def simulate_climate_impact(self, years=100):
        """
        Simula el impacto del cambio climático en la economía durante un número de años.
        
        :param years: Número de años para la simulación.
        :return: Historias de temperatura, emisiones y daños económicos.
        """
        temperatures = []
        emissions = []
        damages = []
        economic_outputs = []

        for year in range(years):
            # Actualizar emisiones y temperatura
            current_emissions = self.carbon_cycle(self.emissions)
            current_temperature = self.temperature_change(current_emissions)
            
            # Calcular daños económicos
            current_damage = self.economic_damage(current_temperature)
            economic_output = self.economic_output * (1 - current_damage)

            # Guardar resultados
            temperatures.append(current_temperature)
            emissions.append(current_emissions)
            damages.append(current_damage)
            economic_outputs.append(economic_output)

            # Actualizar para el próximo año
            self.emissions *= (1 - self.abatement_cost)  # Reducir emisiones gradualmente
            self.economic_output *= (1 + self.discount_rate)  # Crecimiento económico con descuento

        return temperatures, emissions, damages, economic_outputs

    def plot_simulation(self, temperatures, emissions, damages, economic_outputs):
        """
        Genera gráficos de las simulaciones de temperatura, emisiones y daños económicos.
        
        :param temperatures: Historia de las temperaturas simuladas.
        :param emissions: Historia de las emisiones simuladas.
        :param damages: Historia de los daños económicos simulados.
        :param economic_outputs: Historia de la producción económica simulada.
        """
        years = np.arange(len(temperatures))

        # Gráfico de temperatura
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(years, temperatures, label="Temperatura (Celsius)")
        plt.title("Cambio de temperatura global")
        plt.xlabel("Años")
        plt.ylabel("Temperatura (°C)")
        plt.grid()

        # Gráfico de emisiones
        plt.subplot(2, 2, 2)
        plt.plot(years, emissions, label="Emisiones de CO2 (GtC)", color='orange')
        plt.title("Emisiones de CO2")
        plt.xlabel("Años")
        plt.ylabel("Emisiones (GtC)")
        plt.grid()

        # Gráfico de daños económicos
        plt.subplot(2, 2, 3)
        plt.plot(years, damages, label="Daños económicos (%)", color='red')
        plt.title("Impacto económico del cambio climático")
        plt.xlabel("Años")
        plt.ylabel("Daños económicos (%)")
        plt.grid()

        # Gráfico de producción económica
        plt.subplot(2, 2, 4)
        plt.plot(years, economic_outputs, label="Producción económica (trillones USD)", color='green')
        plt.title("Producción económica con cambio climático")
        plt.xlabel("Años")
        plt.ylabel("Producción económica (USD trillones)")
        plt.grid()

        plt.tight_layout()
        plt.show()


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
        plt.grid()

        # Gráficos de emisiones por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['emissions'], label=scenario_name)
        plt.title("Comparación de escenarios - Emisiones de CO2")
        plt.xlabel("Años")
        plt.ylabel("Emisiones (GtC)")
        plt.grid()

        # Gráficos de daños económicos por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['damages'], label=scenario_name)
        plt.title("Comparación de escenarios - Daños Económicos")
        plt.xlabel("Años")
        plt.ylabel("Daños Económicos (%)")
        plt.grid()

        # Gráficos de producción económica por escenario
        plt.figure(figsize=(10, 6))
        for scenario_name, data in results.items():
            plt.plot(years, data['economic_outputs'], label=scenario_name)
        plt.title("Comparación de escenarios - Producción Económica")
        plt.xlabel("Años")
        plt.ylabel("Producción Económica (USD trillones)")
        plt.grid()

        plt.show()


