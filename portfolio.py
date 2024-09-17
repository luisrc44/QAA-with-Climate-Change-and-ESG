
# Class to initialize a portfolio with its assets and methods to calculate expected return and total risk, footprints.
class Portfolio:
    def __init__(self):
        """
        Inicializa un portafolio vacío.
        """
        self.assets = []
    
    def add_asset(self, asset):
        """
        Agrega un activo al portafolio.
        
        :param asset: Instancia de la clase Asset o GreenAsset
        """
        self.assets.append(asset)
    
    def portfolio_info(self):
        """
        Muestra información de todos los activos en el portafolio.
        """
        for asset in self.assets:
            asset.info()
    
    def expected_portfolio_return(self):
        """
        Calcula el retorno esperado del portafolio basado en los retornos esperados de los activos.
        """
        total_return = sum(asset.expected_return for asset in self.assets)
        return total_return / len(self.assets)
    
    def total_risk(self):
        """
        Calcula el riesgo total del portafolio (simplificación asumiendo activos no correlacionados).
        """
        total_risk = sum(asset.risk for asset in self.assets)
        return total_risk / len(self.assets)

    def total_carbon_footprint(self):
        """
        Calcula la huella de carbono total del portafolio.
        """
        total_footprint = sum(asset.carbon_footprint for asset in self.assets if asset.carbon_footprint is not None)
        return total_footprint

    def optimize_esg_portfolio(self, risk_tolerance):
        """
        Optimiza el portafolio seleccionando los activos con mejores puntajes ESG.
        Prioriza activos con alto puntaje ESG en relación a su retorno/riesgo.
        
        :param risk_tolerance: Tolerancia al riesgo (número de activos a seleccionar basado en menor riesgo)
        :return: Lista de los activos seleccionados para el portafolio optimizado.
        """
        # Filtrar solo los activos verdes con puntajes ESG
        green_assets = [asset for asset in self.assets if isinstance(asset, GreenAsset)]
        
        # Ordenar activos verdes según la relación retorno/riesgo ajustado por ESG
        sorted_assets = sorted(green_assets, key=lambda x: (x.expected_return / x.risk) * x.esg_score, reverse=True)
        
        # Seleccionar activos según la tolerancia al riesgo
        optimized_portfolio = sorted_assets[:risk_tolerance]
        
        return optimized_portfolio


import matplotlib.pyplot as plt

class Visualizer:
    
    @staticmethod
    def plot_portfolio(assets):
        """
        Crea un gráfico de dispersión de los activos en el portafolio, mostrando retorno esperado vs riesgo.
        Se agrega el tamaño de los puntos en función del puntaje ESG.
        
        :param assets: Lista de activos a graficar (GreenAsset o Asset).
        """
        # Listas para guardar los valores de retorno, riesgo y puntajes ESG
        returns = [asset.expected_return for asset in assets]
        risks = [asset.risk for asset in assets]
        esg_scores = [(asset.esg_score if isinstance(asset, GreenAsset) else 0) for asset in assets]
        labels = [asset.name for asset in assets]
        
        # Crear el gráfico
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(risks, returns, c=esg_scores, cmap='Greens', s=[score*100 for score in esg_scores], alpha=0.6, edgecolor='k')
        plt.colorbar(scatter, label='ESG Score')
        plt.xlabel('Risk (Volatility)')
        plt.ylabel('Expected Return')
        plt.title('Portfolio Assets: Expected Return vs Risk')
        
        # Añadir etiquetas a los puntos
        for i, label in enumerate(labels):
            plt.annotate(label, (risks[i], returns[i]), textcoords="offset points", xytext=(5,-5), ha='center')
        
        plt.grid(True)
        plt.show()


# Testing the classes
# Crear algunos activos
stock = Asset("Company XYZ", "Stock", 0.08, 0.15)
bond = Asset("Gov Bond", "Bond", 0.03, 0.05)
green_bond = GreenAsset("Green Bond Fund", "Green Bond", 0.04, 0.03, 0.10, 0.8, 0.93, 0.72, 0.4, 0.2, 0.4, 0.02)

# Crear un portafolio y agregar activos
portfolio = Portfolio()
portfolio.add_asset(stock)
portfolio.add_asset(bond)
portfolio.add_asset(green_bond)

# Mostrar información del portafolio
portfolio.portfolio_info()

# Calcular retorno y riesgo del portafolio
print(f"Expected Portfolio Return: {portfolio.expected_portfolio_return()}")
print(f"Total Portfolio Risk: {portfolio.total_risk()}")

# Optimizar el portafolio basado en ESG con una tolerancia al riesgo de 2 activos
optimized_portfolio = portfolio.optimize_esg_portfolio(risk_tolerance=2)

# Mostrar el gráfico de los activos optimizados
Visualizer.plot_portfolio(optimized_portfolio)

# Mostrar información de los activos seleccionados
for asset in optimized_portfolio:
    asset.green_info()
