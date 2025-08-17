class Material:
    def __init__(self, type="linear", E=0.0, nu=0.0, rho=0.0, units="SI"):
        """
        Intern werden alle Größen in SI gespeichert:
        - E in Pa
        - rho in kg/m³
        - nu dimensionslos
        """
        self.type = type
        self.nu = nu

        # Umrechnung in SI, falls Eingabe in mm-System
        if units == "mm":
            self.E_SI = E * 1e6       # N/mm² -> Pa
            self.rho_SI = rho * 1e9   # kg/mm³ -> kg/m³
        else:
            self.E_SI = E
            self.rho_SI = rho

        # Lamé-Parameter
        if (abs(self.E_SI) > 1e-15 and 
            abs(nu) > 1e-15 and 
            abs(nu + 1) > 1e-15 and 
            abs(nu - 0.5) > 1e-15 and 
            abs(self.rho_SI) > 1e-15):
            self.mu = self.E_SI / (2 * (1 + nu))
            self.lam = self.E_SI * nu / ((1 + nu) * (1 - 2 * nu))
        else:
            self.mu = 0.0
            self.lam = 0.0

    # Methoden zum Abrufen der Werte in beliebigem System
    def E(self, units="SI"):
        """Gibt E zurück, units="SI" (Pa) oder "mm" (N/mm²)"""
        if units == "mm":
            return self.E_SI * 1e-6
        return self.E_SI

    def rho(self, units="SI"):
        """Gibt rho zurück, units="SI" (kg/m³) oder "mm" (kg/mm³)"""
        if units == "mm":
            return self.rho_SI * 1e-9
        return self.rho_SI

    def wave_speed(self):
        """Charakteristische Geschwindigkeit sqrt(E/rho) in m/s"""
        if self.E_SI > 0 and self.rho_SI > 0:
            return (self.E_SI / self.rho_SI) ** 0.5
        return 0.0

# Beispiele
aluminium = Material("linear", 69e3, 0.33, 2.5355e-6, units="mm")
steel = Material("linear", 220e3, 0.28, 7.85e-6, units="mm")

print("Aluminium E (Pa): {:.3e}".format(aluminium.E("SI")))
print("Aluminium E (N/mm²): {:.3e}".format(aluminium.E("mm")))
print("Aluminium rho (kg/m³): {:.3e}".format(aluminium.rho("SI")))
print("Aluminium rho (kg/mm³): {:.3e}".format(aluminium.rho("mm")))
print("Aluminium wave speed (m/s): {:.3e}".format(aluminium.wave_speed()))
