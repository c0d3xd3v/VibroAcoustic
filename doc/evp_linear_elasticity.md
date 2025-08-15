# Eigenwertproblem der linearen Elastizität

Im Fokus steht die Bestimmung der Eigenfrequenzen und Eigenschwingformen elastischer Körper im Rahmen der linearen Elastizitätstheorie.

## Zeitabhängige Bewegungsgleichung

Ausgangspunkt ist die **zeitabhängige Navier-Lamé-Gleichung** in der linearen Elastizität:

$$
\rho \, \ddot{u}(x,t) = \nabla \cdot \sigma(u(x,t)) + f_{\text{ext}}(x,t)
$$

mit:

* $\rho$: Massendichte
* $\sigma(u)$: Spannungstensor mit Hooke’schem Gesetz

$$
\sigma(u) = 2\mu \, \varepsilon(u) + \lambda \, \text{div}(u) \cdot I
$$

* $\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)$: Verzerrungstensor
* $f_{\text{ext}}(x,t)$: äußere Krafteinwirkung

Das ist im Prinzip **das Kontinuums-Analogon** zu $F = m \cdot a$ aus der klassischen Mechanik.

* $\rho$ ist die **Massendichte** (Masse pro Volumen), also quasi „Masse pro Einheit Raum“.
* $\ddot u(x,t)$ ist die **Beschleunigung** des Punktes im Material.
* $\nabla \cdot \sigma(u(x,t))$ ist die **innere Kraftdichte** (also Kraft pro Volumen), die durch Spannungen im Material verursacht wird.

## Annahme: Keine äußeren Kräfte

Für die **Modalanalyse** betrachten wir freie Schwingungen – also:

$$
f_{\text{ext}} = 0
$$

Dann ergibt sich:

$$
\rho \, \ddot{u}(x,t) = \nabla \cdot \sigma(u(x,t))
$$

## Frequenzraum via Fourier-Transformation

Anstatt eine harmonische Lösung direkt anzusetzen, kann man die Fourier-Transformation in der Zeit auf $u(x,t)$ anwenden:

$$
\tilde{u}(x,\omega) = \int_{-\infty}^{\infty} u(x,t) \, e^{-i \omega t} \, dt
$$

Die zweite Zeitableitung transformiert sich zu:

$$
\mathcal{F}[\ddot{u}(x,t)](\omega) = -\omega^2 \, \tilde{u}(x,\omega)
$$

Diese Beziehung kann man elegant **direkt beweisen** (siehe Anhang). Damit ergibt sich im Frequenzraum:

$$
-\omega^2 \, \rho \, \tilde{u}(x,\omega) = \nabla \cdot \sigma(\tilde{u}(x,\omega))
$$

Das ist exakt das frequenzraum-basierte Eigenwertproblem der linearen Elastizität.

## Variationsformulierung (Schwache Form)

Das zugehörige Eigenwertproblem lautet:

Finde $u \in V \setminus \{0\}$ und $\lambda \in \mathbb{R}_+$, sodass:

$$
a(u, v) = \lambda \, b(u, v) \quad \forall v \in V
$$

mit:

* **Steifigkeitsform**:

$$
a(u, v) = \int_\Omega 2\mu \, \varepsilon(u) : \varepsilon(v) + \lambda \, \text{div}(u) \, \text{div}(v) \, dx
$$

* **Massenform**:

$$
b(u, v) = \int_\Omega \rho \, u \cdot v \, dx
$$

## Problem: Starre Körperbewegungen

Ohne Dirichlet-Bedingungen hat das Eigenwertproblem **Nullfrequenzlösungen**:

* **Translationen**: konstante Verschiebung
* **Rotationen**: drehende Bewegung ohne Deformation

Diese **starren Körpermoden** sind zwar mathematisch korrekt, aber oft unerwünscht in der Modalanalyse.

## Lösung mit Lagrange-Multiplikatoren

Um die starren Bewegungen systematisch zu eliminieren, erweitern wir das Eigenwertproblem um Nebenbedingungen:

$$
b(u, \phi_i) = 0 \quad \forall i = 1, \dots, n
$$

mit $\phi_i$: starren Bewegungen (z. B. $(1,0,0), (0,1,0), (0,0,1), (-y,x,0), \dots$)

Diese Bedingungen werden per **Lagrange-Multiplikatoren** eingebaut.

### Gemischte Eigenwertformulierung

Finde $u \in V, \lambda \in \mathbb{R}^n$, sodass:

$$
\begin{aligned}
a(u, v) + \sum_{i=1}^n \lambda_i \, b(v, \phi_i) &= \mu \, b(u, v) \quad &\forall v \in V \\
b(u, \phi_i) &= 0 \quad &\forall i = 1, \dots, n
\end{aligned}
$$

Dies ergibt ein **gemischtes Eigenwertproblem**, das alle starren Bewegungen exakt eliminiert.

## Anhang

#### **Beweis: Fourier-Transformation einer zeitlichen Ableitung**

Wir verwenden die Definition der kontinuierlichen Fourier-Transformation:

$$
\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) \cdot e^{-i\omega t} \, dt
$$

Jetzt nehmen wir die Fourier-Transformation der Ableitung $f'(t) = \frac{d}{dt}f(t)$:

$$
\mathcal{F}\left\{ f'(t) \right\} = \int_{-\infty}^{\infty} f'(t) \cdot e^{-i\omega t} \, dt
$$

Das integrieren wir **partiell**, mit:

* $u = e^{-i\omega t} \Rightarrow \frac{du}{dt} = -i\omega e^{-i\omega t}$
* $dv = f'(t) dt \Rightarrow v = f(t)$

Anwendung der partiellen Integration:

$$
\int f'(t) e^{-i\omega t} dt = \left[ f(t) e^{-i\omega t} \right]_{-\infty}^{\infty} - \int f(t) \cdot (-i\omega) e^{-i\omega t} dt
$$

$$
= \underbrace{\left[ f(t) e^{-i\omega t} \right]_{-\infty}^{\infty}}_{\text{Grenzwertterm}} + i\omega \int f(t) e^{-i\omega t} dt
$$

Die Funktion $f(t)$ muss „schnell genug gegen 0 gehen“ für $t \to \pm \infty$, also z. B. absolut integrierbar sein. Dann gilt:

$$
\lim_{t \to \pm\infty} f(t) e^{-i\omega t} = 0
$$

Daher verschwindet der Randterm, und es bleibt:

$$
\mathcal{F}\left\{ f'(t) \right\} = i\omega \cdot \hat{f}(\omega)
$$

**q.e.d.**
