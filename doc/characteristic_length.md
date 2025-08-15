### Was ist die charakteristische Länge?

Die **charakteristische Länge** ist eine typische Längenskala des Bauteils, zum Beispiel die Länge, Breite oder ein typischer Abmessungswert, der als Maß für die geometrische Ausdehnung dient.

### Warum ist die charakteristische Länge wichtig?

Die Eigenfrequenz $\omega$ eines Bauteils hängt näherungsweise vom Elastizitätsmodul $E$, der Dichte $\rho$ und der charakteristischen Länge $L$ wie folgt ab:

$$
\omega \sim \frac{1}{L} \sqrt{\frac{E}{\rho}}
$$

Das bedeutet:

- Je **kleiner** die charakteristische Länge $L$, desto **höher** die Eigenfrequenz $\omega$.
- Je **größer** die charakteristische Länge $L$, desto **niedriger** die Eigenfrequenz $\omega$.

> Wenn du die charakteristische Länge um den Faktor 10 änderst, ändert sich die Eigenfrequenz ungefähr um den Faktor $\frac{1}{10}$.

### Charakteristische Länge für Frequenzabschätzungen

Die charakteristische Länge $L_c$ ist typischerweise eine typische Ausdehnung des Bauteils, die die „Wellenlänge“ der Schwingungen beschreibt.

**Ein paar einfache Ansätze:**

1. **Geometrische Maße nehmen:**  
   Je nachdem, welche Dimension die Schwingung hauptsächlich prägt (z.B. Länge, Breite, Dicke) nimmst du diese als $L_c$.  
   Für einen Balken z.B.  
   
   $$
   L_c = \text{Länge des Balkens}.
   $$

2. **Volumen-basierter Ansatz:**  
   Wenn die Geometrie komplex ist, kannst du die charakteristische Länge als Kubikwurzel des Volumens definieren:  
   
   $$
   L_c = \sqrt[3]{\text{Volumen}}
   $$

### Abschätzung der unteren grenze des Frequenzbands

Die Ausbreitungsgeschwindigkeit $c$ berechnet sich mit

$$
c = \sqrt{\frac{E}{\rho}}
$$

wobei

- $E$ der Elastizitätsmodul ist (z.B. in N/mm$^2$),
- $\rho$ die Dichte (z.B. in kg/mm$^3$).

Die untere Schranke der Eigenfrequenz $f_{\min}$ ist näherungsweise

$$
f_{\min} \approx \frac{c}{2 L_c}
$$

wobei $L_c$ die charakteristische Länge ist.
