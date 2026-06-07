# ADR-0003: Objetivo — predecir propina baja (clase minoritaria)

**Estado**: Aceptada
**Fecha**: 2026-06-07
**Decisor**: Niels Pacheco

## Contexto

La tarea de clasificación debía (a) tener valor de negocio, (b) presentar el
desbalance de clases que el curso enfatizó (donde la *accuracy* engaña) y (c)
evitar fuga de objetivo. El análisis exploratorio mostró que, entre los viajes
pagados con tarjeta, ~75% dejan propina ≥20% (efecto de las sugerencias por
defecto de la pantalla de pago), por lo que un umbral "propina alta ≥20%" da una
clase mayoritaria poco interesante.

## Decisión

Definir la etiqueta positiva como **propina baja**: `tip_amount / fare_amount <
0.10`. Esta clase es minoritaria (~9.4% de los viajes con tarjeta), generando un
**desbalance ~9.7:1**. Para garantizar veracidad, el análisis se restringe a
pagos con tarjeta (`payment_type == 1`), pues las propinas en efectivo no se
registran. Se EXCLUYEN `tip_amount` y `total_amount` como predictores (fuga).

## Opciones consideradas

- **Propina baja (<10%)** — Minoritaria (~9%), desbalance genuino, valor de
  negocio (anticipar viajes de baja recompensa para el conductor/despacho).
- **Propina alta (≥20%)** — Clase mayoritaria (~75%), poco informativa.
- **Propina muy alta (≥30%)** — ~28%, desbalance moderado pero menor señal.

## Consecuencias

**Positivas**: escenario desbalanceado realista que obliga a usar AUC y recall
de la minoría; permite demostrar técnicas de manejo de desbalance
([ADR-0005](0005-manejo-desbalance.md)); narrativa de negocio clara.

**Negativas**: el sesgo de selección (solo tarjeta) limita la generalización a
viajes en efectivo; se documenta como amenaza a la validez externa.
