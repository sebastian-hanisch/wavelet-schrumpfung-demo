# Wavelet-Schrumpfung: Schwellwerte entscheiden – bei Rauschabstand 7 ist die universale Schwelle zu hoch, SURE und Cycle Spinning senken den Fehler – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-wavelet-schrumpfung-demo.streamlit.app/)**

Dreizehnte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf der Demo zu den **diskreten Wavelets**. Die vier
Testsignale von Donoho und Johnstone (Blocks, Bumps, HeaviSine, Doppler) werden verrauscht und per **Wavelet-Schrumpfung** entrauscht: DWT, Schwelle auf den Detailkoeffizienten, Rücktransformation. Gemessen werden die
Schwellenverfahren (universal, SURE je Stufe, Hybrid, Orakel), der Fehler als Funktion der Schwelle, **Cycle Spinning**, die Kompression (beste k-Term-Näherung gegen Fourier) und der Rauschpegel.

## Warum dieses Problem

Rauschen verteilt sich in der orthogonalen Wavelet-Basis gleichmäßig auf alle Koeffizienten, ein glattes oder stückweise glattes Signal sitzt in wenigen großen: Eine Schwelle trennt beides. Die Frage ist die Höhe. Die universale
Schwelle σ√(2 ln n) nullt bei reinem Rauschen mit großer Wahrscheinlichkeit alle Rauschkoeffizienten (gemessen: in etwa 82 % der Läufe), ist aber bei Rauschabstand 7 oft zu hoch; SURE schätzt das Risiko aus den Daten und wählt je Stufe; die DWT ist verschiebungsvariant, was Cycle Spinning ausgleicht. Die Demo misst das
gegen die Wahrheit.

## Modell

Signale Blocks, Bumps, HeaviSine und Doppler (Formeln nach Donoho und Johnstone 1994) auf n Punkten, auf die Standardabweichung des Signal-Rausch-Verhältnisses skaliert, plus weißes Rauschen N(0, 1). Periodische DWT mit Daubechies-Filtern
(Haar bis D16), J Stufen; geschwellt werden nur die Details, die Glättung bleibt. Weiche Schwelle sign(d)(|d| − λ)₊, harte d·1{|d| > λ}. Schwellen: universal σ√(2 ln n), SURE je Stufe (Minimum von Steins Risikoschätzer über
{0} ∪ {|z|}, höchstens √(2 ln m)), Hybrid (SureShrink: universal bei Stufen mit (Σz² − m)/m ≤ (log₂ m)^(3/2)/√m, sonst SURE), feste Schwelle. Rauschpegel σ̂ = Median(|d₁|)/0,6745 oder bekannt. Cycle Spinning mittelt über
min(2^J, 64) zirkuläre Verschiebungen. Alles selbst in numpy; scipy dient nur als Gegenprobe in den Tests.

## Methodik

n = 256 bis 4096 (Experimente: 1024), Signal-Rausch-Verhältnis 1 bis 20 (Experimente: 7), D8 (p = 4), 6 Stufen. Fehler = mittlere quadratische Abweichung je Abtastwert gegen die Wahrheit; „verrauscht“ ist der Fehler ohne Schwelle (etwa 1).
Experimente über feste Zufallsströme (Seed 970000): 100 Läufe bei den Schwellenverfahren, 60 bei der Fehlerkurve, 40 beim Cycle Spinning, 200 beim Rauschpegel; die Kompression ist ohne Rauschen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie gut sind die Schwellenverfahren (Fehler gegen die Wahrheit, σ aus der Stufe 1 geschätzt)? | Verrauscht / universal weich / universal hart / SURE / Hybrid / beste globale Schwelle (Orakel): Blocks 1,007 / 1,662 / 0,516 / 0,365 / 0,513 / 0,419; Bumps 1,008 / 2,011 / 0,534 / 0,395 / 0,457 / 0,450; HeaviSine 1,005 / 0,185 / 0,122 / 0,105 / 0,142 / 0,111; Doppler 0,998 / 0,757 / 0,239 / 0,229 / 0,285 / 0,260. **Die weiche universale Schwelle verschlechtert Blocks und Bumps sogar; Bei Rauschabstand 7 liegt SURE im Mittel auf allen vier Signalen unter dem Orakel mit der besten gemeinsamen Schwelle (bei HeaviSine nur knapp).** |
| Wie hängt der Fehler von der Schwelle ab (weich, global, σ = 1 bekannt)? | Die beste Schwelle liegt bei 1,0 (Blocks), 1,0 (Bumps), 1,8 (HeaviSine) und 1,4 (Doppler) mit den Fehlern 0,414 / 0,443 / 0,111 / 0,268; bei der universalen Schwelle (3,72, nächster Gitterpunkt 3,8) sind es 1,586 / 1,897 / 0,186 / 0,750. Das Minimum ist flach, die Lagen sind nur auf etwa ± 0,2 bestimmt (Gitter 0,2). Ohne Schwelle ist es das Rauschen (0,997 bis 1,005). |
| Was bringt Cycle Spinning (alle 64 Verschiebungen)? | Fehler der universalen Schwelle ohne / mit: D8 hart: Blocks 0,521 / 0,231, HeaviSine 0,126 / 0,059, Doppler 0,249 / 0,119; D8 weich: 1,676 / 1,340, 0,190 / 0,144, 0,755 / 0,547; Haar hart: Blocks 0,099 / 0,052, HeaviSine 0,337 / 0,083, Doppler 0,964 / 0,277. Der größte Fehler im Fenster von ± 20 Abtastwerten um den Sprung sinkt bei D8 hart von 3,26 auf 2,12 (Blocks), 2,66 auf 2,21 (HeaviSine) und 0,41 auf 0,30 (Doppler); bei Haar und Blocks bleibt er (0,89 gegen 0,87). |
| Wie dünn sind die Signale (beste k-Term-Näherung, relativer L2-Fehler, k = 8 / 16 / 32 / 64 / 128)? | Blocks: Haar 0,471 / 0,300 / 0,171 / 0,000 / 0,000 (die Stufen sind in Haar exakt dünn), D8 0,496 / 0,352 / 0,197 / 0,111 / 0,031, Fourier 0,421 / 0,354 / 0,252 / 0,166 / 0,116. Koeffizienten für 99 % der Energie (D8 / Fourier): Blocks 70 / 168, Bumps 107 / 410, HeaviSine 12 / 9, Doppler 40 / 175. Nach dem Kriterium „99 % der Energie“ ist nur HeaviSine in Fourier dünner (9 gegen 12); nach dem k-Term-Fehler hängt die Reihenfolge von k ab. |
| Wie gut ist der Rauschpegel aus dem Median der Stufe 1? | σ̂ bei Blocks 1,051 ± 0,060, Bumps 1,060 ± 0,056, HeaviSine 1,002 ± 0,050, Doppler 1,025 ± 0,050 (wahr 1). Mit einem falschen σ (0,5 / 0,75 / 1 / 1,5 / 2 mal das wahre) ist der Fehler bei Blocks (universal weich) 0,58 / 1,01 / 1,53 / 2,73 / 3,94. Bei reinem Rauschen nullt die universale Schwelle alle Details in 81,3 / 82,7 / 81,7 % der Läufe (n = 256 / 1024 / 4096; Näherung 76,0 / 78,6 / 80,4 %; 300 Läufe, Standardfehler etwa 2 Punkte, σ = 1 bekannt, D4, 4 Stufen). |
| Ist SURE unverzerrt? | Für eine feste Schwelle ja: Bei 20 großen Koeffizienten unter 200 und Rauschen N(0, 1) stimmen der mittlere wahre Fehler und das mittlere SURE auf 1 % überein (in den Tests, 1500 bis 3000 Läufe). Für die per Minimum von SURE gewählte Schwelle ist SURE dagegen zu optimistisch. |

## Ehrliche Grenzen

- **Heimvorteil:** Vier Standardsignale mit bekanntem weißem Rauschen; echte Signale haben unbekannte, oft farbige Störungen.
- **Rauschabstand 7 ist hoch:** Die Rangfolge der Schwellen hängt vom Rauschabstand ab; die Experimente gelten für 7. Bei niedrigem Rauschabstand (die App erlaubt 1) ist jede Schwelle weit besser als keine (Doppler bei Rauschabstand 1, ein Lauf mit Seed 39: 0,157 gegen 0,955 ohne Schwelle); die universale Schwelle liegt dort hinter SURE bei Blocks, Bumps und Doppler, vor SURE bei HeaviSine.
- **Das Orakel ist eine globale Schwelle:** SURE schlägt es vermutlich, weil es je Stufe wählt (nicht isoliert gemessen); das ist kein Beleg, dass SURE das beste mögliche Verfahren ist.
- **Periodische Fortsetzung:** Blocks hat am Rand einen Sprung; bei nichtperiodischen Signalen entstehen Randeffekte (nicht gemessen).
- **Cycle Spinning:** mittelt höchstens über 64 Verschiebungen (bei mehr als 6 Stufen nicht über alle); die Gibbs-Artefakte sind nur am größten Fehler im Fenster gemessen (das Maß vermischt sie mit dem Bias der weichen Schwelle), nicht visuell.
- **Weißes Rauschen mit gemeinsamer Varianz:** Farbiges Rauschen braucht eine Schwelle je Stufe (nicht gebaut), ebenso Bayes-Schrumpfung, Block-Thresholding und 2D-Entrauschung.
- **Quellen** (Donoho und Johnstone 1994 und 1995; Coifman und Donoho 1995; Stein 1981): nicht gelesen; Formeln per Simulation, Handrechnung und gegen geschlossene Ausdrücke geprüft.

## Verifikation

- **Kern gegen Handrechnung:** Schwellen (weich, hart) an Zahlen; SURE für d = (3, −1; 0,5; 2): SURE(0) = 4, SURE(0,5) = 3, SURE(1) = 3,25, SURE(3) = 10,25, Minimum bei 0,5; Haar-Mini-Instanz x = (1, 2, 3, 4) mit der Schwelle 0,8 liefert die Glättung allein (1,5; 1,5; 3,5; 3,5);
  die Signale an den Rastermitten (Blocks 4 / 3 / 0,9 / 3, HeaviSine 4 / −6 / 2 / −4).
- **Orakel (`tests/test_oracle_*.py`):** Filter gegen die Formel und Bedingungen, DWT gegen die Summendefinition und als Batch gegen Einzelreihen, SURE gegen die direkte Schleife und gegen den wahren Fehler (unverzerrt), Schwelle null gleich Identität,
  unendliche Schwelle gleich Glättung, Cycle Spinning gleich dem Mittel über die Verschiebungen und verschiebungsinvariant, k-Term-Fehler gegen die Rekonstruktion (Parseval).
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, Stufen je n, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich; hier nur der Zufallsstrom SplitMix64) |
| `shr_wavelet.py` | Daubechies-Filter, periodische DWT und Inverse (auch für mehrere Reihen) |
| `shr_signals.py` | Blocks, Bumps, HeaviSine, Doppler |
| `shr_shrink.py` | Schwellen, universal, SURE, Hybrid, Rauschpegel, Cycle Spinning, k-Term-Näherung |
| `shr_resolution.py` | Fehlermaße, Fehler gegen die Schwelle, Orakel, k-Term-Fehler |
| `shr_constants.py` | Regler-Grenzen, Signale, feste Annahmen, Experimente, Presets |
| `shr_evaluation.py` | Analyse, Experimente, Verdikte |
| `shr_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `shr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `shr_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Signale, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Minimax-Schwellen, Bayes-Schrumpfung, Block-Thresholding:** nicht enthalten.
- **Farbiges Rauschen mit stufenabhängiger Schwelle, 2D-Entrauschung:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf wenige Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
