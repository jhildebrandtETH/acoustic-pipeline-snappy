# Ansys-Mesh-Pipeline für OpenFOAM 13

Eigenständige Kopie der Snappy-Pipeline. Eingabe ist ein fertiges Fluent-Netz,
kein STL. Die ursprüngliche Pipeline im übergeordneten Ordner bleibt unverändert.

## Ablauf

MESH → Formatkonvertierung → unbenutzte Punkte entfernen → checkMesh →
Rotorgrenze trennen → NCC koppeln → checkMesh → bestehender Solver →
bestehende Akustik, Visualisierungen und Reports.

Es werden weder blockMesh noch surfaceFeatures noch snappyHexMesh ausgeführt.
Die Zellkonnektivität des importierten Netzes wird nicht neu vernetzt. Für die
Rotation werden interne Interface-Flächen zu zwei Randflächen und ihre gemeinsamen
Punkte getrennt; die Zahl der Volumenzellen bleibt erhalten.

## Umgebung

Wie bisher unter Linux/WSL mit Docker und `acoustic-pipeline/openfoam:13-20260624` arbeiten.
Eine Fluent-Lizenz ist für den Import nicht erforderlich.
Vor dem ersten Lauf das gepinnte Image aus Linux/WSL bauen:

```sh
bash runtime/build.sh
```

Version, Update und Rücknahme: [Runtime](runtime/README.md).

Die zuletzt getesteten expliziten Omega-Solverkontrollen wurden zurückgenommen.
Neue SST-/DES-Aufträge verwenden wieder die gemeinsamen U/k/Omega-Kontrollen
mit `tolerance 1e-6` ohne erzwungene Mindestiteration. Bestehende Aufträge
behalten ihre gesicherten Vorlagen. Details:
[Omega-Rücknahme](docs/runtime-crash-workarounds.md#omega-rollback-24-october-2026).

Die Umgebungsdatei enthält die Python-Abhängigkeiten für Import, Solversteuerung,
Akustik, Visualisierung, Courant-Diagnostik, Notebooks und Regressionstests.
Eine vorhandene Umgebung aus diesem `ansys_pipeline`-Ordner aktualisieren:

```sh
conda env update -n of_pipeline_env -f of_pipeline_env.yml
conda activate of_pipeline_env
```

`-n` muss dem Namen der tatsächlich verwendeten Umgebung entsprechen; für die
separate Ansys-Umgebung `-n ansys_pipeline_env` verwenden. Die Aktivierung allein
installiert keine neuen Abhängigkeiten. Keine zusätzlichen manuellen pip-Schritte
sind für diese Pipeline erforderlich.

Alternativ aus diesem Ordner eine eigene Umgebung erstellen:

```sh
conda env create -f of_pipeline_env.yml
conda activate ansys_pipeline_env
```

## Eingabeordner

```text
output/meine-ansys-studie/
  MESH/
    mit_schichten.msh.h5
    ohne_schichten.msh.h5
```

Jede Datei wird für alle angeforderten Drehzahlen als eigener Fall gerechnet.
Unterstützt werden Fluent CFF `.msh.h5` und klassische **ASCII** `.msh`-Dateien.
Keine `.cas.h5`, keine Gmsh-Dateien, keine bereits nichtkonform gekoppelten oder
periodischen/adaptierten Netze. Ein mehrfach vorhandener Basisname wird abgelehnt.

Das Netz muss einen geschlossenen Propeller-Wandpatch sowie inlet/outlet/walls,
zwei getrennte Fluidzellzonen und eine interne Face-Zone an der gesamten
Rotorgrenze besitzen. Die Rotorachse ist wie bisher Y durch den Ursprung.
Es wird kein ursprüngliches Propeller-STL benötigt: Die Referenzoberfläche für
Visualisierung wird aus dem importierten Wandpatch erzeugt.

## Einheiten und Namen

`Parameters/ansysMesh.json` enthält die Importvorgaben:

```json
{
  "scale": 0.001,
  "rotorCellZone": "rotaryregion",
  "rotorFaceZone": "rotor_interface",
  "patches": {
    "propeller": "propeller",
    "inlet": "inlet",
    "outlet": "outlet",
    "walls": "walls"
  }
}
```

**scale=0.001 gilt für in Millimetern gespeicherte Koordinaten**, wie bei dem
bereits geprüften Fluent-Netz. Für bereits in Metern gespeicherte Koordinaten
`scale=1.0` verwenden. Es gibt bewusst keine automatische Einheitenvermutung.

Pro Datei kann eine JSON-Datei mit demselben Basisnamen diese Werte überschreiben,
z.B. `MESH/mit_schichten.json`. Nur abweichende Schlüssel müssen darin stehen.
Die Patch-Zuordnung lautet jeweils Zielname: vorhandener Quellname. Der Rotorname
wird für die bestehenden Vorlagen auf `rotaryRegion` vereinheitlicht.

Lokale Tip-Verfeinerungen dürfen zusätzliche Wandzonen `tip_negative` und
`tip_positive` erzeugen. Auch Fluent-Namen wie
`propeller-tip_negative-rotaryregion` bzw. die Variante mit Unterstrichen werden
automatisch erkannt. Nach der Konvertierung fasst `createPatch` diese Flächen
mit der Hauptwand zu `propeller` zusammen, ohne Zellen neu zu vernetzen. Damit
gelten Wandbedingungen, Kräfte und akustische Abtastung für den vollständigen
Propeller einschließlich beider Tips. Die lokalen Zellgrößen bleiben erhalten.
Die Quellnamen und Flächenzahlen stehen in `mesh/import.json` und
`mesh/geometry.json`; Details im Fallprotokoll `log.mergePropellerTips`.
Andere unerwartete Randzonen werden weiterhin abgelehnt. Tip-Flächen müssen
Wände sein und vollständig in der rotierenden Zellzone liegen.

HDF5-Prismenzellflags werden als Quelldiagnostik gespeichert. Sie sind kein Beleg
für vollständige Schichtabdeckung oder ausreichendes y+. Bei ASCII-Eingaben ist
die Schichtinformation unbekannt; die vorhandenen Dicken-/y+-Reports bleiben aktiv.

## Erst Import und Report prüfen

Aus diesem Pipeline-Ordner:

```sh
python main.py --sim-dir output/meine-ansys-studie \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions yes \
  --total-cores 4 --cores-per-case 4 --mesh-only
```

`--mesh-only` bedeutet hier Import, Punktbereinigung, Rotor-Kopplung und
Mesh-Report. Es erzeugt kein neues Volumennetz.

## Vollständige Rechnung starten

Einen neuen Eingabe-/Auftragsordner mit MESH-Dateien verwenden:

```sh
python main.py --sim-dir output/meine-ansys-rechnung \
  --rpms 4000 5000 --mode AMI --turbulence kOmegaSST --wall-functions yes \
  --total-cores 8 --cores-per-case 4 \
  --acoustic-surface impermeable --end-on rev 30
```

Die 30 Umdrehungen sind ein Beispiel für die Laufdauer, keine Empfehlung für
statistische oder akustische Konvergenz. `--aerodynamics-only` deaktiviert wie
bisher die Akustik. Für permeable Akustik muss die gewählte Kugel vollständig
in die bereits feststehende Domäne passen, z.B.
`--acoustic-surface permeable --acoustic-sphere-diameter 2` für die vorbereitete Box.
Der Import vergrößert die Ansys-Domäne nicht automatisch.

MRF, Turbulenzvorlagen, Wandbehandlung, Kernbudget, Feldinitialisierung,
Abbruchkriterien und Slurm verwenden die übernommenen Implementierungen.
Slurm benötigt dieselbe Python-Umgebung einschließlich h5py auf den Rechenknoten.
Mesh-/Geometrie-Parameterstudien sind bei festen Eingaben gesperrt; verschiedene
Netze werden stattdessen als einzelne MESH-Dateien verglichen. Solverparameter-
Studien bleiben verfügbar.

## Wiederaufnahme und Ergebnisse

```sh
python main.py --sim-dir output/meine-ansys-rechnung --resume
python main.py --sim-dir output/meine-ansys-rechnung --status
```

Neue Aufträge sichern Parameters/CoreTemplates in einer eigenen Momentaufnahme.
SHA-256 und aufgelöste Importkonfigurationen werden gespeichert. Veränderte
Eingaben werden beim Wiederaufnehmen abgelehnt. Netz und JSON-Eingabedateien
während eines Auftrags unverändert aufbewahren. Fehlgeschlagene Fallverzeichnisse
werden vor einem Neustart in einem datierten Nachbarordner erhalten.

Pro Fall entstehen wie bisher `sim.foam`, `postProcessing/` und `report/`.
Zusätzlich: `mesh/import.json`, `mesh/geometry.json`, `mesh/mesh-status.json`,
`mesh/ncc-status.json` sowie getrennte Konvertierungs- und Prüflogs.
Die Reports kennzeichnen den Ansys-Import und zeigen keine Snappy-Einstellungen
als vermeintliche Herkunft des Netzes an. Solver-/Akustikeinstellungen bleiben
gegenüber der kopierten Pipeline unverändert; beim Netzschreiben wird zum Schutz
kleiner Flächen eine Koordinatengenauigkeit von 17 Stellen verwendet.

## Prüfungen

```sh
python -m unittest discover -s tests
```

Die Importtests verwenden kleine synthetische Netze; große Fluent-Eingaben und
Simulationsergebnisse werden nicht mit versioniert. Ein zusätzlicher Test lädt
alle neun Vorlagen und erstellt ihre NCC-Kopplung mit OpenFOAM 13:

```sh
OPENFOAM_TEMPLATE_SMOKE=1 python -m unittest discover -s tests -p test_template_ncc.py
```

Dieser Test benötigt Linux/WSL, Docker und das lokal verfügbare Image
`acoustic-pipeline/openfoam:13-20260624`. Die Python-Testabhängigkeiten sind in der oben
beschriebenen Pipeline-Umgebung enthalten.

## Einstellungen und Versionskontrolle

`Parameters/` enthält die Vorgaben für neue Aufträge. Insbesondere Zeitschritt,
Courant-Grenze, Wandbehandlung und Akustik vor dem Start prüfen. Lokale Messpfade
in `Parameters/acousticComparison.json` müssen zur eigenen Umgebung passen.
Bestehende Aufträge verwenden ihre gesicherte Parameterkopie.

Die Akustikeinstellungen entsprechen der normalen Pipeline: vorläufige obere
Diagrammgrenze 5000 Hz und konfigurierbare untere Grenze bei 0,5 BPF.
Der Report berechnet außerdem einen räumlichen Frequenz-Cap aus dem lokalen
k-Feld und der Zellgröße; dieser Wert verändert die Diagrammgrenze nicht.
Details: [räumlicher Frequenz-Cap](Parameters/acoustic-resolution.md).

`Parameters/snappyGeometry.json` behält seinen Namen aus der ursprünglichen
Pipeline. Beim Ansys-Import enthält es weiterhin die Größenangabe für die
akustische Oberfläche sowie die Toleranzen der Volumen- und NCC-Prüfungen;
es löst keine Snappy-Vernetzung aus.

`output/`, `runs/`, Fluent-Eingaben, generierte Netze, Reports und Python-Caches
gehören nicht zum Quellstand. Eingaben unter `output/<studie>/MESH/` separat
aufbewahren. Die Vorlagen enthalten ausschließlich Anfangsfelder und
Solver-/Kopplungswörterbücher; Grenzschichten kommen aus dem Fluent-Netz.

`--allow-bad-mesh` erlaubt das Fortsetzen trotz fehlgeschlagener Qualitätsprüfung.
Die Fehler bleiben im Mesh-Status und in den Prüflogs sichtbar. Das Übergehen
einer Prüfung belegt weder numerische Stabilität noch physikalische Genauigkeit;
die erweiterte checkMesh-Diagnostik und die Solververläufe separat beurteilen.

Akustische Oberflächendaten p/U werden mit **65536 Hz** in festen physikalischen
Zeitabständen gespeichert. Details zu adaptiven Zeitschritten, Volumenfeldausgabe
und bestehenden Fällen: [Acoustic sampling](Parameters/acoustic-sampling.md).

## Temporäre Laufzeit-Workarounds

Neue Fälle deaktivieren den eingebauten yPlus-Monitor und verwenden PCG/DIC für Druck und Druckkorrektur. Hintergrund, Grenzen und genaue Rücknahme: [Runtime crash workarounds](docs/runtime-crash-workarounds.md).
